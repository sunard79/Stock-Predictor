import sqlite3
import argparse
import pandas as pd
import numpy as np
import os
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

DB_PATH = "database/stocks.db"

ACTIVE_GROUPS = {
    "US": ["SPY", "QQQ"],
    "Australia": ["^AXJO", "EWA"],
    "Commodities": ["GLD", "SLV", "USO"],
}

ALL_TICKERS = [t for tickers in ACTIVE_GROUPS.values() for t in tickers]

# --- 35+ features including Momentum Overrides and Regime Filters ---
BASE_FEATURES = [
    'return_1d', 'return_3d', 'return_7d', 'return_14d', 'return_30d',
    'rsi', 'macd_diff', 'volatility_7d', 'atr_14', 'bb_high_dist', 'bb_low_dist',
    'vol_roc_10', 'obv_trend', 'corr_30d', 'vix_rank', 'rel_strength_spy',
    'sentiment_score', 'sector_avg_sentiment', 'sentiment_divergence',
    'sector_sentiment_momentum', 'sector_divergence',
    # Macro/regime features
    'vix_term_structure', 'credit_spread_5d', 'yield_curve_slope', 'momentum_divergence_20d',
    'real_yield_proxy', 'dollar_momentum_10d', 'gs_ratio_mom_5d', 'aud_momentum_5d',
    'dow_sin', 'dow_cos', 'moy_sin', 'moy_cos', 'is_month_end',
    # --- New Phase 13 Momentum Overrides ---
    'trend_ext_idx', 'momentum_accel', 'vpt_ratio', 'macro_price_div', 'price_vix_corr',
    'asx_spy_alpha', 'asx_global_beta',
    # US cross-asset features
    'tech_broad_spread_5d', 'size_spread_10d', 'hyg_momentum_5d',
    'equity_bond_rotation', 'us_breadth_5d',
]

GROUPS_WITH_MARKET_ENCODED = {"Commodities"}

DEFAULT_XGB_PARAMS = {
    'n_estimators': 800,
    'learning_rate': 0.007,
    'max_depth': 4,
    'min_child_weight': 4,
    'gamma': 0.31,
    'subsample': 0.81,
    'colsample_bytree': 0.55,
}

DEFAULT_LGB_PARAMS = {
    'n_estimators': 800,
    'learning_rate': 0.007,
    'num_leaves': 31,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.55,
    'verbosity': -1,
}

DEFAULT_THRESHOLD = 0.15

def load_and_prepare_data():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM features_data_robust", conn)
    conn.close()

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['ticker', 'date'])
    df = df[df['ticker'].isin(ALL_TICKERS)].copy()

    # --- Volatility-Adjusted Target (Sharpe Target) ---
    # Captures risk-adjusted move instead of raw percentage
    df['daily_vol_20d'] = df.groupby('ticker')['return_1d'].transform(lambda x: x.rolling(20).std())
    expected_3d_vol = df['daily_vol_20d'] * np.sqrt(3)
    
    df['target_3d_raw'] = df.groupby('ticker')['close'].transform(lambda x: x.shift(-3) / x - 1)
    df['target_3d_z'] = df['target_3d_raw'] / expected_3d_vol.replace(0, np.nan)

    df = df.replace([np.inf, -np.inf], np.nan)
    print(f"Loaded {len(df):,} rows. Target: Volatility-Adjusted 3-Day Sharpe.")
    return df

class PerMarketPredictor:
    def __init__(self, blend_mode='xgb_only'):
        self.blend_mode = blend_mode
        self.models = {}

    def train_all(self, df):
        print(f"\nTRAINING WITH MOMENTUM OVERRIDES & VOL-ADJUSTED TARGETS")
        for group_name, tickers in ACTIVE_GROUPS.items():
            features = list(BASE_FEATURES)
            if group_name in GROUPS_WITH_MARKET_ENCODED: features.append('market_encoded')
            
            group_df = df[df['ticker'].isin(tickers)].copy()
            # Ensure all new features exist
            actual_features = [f for f in features if f in group_df.columns]
            group_clean = group_df.dropna(subset=actual_features + ['target_3d_z'])

            split_idx = int(len(group_clean) * 0.9)
            train, test = group_clean.iloc[:split_idx], group_clean.iloc[split_idx:]

            X_train, y_train = train[actual_features].astype(float), train['target_3d_z'].astype(float)
            X_test, y_test = test[actual_features].astype(float), test['target_3d_z'].astype(float)

            xgb_model = xgb.XGBRegressor(**DEFAULT_XGB_PARAMS, random_state=42, tree_method='hist')
            xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            
            lgb_model = lgb.LGBMRegressor(**DEFAULT_LGB_PARAMS, random_state=42)
            lgb_model.fit(X_train, y_train)

            self.models[group_name] = {'xgb': xgb_model, 'lgb': lgb_model, 'features': actual_features}
            print(f"  {group_name:12s} | Train: {len(train):>5} | Features: {len(actual_features)}")

    def predict(self, df):
        results = []
        for group_name, tickers in ACTIVE_GROUPS.items():
            if group_name not in self.models: continue
            
            m = self.models[group_name]
            for ticker in tickers:
                t_data = df[df['ticker'] == ticker]
                if t_data.empty: continue
                latest = t_data.loc[t_data['date'].idxmax()]
                X = pd.DataFrame([latest[m['features']].astype(float)], columns=m['features'])
                
                xp, lp = m['xgb'].predict(X)[0], m['lgb'].predict(X)[0]
                
                if self.blend_mode == 'agreement':
                    # Signal only if both agree on direction and magnitude
                    if (xp > DEFAULT_THRESHOLD and lp > DEFAULT_THRESHOLD): signal = 'BUY'
                    elif (xp < -DEFAULT_THRESHOLD and lp < -DEFAULT_THRESHOLD): signal = 'SELL'
                    else: signal = 'NEUTRAL'
                    pz = (xp + lp) / 2
                else:
                    pz = xp
                    signal = 'BUY' if pz > DEFAULT_THRESHOLD else 'SELL' if pz < -DEFAULT_THRESHOLD else 'NEUTRAL'

                results.append({'Ticker': ticker, 'Market': group_name, 'Price': f"{latest['close']:.2f}", 
                                'Pred_Z': f"{pz:+.2f}", 'XGB_Z': f"{xp:+.2f}", 'LGB_Z': f"{lp:+.2f}", 
                                'Signal': signal, 'Horizon': '3d'})
        return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--blend-mode', default='xgb_only')
    args = parser.parse_args()

    df = load_and_prepare_data()
    predictor = PerMarketPredictor(blend_mode=args.blend_mode)
    predictor.train_all(df)
    
    results = predictor.predict(df).sort_values("Pred_Z", ascending=False)
    print(f"\n--- AI EXECUTION SHEET ({df['date'].max().strftime('%Y-%m-%d')}) ---")
    print(results.to_string(index=False))

    try:
        conn = sqlite3.connect(DB_PATH)
        results['prediction_date'] = df['date'].max().strftime('%Y-%m-%d')
        results.to_sql("daily_predictions_history", conn, if_exists='append', index=False)
        conn.close()
        print("\nPredictions saved.")
    except Exception as e:
        print(f"DB Error: {e}")

if __name__ == "__main__":
    main()
