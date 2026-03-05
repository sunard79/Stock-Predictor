import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import os
import warnings
warnings.filterwarnings('ignore')

DB_PATH = "database/stocks.db"

ACTIVE_GROUPS = {
    "US": ["SPY", "QQQ", "DIA", "IWM"],
    "Australia": ["^AXJO", "EWA"],
    "Commodities": ["GLD", "SLV", "USO"],
}

ALL_TICKERS = [t for tickers in ACTIVE_GROUPS.values() for t in tickers]

BASE_FEATURES = [
    'return_1d', 'return_3d', 'return_7d', 'return_14d', 'return_30d',
    'rsi', 'macd_diff', 'volatility_7d', 'atr_14', 'bb_high_dist', 'bb_low_dist',
    'vol_roc_10', 'obv_trend', 'corr_30d', 'vix_rank', 'rel_strength_spy',
    'sentiment_score', 'sector_avg_sentiment', 'sentiment_divergence',
    'sector_sentiment_momentum', 'sector_divergence',
    'vix_term_structure', 'credit_spread_5d', 'yield_curve_slope', 'momentum_divergence_20d',
    'dow_sin', 'dow_cos', 'moy_sin', 'moy_cos', 'is_month_end',
]

def quick_predict():
    if not os.path.exists(DB_PATH):
        print("Database not found.")
        return

    conn = sqlite3.connect(DB_PATH)
    # Using the robust features table
    try:
        df = pd.read_sql_query("SELECT * FROM features_data_robust", conn)
    except:
        df = pd.read_sql_query("SELECT * FROM features_data", conn)
    conn.close()

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['ticker', 'date'])
    df = df[df['ticker'].isin(ALL_TICKERS)].copy()

    # Encode market
    le = LabelEncoder()
    df['market_encoded'] = le.fit_transform(df['market'].fillna('Unknown'))

    # Build 3-day target for training (z-scored)
    df['target_3d_raw'] = df.groupby('ticker')['close'].transform(lambda x: x.shift(-3) / x - 1)
    df['return_3d_hist'] = df.groupby('ticker')['close'].transform(lambda x: x / x.shift(3) - 1)
    rolling_std_3d = df.groupby('ticker')['return_3d_hist'].transform(lambda x: x.rolling(30).std())
    df['target_3d_z'] = df['target_3d_raw'] / rolling_std_3d
    df = df.replace([np.inf, -np.inf], np.nan)

    results = []
    print("\n" + "="*70)
    print("QUICK PREDICTIONS FOR MARCH 6, 2026 (Excluding Asia)")
    print("="*70)

    for group_name, tickers in ACTIVE_GROUPS.items():
        group_df = df[df['ticker'].isin(tickers)].copy()
        features = list(BASE_FEATURES)
        if group_name in ["US", "Commodities"]: features.append('market_encoded')
        
        # Filter features that exist in the DB
        actual_features = [f for f in features if f in group_df.columns]
        
        group_clean = group_df.dropna(subset=actual_features + ['target_3d_z'])
        
        if len(group_clean) < 100:
            print(f"  {group_name}: Insufficient data for training.")
            continue

        # Simple train/test split for this quick prediction
        split_idx = int(len(group_clean) * 0.9)
        train = group_clean.iloc[:split_idx]
        test = group_clean.iloc[split_idx:]

        X_train = train[actual_features].astype(float)
        y_train = train['target_3d_z'].astype(float)
        X_test = test[actual_features].astype(float)
        y_test = test['target_3d_z'].astype(float)

        # Train XGBoost (Simplified params for speed)
        model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=5, random_state=42)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # Get latest data for each ticker in group
        for ticker in tickers:
            t_data = group_df[group_df['ticker'] == ticker]
            if t_data.empty: continue
            
            latest = t_data.loc[t_data['date'].idxmax()]
            X_latest = pd.DataFrame([latest[actual_features].astype(float)], columns=actual_features)
            
            pred_z = model.predict(X_latest)[0]
            
            # Simple static threshold
            threshold = 0.15
            signal = 'BUY' if pred_z > threshold else 'SELL' if pred_z < -threshold else 'NEUTRAL'
            
            results.append({
                'Ticker': ticker,
                'Market': group_name,
                'Signal': signal,
                'Pred_Z': round(float(pred_z), 3)
            })

    if results:
        res_df = pd.DataFrame(results).sort_values('Pred_Z', ascending=False)
        print(res_df.to_string(index=False))
        
        # Save to a temporary text file so I can read it if the shell output fails
        with open("results/march6_predictions_temp.txt", "w") as f:
            f.write(res_df.to_string(index=False))
        print("\nPredictions saved to results/march6_predictions_temp.txt")

if __name__ == "__main__":
    quick_predict()
