"""
Systematic Backtesting & Model Tuning Pipeline — Per-Market Models + 3-Day Targets
XGBoost + LightGBM with 3 blend strategies.
"""
import sqlite3
import argparse
import json
import os
from datetime import timedelta

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

DB_PATH = "database/stocks.db"
RESULTS_DIR = "results/backtest"

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
    # Macro/regime features
    'vix_term_structure', 'credit_spread_5d', 'yield_curve_slope', 'momentum_divergence_20d',
    'real_yield_proxy', 'dollar_momentum_10d', 'gs_ratio_mom_5d', 'aud_momentum_5d',
    'dow_sin', 'dow_cos', 'moy_sin', 'moy_cos', 'is_month_end',
    # Momentum Overrides & Regime Filters
    'trend_ext_idx', 'momentum_accel', 'vpt_ratio', 'macro_price_div', 'price_vix_corr',
    'asx_spy_alpha', 'asx_global_beta'
]

GROUPS_WITH_MARKET_ENCODED = {"US", "Commodities"}

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

STRATEGIES = ['xgb_only', 'blend', 'agreement']
THRESHOLD_MODES = ['static', 'dynamic']

def dynamic_threshold(base_threshold, vix_rank):
    return np.where(
        vix_rank > 0.7, base_threshold * 1.25,
        np.where(vix_rank < 0.3, base_threshold * 0.75, base_threshold)
    )

REGIMES = {
    "COVID Crash": ("2020-02-01", "2020-06-30"),
    "2020 Recovery": ("2020-07-01", "2020-12-31"),
    "2021 Bull": ("2021-01-01", "2021-12-31"),
    "2022 Rate Hikes": ("2022-01-01", "2022-10-31"),
    "2022 Q4 Bounce": ("2022-11-01", "2023-02-28"),
    "2023 Banking Crisis": ("2023-03-01", "2023-06-30"),
    "2023 H2": ("2023-07-01", "2023-12-31"),
    "2024 AI Rally": ("2024-01-01", "2024-12-31"),
    "2025-2026 Recent": ("2025-01-01", "2026-12-31"),
}

def get_features_for_group(group_name):
    features = list(BASE_FEATURES)
    if group_name in GROUPS_WITH_MARKET_ENCODED:
        features.append('market_encoded')
    return features

def evaluate_strategy(xgb_preds, lgb_preds, target_raw, threshold, strategy,
                      vix_rank=None, use_dynamic=False):
    if use_dynamic and vix_rank is not None:
        thresholds = dynamic_threshold(threshold, vix_rank)
    else:
        thresholds = threshold

    if strategy == 'xgb_only':
        preds = xgb_preds
        decisive_mask = np.abs(preds) > thresholds
    elif strategy == 'blend':
        preds = 0.5 * xgb_preds + 0.5 * lgb_preds
        decisive_mask = np.abs(preds) > thresholds
    elif strategy == 'agreement':
        xgb_buy = xgb_preds > thresholds
        xgb_sell = xgb_preds < -thresholds
        lgb_buy = lgb_preds > thresholds
        lgb_sell = lgb_preds < -thresholds
        decisive_mask = (xgb_buy & lgb_buy) | (xgb_sell & lgb_sell)
        preds = 0.5 * xgb_preds + 0.5 * lgb_preds
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    n_signals = int(decisive_mask.sum())
    if n_signals == 0:
        return 0, 0

    correct = (
        ((preds[decisive_mask] > 0) & (target_raw[decisive_mask] > 0)) |
        ((preds[decisive_mask] < 0) & (target_raw[decisive_mask] < 0))
    )
    return int(correct.sum()), n_signals

class PerMarketBacktester:
    def __init__(self):
        self.df = None
        self.results = {}

    def load_data(self):
        if not os.path.exists(DB_PATH):
            raise FileNotFoundError(f"Database not found at {DB_PATH}")

        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM features_data_robust", conn)
        conn.close()

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['ticker', 'date'])
        df = df[df['ticker'].isin(ALL_TICKERS)].copy()

        le = LabelEncoder()
        df['market_encoded'] = le.fit_transform(df['market'].fillna('Unknown'))

        # --- Volatility-Adjusted Target ---
        df['daily_vol_20d'] = df.groupby('ticker')['return_1d'].transform(lambda x: x.rolling(20).std())
        expected_3d_vol = df['daily_vol_20d'] * np.sqrt(3)
        df['target_3d_raw'] = df.groupby('ticker')['close'].transform(lambda x: x.shift(-3) / x - 1)
        df['target_3d_z'] = df['target_3d_raw'] / expected_3d_vol.replace(0, np.nan)
        
        df = df.replace([np.inf, -np.inf], np.nan)
        self.df = df
        print(f"Loaded {len(df):,} rows. Target: Vol-Adjusted Sharpe.")

    def _train_predict_group(self, group_name, train_df, test_df,
                             xgb_params=None, lgb_params=None, threshold=None):
        if xgb_params is None: xgb_params = DEFAULT_XGB_PARAMS
        if lgb_params is None: lgb_params = DEFAULT_LGB_PARAMS
        if threshold is None: threshold = DEFAULT_THRESHOLD

        features = get_features_for_group(group_name)
        # Ensure all columns exist
        actual_features = [f for f in features if f in train_df.columns]
        
        train_clean = train_df.dropna(subset=actual_features + ['target_3d_z', 'target_3d_raw', 'vix_rank'])
        test_clean = test_df.dropna(subset=actual_features + ['target_3d_z', 'target_3d_raw', 'vix_rank'])

        if len(train_clean) < 100 or test_clean.empty:
            return None

        X_train = train_clean[actual_features].astype(float)
        y_train = train_clean['target_3d_z'].astype(float)
        X_test = test_clean[actual_features].astype(float)

        xgb_model = xgb.XGBRegressor(**xgb_params, random_state=42, tree_method='hist')
        xgb_model.fit(X_train, y_train, verbose=False)
        xgb_preds = xgb_model.predict(X_test)

        lgb_model = lgb.LGBMRegressor(**lgb_params, random_state=42)
        lgb_model.fit(X_train, y_train)
        lgb_preds = lgb_model.predict(X_test)

        importance = dict(zip(actual_features, xgb_model.feature_importances_))

        return {
            'xgb_preds': xgb_preds,
            'lgb_preds': lgb_preds,
            'target_raw': test_clean['target_3d_raw'].values,
            'vix_rank': test_clean['vix_rank'].values,
            'importance': importance
        }

    def run_multi_period_backtest(self, label="baseline"):
        print(f"\n{'='*60}")
        print(f"WALK-FORWARD BACKTEST: {label.upper()} (MOMENTUM + VOL-ADJ)")
        print(f"{'='*60}")

        combo_keys = [(s, m) for s in STRATEGIES for m in THRESHOLD_MODES]
        group_strategy_results = {key: {g: {'correct': 0, 'signals': 0} for g in ACTIVE_GROUPS} for key in combo_keys}
        all_importances = {g: [] for g in ACTIVE_GROUPS}

        for regime_name, (start_str, end_str) in REGIMES.items():
            regime_start, regime_end = pd.Timestamp(start_str), pd.Timestamp(end_str)
            
            # Simple regime evaluation
            r_signals, r_correct = {key: 0 for key in combo_keys}, {key: 0 for key in combo_keys}

            # Train on expanding window, test on regime
            for group_name, tickers in ACTIVE_GROUPS.items():
                group_df = self.df[self.df['ticker'].isin(tickers)]
                train_df = group_df[group_df['date'] < regime_start]
                test_df = group_df[(group_df['date'] >= regime_start) & (group_df['date'] < regime_end)]

                result = self._train_predict_group(group_name, train_df, test_df)
                if result:
                    all_importances[group_name].append(result['importance'])
                    for strategy in STRATEGIES:
                        for thr_mode in THRESHOLD_MODES:
                            use_dyn = (thr_mode == 'dynamic')
                            n_c, n_s = evaluate_strategy(result['xgb_preds'], result['lgb_preds'], result['target_raw'], DEFAULT_THRESHOLD, strategy, result['vix_rank'], use_dyn)
                            key = (strategy, thr_mode)
                            group_strategy_results[key][group_name]['correct'] += n_c
                            group_strategy_results[key][group_name]['signals'] += n_s
                            r_correct[key] += n_c
                            r_signals[key] += n_s

            s_key = ('blend', 'static')
            if r_signals[s_key] > 0:
                print(f"  {regime_name:25s} | Blend-Static: {r_correct[s_key]/r_signals[s_key]:.1%}({r_signals[s_key]})")

        # Top features
        print(f"\n--- Top Features ---")
        for g in ACTIVE_GROUPS:
            if all_importances[g]:
                imp_df = pd.DataFrame(all_importances[g]).mean().sort_values(ascending=False).head(5)
                print(f"  {g:12s}: {', '.join([f'{f}={v:.3f}' for f, v in imp_df.items()])}")

def main():
    bt = PerMarketBacktester()
    bt.load_data()
    bt.run_multi_period_backtest()

if __name__ == "__main__":
    main()
