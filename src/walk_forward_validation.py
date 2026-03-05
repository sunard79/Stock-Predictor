"""
Walk-Forward Validation with Per-Market Models + 3-Day Targets

Trains separate XGBoost + LightGBM models per market group in rolling 15-day windows.
Evaluates 6 strategy variants: 3 blend modes x 2 threshold modes (static/dynamic).

Usage:
    python src/walk_forward_validation.py
"""
import sqlite3
import pandas as pd
import numpy as np
import os
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

DB_PATH = "database/stocks.db"
OUTPUT_DIR = "docs"

# Must match predict_multi_asset_v2.py
ACTIVE_GROUPS = {
    "US": ["SPY", "QQQ"],
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
    'vix_term_structure', 'credit_spread_5d', 'yield_curve_slope',
    'real_yield_proxy', 'dollar_momentum_10d', 'gs_ratio_mom_5d',
    'dow_sin', 'dow_cos', 'moy_sin', 'moy_cos', 'is_month_end',
    # Phase 13 Momentum Overrides
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

STRATEGIES = ['xgb_only', 'blend', 'agreement']
THRESHOLD_MODES = ['static', 'dynamic']


def dynamic_threshold(base_threshold, vix_rank):
    """Adjust threshold based on VIX regime (vectorized)."""
    return np.where(
        vix_rank > 0.7, base_threshold * 1.25,
        np.where(vix_rank < 0.3, base_threshold * 0.75, base_threshold)
    )


def get_features_for_group(group_name):
    features = list(BASE_FEATURES)
    if group_name in GROUPS_WITH_MARKET_ENCODED:
        features.append('market_encoded')
    return features


def evaluate_strategy(xgb_preds, lgb_preds, target_raw, threshold, strategy,
                      vix_rank=None, use_dynamic=False):
    """Evaluate a blending strategy. Returns (correct_mask, decisive_mask)."""
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

    if decisive_mask.sum() == 0:
        return np.array([], dtype=bool), decisive_mask

    correct = (
        ((preds[decisive_mask] > 0) & (target_raw[decisive_mask] > 0)) |
        ((preds[decisive_mask] < 0) & (target_raw[decisive_mask] < 0))
    )
    return correct, decisive_mask


def prepare_data():
    """Load and prepare data."""
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError("Database not found.")

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM features_data_robust", conn)
    conn.close()

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['ticker', 'date'])
    df = df[df['ticker'].isin(ALL_TICKERS)].copy()

    le = LabelEncoder()
    df['market_encoded'] = le.fit_transform(df['market'].fillna('Unknown'))

    # Volatility-Adjusted Target (Sharpe Target) — matches predict_multi_asset_v2.py
    df['daily_vol_20d'] = df.groupby('ticker')['return_1d'].transform(lambda x: x.rolling(20).std())
    expected_3d_vol = df['daily_vol_20d'] * np.sqrt(3)

    df['target_3d_raw'] = df.groupby('ticker')['close'].transform(
        lambda x: x.shift(-3) / x - 1
    )
    df['target_3d_z'] = df['target_3d_raw'] / expected_3d_vol.replace(0, np.nan)
    df = df.replace([np.inf, -np.inf], np.nan)

    # Build test windows
    all_dates = sorted(df['date'].unique())
    start_test_date = pd.Timestamp("2026-01-01")
    test_windows = []
    current_date = start_test_date
    while current_date < all_dates[-1]:
        end_window = current_date + timedelta(days=15)
        test_windows.append((current_date, end_window))
        current_date = end_window

    return df, test_windows


def run_walk_forward():
    df, test_windows = prepare_data()

    print(f"--- Per-Market Walk-Forward Validation ({len(test_windows)} windows, 3-day target) ---")
    print(f"Active tickers: {len(ALL_TICKERS)} across {len(ACTIVE_GROUPS)} groups")
    print(f"Target: Volatility-Adjusted 3-Day Sharpe\n")

    combo_keys = [(s, m) for s in STRATEGIES for m in THRESHOLD_MODES]

    group_results = {
        key: {g: {'correct': 0, 'signals': 0} for g in ACTIVE_GROUPS}
        for key in combo_keys
    }
    reg_return_sums = {
        key: {g: 0.0 for g in ACTIVE_GROUPS}
        for key in combo_keys
    }

    window_results = []

    for win_idx, (start, end) in enumerate(test_windows):
        window_signals = {key: 0 for key in combo_keys}
        window_correct = {key: 0 for key in combo_keys}

        for group_name, tickers in ACTIVE_GROUPS.items():
            features = get_features_for_group(group_name)
            group_df = df[df['ticker'].isin(tickers)].copy()
            # Only use features that exist in the data
            actual_features = [f for f in features if f in group_df.columns]
            group_clean = group_df.dropna(subset=actual_features + ['target_3d_z', 'target_3d_raw', 'vix_rank'])

            train_df = group_clean[group_clean['date'] < start]
            test_df = group_clean[
                (group_clean['date'] >= start) & (group_clean['date'] < end)
            ]

            if test_df.empty or len(train_df) < 500:
                continue

            X_train = train_df[actual_features].astype(float)
            y_train = train_df['target_3d_z'].astype(float)
            X_test = test_df[actual_features].astype(float)
            target_raw = test_df['target_3d_raw'].values
            vix_rank = test_df['vix_rank'].values

            # Train XGBoost
            xgb_model = xgb.XGBRegressor(
                **DEFAULT_XGB_PARAMS, random_state=42, tree_method='hist'
            )
            xgb_model.fit(X_train, y_train, verbose=False)
            xgb_preds = xgb_model.predict(X_test)

            # Train LightGBM
            lgb_model = lgb.LGBMRegressor(**DEFAULT_LGB_PARAMS, random_state=42)
            lgb_model.fit(X_train, y_train)
            lgb_preds = lgb_model.predict(X_test)

            # Evaluate all 6 combos (3 strategies x 2 threshold modes)
            for strategy in STRATEGIES:
                for thr_mode in THRESHOLD_MODES:
                    use_dyn = (thr_mode == 'dynamic')
                    correct, decisive_mask = evaluate_strategy(
                        xgb_preds, lgb_preds, target_raw, DEFAULT_THRESHOLD, strategy,
                        vix_rank=vix_rank, use_dynamic=use_dyn
                    )
                    n_signals = decisive_mask.sum()
                    n_correct = correct.sum() if len(correct) > 0 else 0
                    key = (strategy, thr_mode)
                    group_results[key][group_name]['correct'] += n_correct
                    group_results[key][group_name]['signals'] += n_signals
                    window_signals[key] += n_signals
                    window_correct[key] += n_correct
                    if n_signals > 0:
                        if strategy == 'xgb_only':
                            pred_dir = np.where(xgb_preds[decisive_mask] > 0, 1, -1)
                        else:
                            pred_dir = np.where(
                                (0.5 * xgb_preds + 0.5 * lgb_preds)[decisive_mask] > 0, 1, -1
                            )
                        reg_return_sums[key][group_name] += (
                            pred_dir * target_raw[decisive_mask]
                        ).sum()

        # Per-window summary
        s_key = ('blend', 'static')
        d_key = ('blend', 'dynamic')
        if window_signals[s_key] > 0:
            s_acc = window_correct[s_key] / window_signals[s_key]
            d_acc = window_correct[d_key] / max(window_signals[d_key], 1)
            print(f"  Window {start.date()}: Blend-Static {s_acc:.1%}({window_signals[s_key]}) | "
                  f"Blend-Dynamic {d_acc:.1%}({window_signals[d_key]})")
            window_results.append({
                'window_start': start,
                'blend_static_accuracy': s_acc,
                'blend_dynamic_accuracy': d_acc,
                'signals_static': window_signals[s_key],
                'signals_dynamic': window_signals[d_key],
            })

    if not window_results:
        print("No valid windows found.")
        return

    # --- Strategy Comparison Table ---
    print()
    print("=" * 100)
    print(f"STRATEGY COMPARISON (walk-forward, 3-day Sharpe target)")
    print("=" * 100)
    print(f"  {'':15s} | {'Static Threshold':>40s} | {'Dynamic Threshold':>40s}")
    header = f"  {'Group':15s} | {'XGB':>12s} {'Blend':>12s} {'Agreement':>12s} | {'XGB':>12s} {'Blend':>12s} {'Agreement':>12s}"
    print(header)
    print(f"  {'':-<15s}-|-{'':-<40s}-|-{'':-<40s}")

    total = {key: {'correct': 0, 'signals': 0} for key in combo_keys}

    for group_name in ACTIVE_GROUPS:
        static_parts = []
        dynamic_parts = []
        for strategy in STRATEGIES:
            for thr_mode, parts_list in [('static', static_parts), ('dynamic', dynamic_parts)]:
                key = (strategy, thr_mode)
                r = group_results[key][group_name]
                total[key]['correct'] += r['correct']
                total[key]['signals'] += r['signals']
                if r['signals'] > 0:
                    acc = r['correct'] / r['signals']
                    parts_list.append(f"{acc:.1%}({r['signals']:>3d})")
                else:
                    parts_list.append(f"{'N/A':>5s}({'0':>3s})")
        print(f"  {group_name:15s} | {static_parts[0]:>12s} {static_parts[1]:>12s} {static_parts[2]:>12s} "
              f"| {dynamic_parts[0]:>12s} {dynamic_parts[1]:>12s} {dynamic_parts[2]:>12s}")

    # Overall row
    print(f"  {'':-<15s}-|-{'':-<40s}-|-{'':-<40s}")
    static_parts = []
    dynamic_parts = []
    for strategy in STRATEGIES:
        for thr_mode, parts_list in [('static', static_parts), ('dynamic', dynamic_parts)]:
            key = (strategy, thr_mode)
            t = total[key]
            if t['signals'] > 0:
                acc = t['correct'] / t['signals']
                parts_list.append(f"{acc:.1%}({t['signals']:>3d})")
            else:
                parts_list.append(f"{'N/A':>5s}({'0':>3s})")
    print(f"  {'OVERALL':15s} | {static_parts[0]:>12s} {static_parts[1]:>12s} {static_parts[2]:>12s} "
          f"| {dynamic_parts[0]:>12s} {dynamic_parts[1]:>12s} {dynamic_parts[2]:>12s}")

    # --- Auto-select best combination ---
    min_signals_per_group = 15
    combo_labels = {
        ('xgb_only', 'static'): 'XGBoost Only (Static)',
        ('xgb_only', 'dynamic'): 'XGBoost Only (Dynamic)',
        ('blend', 'static'): 'Blend 50/50 (Static)',
        ('blend', 'dynamic'): 'Blend 50/50 (Dynamic)',
        ('agreement', 'static'): 'Agreement Only (Static)',
        ('agreement', 'dynamic'): 'Agreement Only (Dynamic)',
    }
    valid_combos = []
    for key in combo_keys:
        meets_min = all(
            group_results[key][g]['signals'] >= min_signals_per_group
            for g in ACTIVE_GROUPS
        )
        t = total[key]
        if t['signals'] > 0:
            acc = t['correct'] / t['signals']
            valid_combos.append((key, acc, meets_min))

    candidates = [(k, a) for k, a, m in valid_combos if m]
    if not candidates:
        candidates = [(k, a) for k, a, m in valid_combos]

    if candidates:
        best_key, best_acc = max(candidates, key=lambda x: x[1])
        print(f"\n  BEST: {combo_labels[best_key]} at {best_acc:.1%}")

    # --- Avg return per signal for best combo ---
    if candidates:
        best_ret = sum(reg_return_sums[best_key][g] for g in ACTIVE_GROUPS)
        best_signals = total[best_key]['signals']
        if best_signals > 0:
            print(f"  Avg return per signal: {best_ret / best_signals * 100:+.2f}%")

    # --- Visualization ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if window_results:
        results_summary = pd.DataFrame(window_results)
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=results_summary, x='window_start', y='blend_static_accuracy',
                     marker='o', label='Blend (Static)')
        sns.lineplot(data=results_summary, x='window_start', y='blend_dynamic_accuracy',
                     marker='s', label='Blend (Dynamic)')
        plt.axhline(y=0.5, color='r', linestyle='--', label='Random Chance')
        plt.title("Per-Market Model Accuracy Over Time (3-Day Target, 15-Day Windows)")
        plt.ylabel("Accuracy Rate")
        plt.xlabel("Window Start Date")
        plt.ylim(0.3, 1.0)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(OUTPUT_DIR, "walk_forward_accuracy.png"))
        print(f"\nVisualization saved to {OUTPUT_DIR}/walk_forward_accuracy.png")


if __name__ == "__main__":
    run_walk_forward()
