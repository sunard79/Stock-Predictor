"""
SHAP Interaction Analysis (Phase 5)

Discovers top feature interactions per market group using SHAP interaction values.
Trains XGBoost models identically to predict_multi_asset_v2.py, then computes
SHAP interaction values to find which feature pairs have the strongest joint effects.

Usage:
    python src/shap_interaction_analysis.py
"""
import sqlite3
import pandas as pd
import numpy as np
import os
import shap
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

DB_PATH = "database/stocks.db"
RESULTS_DIR = "results"
DOCS_DIR = "docs"

ACTIVE_GROUPS = {
    "US": ["SPY", "QQQ", "DIA", "IWM"],
    "Australia": ["^AXJO", "EWA"],
    "Asia": ["EWJ", "^N225", "MCHI", "FXI", "KWEB", "^HSI", "EWY"],
    "Commodities": ["GLD", "SLV", "USO"],
    "Safe Havens": ["TLT", "^VIX"],
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

GROUPS_WITH_MARKET_ENCODED = {"US", "Asia", "Commodities"}

DEFAULT_XGB_PARAMS = {
    'n_estimators': 800,
    'learning_rate': 0.007,
    'max_depth': 4,
    'min_child_weight': 4,
    'gamma': 0.31,
    'subsample': 0.81,
    'colsample_bytree': 0.55,
}

SHAP_SUBSAMPLE = 2000
TOP_N_INTERACTIONS = 5


def get_features_for_group(group_name):
    features = list(BASE_FEATURES)
    if group_name in GROUPS_WITH_MARKET_ENCODED:
        features.append('market_encoded')
    return features


def load_and_prepare_data():
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

    # 3-day forward return target (z-scored)
    df['target_3d_raw'] = df.groupby('ticker')['close'].transform(
        lambda x: x.shift(-3) / x - 1
    )
    df['return_3d_hist'] = df.groupby('ticker')['close'].transform(
        lambda x: x / x.shift(3) - 1
    )
    rolling_std_3d = df.groupby('ticker')['return_3d_hist'].transform(
        lambda x: x.rolling(30).std()
    )
    df['target_3d_z'] = df['target_3d_raw'] / rolling_std_3d
    df = df.replace([np.inf, -np.inf], np.nan)

    print(f"Loaded {len(df):,} rows for {len(ALL_TICKERS)} active tickers")
    return df


def analyze_interactions():
    df = load_and_prepare_data()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)

    all_interactions = []

    print(f"\n{'='*70}")
    print("SHAP INTERACTION ANALYSIS (Phase 5)")
    print(f"{'='*70}")

    for group_name, tickers in ACTIVE_GROUPS.items():
        features = get_features_for_group(group_name)
        group_df = df[df['ticker'].isin(tickers)].copy()
        group_clean = group_df.dropna(subset=features + ['target_3d_z', 'target_3d_raw'])

        if len(group_clean) < 100:
            print(f"\n{group_name}: SKIP (only {len(group_clean)} samples)")
            continue

        # Chronological 90/10 split (same as train_all)
        split_idx = int(len(group_clean) * 0.9)
        train = group_clean.iloc[:split_idx]
        test = group_clean.iloc[split_idx:]

        X_train = train[features].astype(float)
        y_train = train['target_3d_z'].astype(float)
        X_test = test[features].astype(float)
        y_test = test['target_3d_z'].astype(float)

        # Train XGBoost
        xgb_model = xgb.XGBRegressor(
            **DEFAULT_XGB_PARAMS, random_state=42, tree_method='hist',
            early_stopping_rounds=50
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # Subsample for SHAP interaction values (computationally expensive)
        if len(X_train) > SHAP_SUBSAMPLE:
            sample_idx = np.random.RandomState(42).choice(
                len(X_train), SHAP_SUBSAMPLE, replace=False
            )
            X_sample = X_train.iloc[sample_idx]
        else:
            X_sample = X_train

        print(f"\n{group_name} ({len(X_sample)} samples, {len(features)} features)")
        print("-" * 50)

        # Compute SHAP interaction values
        # Use booster directly to avoid shap/xgboost version incompatibility
        explainer = shap.TreeExplainer(xgb_model.get_booster())
        shap_interaction = explainer.shap_interaction_values(X_sample.values)
        # shape: (n_samples, n_features, n_features)

        # Mean absolute interaction values
        mean_abs = np.abs(shap_interaction).mean(axis=0)  # (n_features, n_features)

        # Extract top interactions (upper triangle only, exclude diagonal)
        n_feat = len(features)
        interactions = []
        for i in range(n_feat):
            for j in range(i + 1, n_feat):
                interactions.append({
                    'group': group_name,
                    'feature_a': features[i],
                    'feature_b': features[j],
                    'mean_abs_interaction': mean_abs[i, j],
                })

        interactions_df = pd.DataFrame(interactions).sort_values(
            'mean_abs_interaction', ascending=False
        )

        # Print top interactions
        print(f"  Top {TOP_N_INTERACTIONS} feature interactions:")
        for rank, (_, row) in enumerate(interactions_df.head(TOP_N_INTERACTIONS).iterrows(), 1):
            print(f"    {rank}. {row['feature_a']} × {row['feature_b']}: "
                  f"{row['mean_abs_interaction']:.6f}")

        # Also print top 3 main effects (diagonal)
        main_effects = [(features[i], mean_abs[i, i]) for i in range(n_feat)]
        main_effects.sort(key=lambda x: x[1], reverse=True)
        print(f"  Top 3 main effects:")
        for rank, (feat, val) in enumerate(main_effects[:3], 1):
            print(f"    {rank}. {feat}: {val:.6f}")

        all_interactions.append(interactions_df)

        # Create interaction heatmap
        fig, ax = plt.subplots(figsize=(14, 12))
        # Use top 15 features by main effect for readability
        top_feat_idx = sorted(range(n_feat), key=lambda i: mean_abs[i, i], reverse=True)[:15]
        top_feat_names = [features[i] for i in top_feat_idx]
        sub_matrix = mean_abs[np.ix_(top_feat_idx, top_feat_idx)]

        sns.heatmap(sub_matrix, xticklabels=top_feat_names, yticklabels=top_feat_names,
                    cmap='YlOrRd', annot=True, fmt='.4f', ax=ax,
                    square=True, linewidths=0.5)
        ax.set_title(f"SHAP Interaction Values — {group_name}\n(Top 15 features by main effect)")
        plt.tight_layout()
        fig.savefig(os.path.join(DOCS_DIR, f"shap_interactions_{group_name.lower().replace(' ', '_')}.png"),
                    dpi=150)
        plt.close(fig)
        print(f"  Heatmap saved to docs/shap_interactions_{group_name.lower().replace(' ', '_')}.png")

    # Save all interactions to CSV
    if all_interactions:
        full_df = pd.concat(all_interactions, ignore_index=True)
        full_df.to_csv(os.path.join(RESULTS_DIR, "shap_interactions.csv"), index=False)
        print(f"\nFull interaction data saved to {RESULTS_DIR}/shap_interactions.csv")

        # Summary: top 2 interactions per group
        print(f"\n{'='*70}")
        print("SUMMARY: Top 2 Interactions Per Group (candidates for explicit features)")
        print(f"{'='*70}")
        for group_name in ACTIVE_GROUPS:
            group_int = full_df[full_df['group'] == group_name].head(2)
            if group_int.empty:
                continue
            print(f"\n  {group_name}:")
            for _, row in group_int.iterrows():
                print(f"    {row['feature_a']} × {row['feature_b']}: "
                      f"{row['mean_abs_interaction']:.6f}")


if __name__ == "__main__":
    analyze_interactions()
