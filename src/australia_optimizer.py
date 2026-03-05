import sqlite3
import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

DB_PATH = "database/stocks.db"
BASELINE_ACCURACY = 0.526

def build_australia_features(df):
    """Adds China and Commodity specific features to Australian data."""
    df['date'] = pd.to_datetime(df['date'])
    
    # Extract China Proxy Data (MCHI)
    china_df = df[df['ticker'] == 'MCHI'][['date', 'return_1d', 'return_7d']].rename(
        columns={'return_1d': 'china_ret_1d', 'return_7d': 'china_ret_7d'}
    )
    
    # Extract China Sentiment
    try:
        conn = sqlite3.connect(DB_PATH)
        sector_sent = pd.read_sql_query("SELECT date, sector, sector_sentiment_momentum FROM sector_sentiment_daily WHERE sector='china'", conn)
        sector_sent['date'] = pd.to_datetime(sector_sent['date'])
        china_sent = sector_sent[['date', 'sector_sentiment_momentum']].rename(columns={'sector_sentiment_momentum': 'china_sentiment_trend'})
        conn.close()
    except:
        china_sent = pd.DataFrame(columns=['date', 'china_sentiment_trend'])
    
    # Extract Commodity Proxy Data (Average of GLD, USO, SLV)
    comm_df = df[df['ticker'].isin(['GLD', 'USO', 'SLV'])]
    comm_agg = comm_df.groupby('date').agg(
        comm_ret_1d=('return_1d', 'mean'),
        comm_ret_7d=('return_7d', 'mean')
    ).reset_index()
    
    # Filter for Australia
    aus_df = df[df['market'] == 'AUS'].copy()
    
    # Merge new regional features
    aus_df = pd.merge(aus_df, china_df, on='date', how='left')
    aus_df = pd.merge(aus_df, comm_agg, on='date', how='left')
    if not china_sent.empty:
        aus_df = pd.merge(aus_df, china_sent, on='date', how='left')
    else:
        aus_df['china_sentiment_trend'] = 0.0
        
    return aus_df.dropna()

def optimize_australia():
    if not os.path.exists(DB_PATH): return

    conn = sqlite3.connect(DB_PATH)
    print("Loading Global Dataset...")
    df = pd.read_sql_query("SELECT * FROM features_data", conn)
    conn.close()

    print("Engineering Australia-Specific Features (China Demand & Commodities)...")
    aus_features_df = build_australia_features(df)
    
    # Define specialized feature set
    aus_features = [
        'return_1d', 'return_3d', 'return_7d', 'rsi', 'macd_diff',
        'volatility_7d', 'sentiment_score', 'vix_return',
        'china_ret_1d', 'china_ret_7d', 'comm_ret_1d', 'comm_ret_7d',
        'china_sentiment_trend'
    ]
    
    # Train/Test Split (Validate on February)
    feb_start = pd.Timestamp("2026-02-01")
    train_df = aus_features_df[aus_features_df['date'] < feb_start]
    test_df = aus_features_df[aus_features_df['date'] >= feb_start]
    
    if test_df.empty or train_df.empty:
        print("Not enough data for validation.")
        return

    X_train = train_df[aus_features].astype(float)
    y_train = train_df['target_z'].astype(float)
    X_test = test_df[aus_features].astype(float)
    
    # Model tuned for regional, commodity-heavy indices
    print(f"Training Australia-Dedicated XGBoost Model ({len(X_train)} samples)...")
    model = xgb.XGBRegressor(
        n_estimators=300, 
        learning_rate=0.015, 
        max_depth=5,
        min_child_weight=3, 
        gamma=0.2, 
        subsample=0.8, 
        random_state=42
    )
    model.fit(X_train, y_train)
    
    test_df['pred_z'] = model.predict(X_test)
    
    # Grade signals
    def grade(row):
        if row['pred_z'] > 0.15 and row['target_raw'] > 0: return "CORRECT (Bull)"
        if row['pred_z'] < -0.15 and row['target_raw'] < 0: return "CORRECT (Bear)"
        if abs(row['pred_z']) <= 0.15: return "NEUTRAL"
        return "WRONG"

    test_df['result'] = test_df.apply(grade, axis=1)
    
    decisive = test_df[test_df['result'] != "NEUTRAL"]
    correct = len(decisive[decisive['result'].str.startswith("CORRECT")])
    total = len(decisive)
    accuracy = correct / total if total > 0 else 0
    
    print()
    print("="*60)
    print("AUSTRALIAN MARKET OPTIMIZATION REPORT")
    print("="*60)
    print(f"Baseline General Model Accuracy: {BASELINE_ACCURACY:.1%}")
    print(f"Specialized Model Accuracy:      {accuracy:.1%} ({total} trades)")
    
    if accuracy > BASELINE_ACCURACY:
        print()
        print(f"[SUCCESS] The specialized model outperformed the baseline by {(accuracy - BASELINE_ACCURACY)*100:.1f}%.")
        print("Recommendation: Use this isolated model for ^AXJO and EWA execution.")
    else:
        print()
        print("[FAILED] The specialized model did not beat the general global model.")
    
    print()
    print("--- Why Australia is Different (Top Drivers) ---")
    importances = pd.Series(model.feature_importances_, index=aus_features).sort_values(ascending=False)
    print(importances.head(6).to_string())
    
    print()
    print("Documentation:")
    print("The Australian market (^AXJO) behaves more like a commodity index than a US tech index.")
    print("By forcing the model to explicitly weight China's 7-day momentum and global commodity")
    print("returns, the AI stops treating Australia like a 'US Echo' and respects its unique mechanics.")

if __name__ == "__main__":
    optimize_australia()
