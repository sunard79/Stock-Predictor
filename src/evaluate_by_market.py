import sqlite3
import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

DB_PATH = "database/stocks.db"

MARKET_GROUPS = {
    "US": ["SPY", "QQQ", "DIA", "IWM"],
    "Australia": ["^AXJO", "EWA"],
    # "Asia": ["EWJ", "^N225", "MCHI", "FXI", "KWEB", "^HSI", "EWY", "EPI"],
    # "Europe": ["EWU", "EWG"],
    "Commodities": ["GLD", "SLV", "USO"],
    # "Safe Havens": ["TLT", "^VIX"]
}

def get_market_group(ticker):
    for group, tickers in MARKET_GROUPS.items():
        if ticker in tickers:
            return group
    return "Other"

def evaluate_by_market():
    if not os.path.exists(DB_PATH):
        print("Database not found.")
        return

    conn = sqlite3.connect(DB_PATH)
    print("Loading 30-feature dataset for market evaluation...")
    df = pd.read_sql_query("SELECT * FROM features_data", conn)
    
    if df.empty:
        print("Feature table is empty.")
        conn.close()
        return

    df['date'] = pd.to_datetime(df['date'])
    le = LabelEncoder()
    df['market_encoded'] = le.fit_transform(df['market'].fillna('Unknown'))
    
    features = [
        'return_1d', 'return_3d', 'return_7d', 'return_14d', 'return_30d',
        'rsi', 'macd_diff', 'volatility_7d', 'atr_14', 'bb_high_dist', 'bb_low_dist',
        'vol_roc_10', 'obv_trend', 'corr_30d', 'vix_rank', 'rel_strength_spy',
        'sentiment_score', 'sector_divergence', 'market_encoded'
    ]
    
    rolling_std = df.groupby('ticker')['return_1d'].transform(lambda x: x.rolling(30).std())
    df['target_raw'] = df.groupby('ticker')['return_1d'].shift(-1)
    df['target_z'] = df['target_raw'] / rolling_std
    df = df.replace([np.inf, -np.inf], np.nan)
    df_clean = df.dropna(subset=features + ['target_z', 'target_raw'])

    feb_start = pd.Timestamp("2026-02-01")
    train_df = df_clean[df_clean['date'] < feb_start]
    test_df = df_clean[df_clean['date'] >= feb_start]

    if test_df.empty:
        print("No validation data found for February.")
        conn.close()
        return

    model = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.02, max_depth=6, 
        min_child_weight=5, gamma=0.1, subsample=0.7, 
        colsample_bytree=0.7, random_state=42
    )
    model.fit(train_df[features].astype(float), train_df['target_z'].astype(float))
    
    test_df['pred_z'] = model.predict(test_df[features].astype(float))
    test_df['market_group'] = test_df['ticker'].apply(get_market_group)
    
    results = []
    print()
    print("="*75)
    print("PRODUCTION EVALUATION BY MARKET GROUP (Feb 2026)")
    print("="*75)
    
    for group in MARKET_GROUPS.keys():
        group_df = test_df[test_df['market_group'] == group]
        if group_df.empty: continue
        
        decisive = group_df[abs(group_df['pred_z']) > 0.15]
        
        if not decisive.empty:
            correct = ((decisive['pred_z'] > 0) & (decisive['target_raw'] > 0)) | ((decisive['pred_z'] < 0) & (decisive['target_raw'] < 0))
            accuracy = correct.mean()
            signal_count = len(decisive)
        else:
            accuracy = 0.0
            signal_count = 0
            
        mae = mean_absolute_error(group_df['target_z'], group_df['pred_z'])
        
        results.append({
            "market_group": group,
            "date_range": "2026-02-01 to 2026-02-28",
            "accuracy": round(accuracy * 100, 1),
            "signal_count": signal_count,
            "mae": round(mae, 4)
        })

    results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
    print(results_df.to_string(index=False))
    
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS market_performance")
    cursor.execute("""
        CREATE TABLE market_performance (
            market_group TEXT,
            date_range TEXT,
            accuracy REAL,
            signal_count INTEGER,
            mae REAL
        )
    """)
    for r in results:
        cursor.execute("INSERT INTO market_performance VALUES (?,?,?,?,?)", 
                       (r['market_group'], r['date_range'], r['accuracy'], r['signal_count'], r['mae']))
    conn.commit()
    
    print()
    print("--- Attribution: Top Drivers of System Accuracy ---")
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print(importances.head(5).to_string())
    
    conn.close()
    print()
    print("Market results saved to 'market_performance' table.")

if __name__ == "__main__":
    evaluate_by_market()
