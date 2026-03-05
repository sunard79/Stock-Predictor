import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

DB_PATH = "database/stocks.db"

ASIA_TICKERS = ["EWJ", "^N225", "MCHI", "FXI", "KWEB", "^HSI", "EWY", "EPI"]

def analyze_asia_tickers():
    conn = sqlite3.connect(DB_PATH)
    print("Loading 30-feature dataset for Asia analysis...")
    df = pd.read_sql_query("SELECT * FROM features_data WHERE ticker IN ('" + "','".join(ASIA_TICKERS) + "')", conn)
    
    if df.empty:
        print("No Asia data found.")
        conn.close()
        return

    df['date'] = pd.to_datetime(df['date'])
    
    features = [
        'return_1d', 'return_3d', 'return_7d', 'return_14d', 'return_30d',
        'rsi', 'macd_diff', 'volatility_7d', 'atr_14', 'bb_high_dist', 'bb_low_dist',
        'vol_roc_10', 'obv_trend', 'corr_30d', 'vix_rank', 'rel_strength_spy',
        'sentiment_score', 'sector_divergence'
    ]
    
    # Calculate target (1-day forward)
    rolling_std = df.groupby('ticker')['return_1d'].transform(lambda x: x.rolling(30).std())
    df['target_raw'] = df.groupby('ticker')['return_1d'].shift(-1)
    df['target_z'] = df['target_raw'] / rolling_std
    df_clean = df.dropna(subset=features + ['target_z', 'target_raw'])

    feb_start = pd.Timestamp("2026-02-01")
    train_df = df_clean[df_clean['date'] < feb_start]
    test_df = df_clean[df_clean['date'] >= feb_start]

    if test_df.empty:
        print("No validation data found for February.")
        conn.close()
        return

    # Train one model for all Asia to see baseline ticker performance
    model = xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.02, max_depth=6, random_state=42
    )
    model.fit(train_df[features].astype(float), train_df['target_z'].astype(float))
    
    test_df['pred_z'] = model.predict(test_df[features].astype(float))
    
    results = []
    print("\n" + "="*75)
    print("ASIA TICKER ACCURACY (Feb 2026)")
    print("="*75)
    
    for ticker in ASIA_TICKERS:
        t_df = test_df[test_df['ticker'] == ticker]
        if t_df.empty: continue
        
        # Using a fixed signal threshold
        decisive = t_df[abs(t_df['pred_z']) > 0.15]
        
        if not decisive.empty:
            correct = ((decisive['pred_z'] > 0) & (decisive['target_raw'] > 0)) | ((decisive['pred_z'] < 0) & (decisive['target_raw'] < 0))
            accuracy = correct.mean()
            signal_count = len(decisive)
        else:
            accuracy = 0.0
            signal_count = 0
            
        results.append({
            "ticker": ticker,
            "accuracy": round(accuracy * 100, 1),
            "signals": signal_count,
            "mae": round(mean_absolute_error(t_df['target_z'], t_df['pred_z']), 4)
        })

    results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
    print(results_df.to_string(index=False))
    conn.close()

if __name__ == "__main__":
    analyze_asia_tickers()
