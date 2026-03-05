import sqlite3
import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

DB_PATH = "database/stocks.db"

def run_synced_predictions():
    if not os.path.exists(DB_PATH):
        print("Database not found.")
        return

    conn = sqlite3.connect(DB_PATH)
    print("Loading robust feature dataset...")
    df = pd.read_sql_query("SELECT * FROM features_data_robust", conn)
    conn.close()

    if df.empty:
        print("Feature table is empty. Run src/feature_engineering_robust.py first.")
        return

    df['date'] = pd.to_datetime(df['date'])
    
    # 1. Setup Model (Same as production v2)
    le = LabelEncoder()
    df['market_encoded'] = le.fit_transform(df['market'].fillna('Unknown'))
    
    features = [
        'return_1d', 'return_3d', 'return_7d', 'return_14d', 'return_30d',
        'rsi', 'macd_diff', 'volatility_7d', 'atr_14', 'bb_high_dist', 'bb_low_dist',
        'vol_roc_10', 'obv_trend', 'corr_30d', 'vix_rank', 'rel_strength_spy',
        'sentiment_score', 'sector_avg_sentiment', 'sector_divergence', 'market_encoded'
    ]
    
    df['target_raw'] = df.groupby('ticker')['return_1d'].shift(-1)
    df = df.replace([np.inf, -np.inf], np.nan)
    df_clean = df.dropna(subset=features + ['target_raw'])
    
    X = df_clean[features].astype(float)
    y = df_clean['target_raw'].astype(float)
    
    split_idx = int(len(df_clean) * 0.9)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    y_mean, y_std = y_train.mean(), y_train.std()
    y_train_zscore = (y_train - y_mean) / y_std
    y_test_zscore = (y_test - y_mean) / y_std
    
    print(f"Training XGBoost on {len(X_train)} robust samples...")
    model = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.02, max_depth=6, 
        min_child_weight=5, gamma=0.1, subsample=0.7, 
        colsample_bytree=0.7, random_state=42, tree_method='hist'
    )
    model.fit(X_train, y_train_zscore, eval_set=[(X_test, y_test_zscore)], verbose=False)

    # 2. Generate Latest Predictions
    latest_indices = df.groupby('ticker')['date'].idxmax()
    latest_data = df.loc[latest_indices].copy()
    X_latest = latest_data[features].astype(float)
    latest_data['pred_zscore'] = model.predict(X_latest)
    
    # 3. Australian Consistency Check
    try:
        axjo_pred = latest_data[latest_data['ticker'] == '^AXJO']['pred_zscore'].values[0]
        ewa_pred = latest_data[latest_data['ticker'] == 'EWA']['pred_zscore'].values[0]
        
        divergence = abs(axjo_pred - ewa_pred)
        avg_pred = (axjo_pred + ewa_pred) / 2
        
        australia_signal = 'BUY' if avg_pred > 0.15 else 'SELL' if avg_pred < -0.15 else 'NEUTRAL'
        
        print("\n" + "="*40)
        print("AUSTRALIAN CONSISTENCY CHECK")
        print("="*40)
        print(f"  ^AXJO prediction : {axjo_pred:+.2f}")
        print(f"  EWA prediction   : {ewa_pred:+.2f}")
        
        if divergence > 0.3:
            print(f"  WARNING: Australian assets diverged by {divergence:.2f}")
            print(f"  Action: Averaging signals to {avg_pred:+.2f}")
            latest_data.loc[latest_data['ticker'] == '^AXJO', 'pred_zscore'] = avg_pred
            latest_data.loc[latest_data['ticker'] == 'EWA', 'pred_zscore'] = avg_pred
        else:
            print(f"  Market assets are consistent (diff: {divergence:.2f})")
            
        print(f"  Final Combined Signal: {australia_signal}")
        print("="*40)
    except Exception as e:
        print(f"\nWarning: Could not perform Australian consistency check: {e}")

    # 4. Display Final Execution Sheet
    print(f"\n--- SYNCED AI EXECUTION SHEET (Updated: {latest_data['date'].max().strftime('%Y-%m-%d')}) ---")
    
    results = []
    for _, row in latest_data.iterrows():
        pred_z = row['pred_zscore']
        signal = 'BUY' if pred_z > 0.15 else 'SELL' if pred_z < -0.15 else 'NEUTRAL'
        
        results.append({
            "Ticker": row['ticker'],
            "Market": row['market'],
            "Price": f"{row['close']:.2f}",
            "Pred_Z": f"{pred_z:+.2f}",
            "Signal": signal
        })
    
    print(pd.DataFrame(results).sort_values("Pred_Z", ascending=False).to_string(index=False))

if __name__ == "__main__":
    run_synced_predictions()
