import sqlite3
import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

DB_PATH = "database/stocks.db"

def run_confidence_analysis():
    if not os.path.exists(DB_PATH):
        print("Database not found.")
        return

    conn = sqlite3.connect(DB_PATH)
    print("Loading robust feature dataset...")
    df = pd.read_sql_query("SELECT * FROM features_data_robust", conn)
    conn.close()

    if df.empty:
        print("Feature table is empty.")
        return

    df['date'] = pd.to_datetime(df['date'])
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
    
    # Chronological Split
    split_idx = int(len(df_clean) * 0.9)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    y_mean, y_std = y_train.mean(), y_train.std()
    y_train_zscore = (y_train - y_mean) / y_std
    y_test_zscore = (y_test - y_mean) / y_std
    
    print(f"Training model on {len(X_train)} samples...")
    model = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.02, max_depth=6, 
        min_child_weight=5, gamma=0.1, subsample=0.7, 
        colsample_bytree=0.7, random_state=42, tree_method='hist'
    )
    model.fit(X_train, y_train_zscore)

    # 1. Performance Tracking by Confidence Level (on Test Set)
    test_preds_z = model.predict(X_test)
    test_actual_z = y_test_zscore.values
    
    test_eval = pd.DataFrame({
        'pred_z': test_preds_z,
        'actual_z': test_actual_z
    })
    
    test_eval['confidence'] = (test_eval['pred_z'].abs() / 1.5).clip(upper=1.0)
    test_eval['correct'] = ((test_eval['pred_z'] > 0) & (test_eval['actual_z'] > 0)) | \
                           ((test_eval['pred_z'] < 0) & (test_eval['actual_z'] < 0))
    
    def get_bucket(conf):
        if conf > 0.60: return "High"
        if conf > 0.30: return "Medium"
        return "Low"
    
    test_eval['bucket'] = test_eval['confidence'].apply(get_bucket)
    
    print("\n--- Performance by Confidence Level (Backtest) ---")
    for b in ["High", "Medium", "Low"]:
        b_data = test_eval[test_eval['bucket'] == b]
        if not b_data.empty:
            acc = b_data['correct'].mean()
            print(f"{b: <6} Confidence Predictions: {acc:.1%} accuracy ({len(b_data)} samples)")
        else:
            print(f"{b: <6} Confidence Predictions: N/A")

    # 2. Generate Latest Predictions for Today
    latest_indices = df.groupby('ticker')['date'].idxmax()
    latest_data = df.loc[latest_indices].copy()
    X_latest = latest_data[features].astype(float)
    preds_z = model.predict(X_latest)
    
    results = []
    print(f"\n--- CONFIDENCE-WEIGHTED EXECUTION SHEET ({latest_data['date'].max().strftime('%Y-%m-%d')}) ---")
    
    for i, (idx, row) in enumerate(latest_data.iterrows()):
        pz = preds_z[i]
        confidence = min(1.0, abs(pz) / 1.5)
        signal = 'BUY' if pz > 0.15 else 'SELL' if pz < -0.15 else 'NEUTRAL'
        
        bucket = get_bucket(confidence)
        star = "⭐" if bucket == "High" else "  "
        
        results.append({
            "Ticker": row['ticker'],
            "Market": row['market'],
            "Price": f"{row['close']:.2f}",
            "Pred_Z": f"{pz:+.2f}",
            "Signal": signal,
            "Conf %": f"{confidence:.0%}",
            "Bucket": bucket,
            "Flag": star
        })
    
    res_df = pd.DataFrame(results).sort_values("Pred_Z", ascending=False)
    print(res_df[["Flag", "Ticker", "Market", "Price", "Pred_Z", "Signal", "Conf %", "Bucket"]].to_string(index=False))

    print("\n--- Trading Strategy Guide ---")
    print("High (>60%): Trade these aggressively")
    print("Medium (30-60%): Trade with caution")
    print("Low (<30%): Skip or paper trade")

if __name__ == "__main__":
    run_confidence_analysis()
