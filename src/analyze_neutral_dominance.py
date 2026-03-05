import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import os
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

DB_PATH = "database/stocks.db"

def analyze_neutral_signals():
    if not os.path.exists(DB_PATH):
        print("Database not found.")
        return

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM features_data_robust", conn)
    conn.close()

    if df.empty:
        print("Data empty.")
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
    
    # 1. Re-train/Load Model to get RAW Z-Scores
    df['target_raw'] = df.groupby('ticker')['return_1d'].shift(-1)
    df_clean = df.dropna(subset=features + ['target_raw']).replace([np.inf, -np.inf], np.nan).dropna(subset=features)
    
    X = df_clean[features].astype(float)
    y = (df_clean['target_raw'] - df_clean['target_raw'].mean()) / df_clean['target_raw'].std()
    
    model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.02, max_depth=6, random_state=42)
    model.fit(X, y)

    # 2. Get Today's Raw Scores
    latest_indices = df.groupby('ticker')['date'].idxmax()
    latest_data = df.loc[latest_indices].copy()
    X_latest = latest_data[features].astype(float)
    latest_data['raw_z'] = model.predict(X_latest)
    
    # 3. Investigation
    print("--- NEUTRAL DOMINANCE INVESTIGATION ---")
    avg_abs_z = latest_data['raw_z'].abs().mean()
    near_zero = latest_data[(latest_data['raw_z'] > -0.05) & (latest_data['raw_z'] < 0.05)]
    
    print(f"Total Tickers: {len(latest_data)}")
    print(f"Average Raw Z-Score Strength: {avg_abs_z:.4f} (Threshold is 0.15)")
    print(f"Tickers near absolute zero (<0.05): {len(near_zero)}/22")
    
    # Check Sentiment Status
    real_news = latest_data[latest_data['sentiment_synthetic'] == False]
    print(f"Tickers with REAL news today: {len(real_news)}/22")
    
    # Check Volatility Regime (VIX)
    try:
        vix_data = yf.download("^VIX", period="1d", progress=False)
        vix_val = float(vix_data['Close'].iloc[-1])
        vix_status = "High" if vix_val > 25 else "Low" if vix_val < 15 else "Moderate"
        print(f"VIX Level: {vix_val:.2f} ({vix_status})")
    except:
        print("VIX Level: Could not fetch.")

    # 4. Comparison to Walk-Forward
    signal_count = len(latest_data[latest_data['raw_z'].abs() > 0.15])
    print(f"\nSignals today: {signal_count}")
    print(f"Historical avg: ~52 signals / 15 days = ~3.4 signals per day")
    
    # 5. Verdict Logic
    is_data_issue = len(real_news) < 2
    is_low_vol = avg_abs_z < 0.10
    
    print("\nLikely cause:")
    print(f"{'[X]' if is_data_issue else '[ ]'} Data issue (Missing recent news/sentiment)")
    print(f"{'[X]' if is_low_vol else '[ ]'} Market is directionless (Low model conviction)")
    print(f"{'[ ]'} Model too conservative (Threshold 0.15 is standard)")

    verdict = "EXPECTED" if (is_low_vol or is_data_issue) else "UNEXPECTED"
    print(f"\nNeutral dominance is: {verdict}")
    
    if is_data_issue:
        print("\nRECOMMENDATION: Run 'src/historical_news_mega_fetch.py' for March 3-4 data.")

if __name__ == "__main__":
    analyze_neutral_signals()
