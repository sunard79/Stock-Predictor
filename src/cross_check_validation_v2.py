import pandas as pd
import numpy as np
import os
import xgboost as xgb
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

DASHBOARD_DATA = "data/processed/multi_asset_dashboard.csv"
SAFE_HAVENS = ['GLD', 'TLT', '^VIX']

def prepare_proof_features(df):
    df = df.sort_values(['ticker', 'date'])
    df['date'] = pd.to_datetime(df['date'])
    
    # 1. Sector Intelligence
    sector_sent = df.groupby(['date', 'market'])['sentiment_score'].mean().reset_index()
    sector_sent = sector_sent.rename(columns={'sentiment_score': 'sector_sentiment'})
    df = pd.merge(df, sector_sent, on=['date', 'market'], how='left')
    
    le = LabelEncoder()
    df['market_encoded'] = le.fit_transform(df['market'])
    
    processed_dfs = []
    for ticker, group in df.groupby('ticker'):
        group = group.copy()
        group['rsi'] = RSIIndicator(close=group['close'], window=14).rsi()
        group['macd_diff'] = MACD(close=group['close']).macd_diff()
        group['return_1d'] = group['close'].pct_change(1)
        group['return_3d'] = group['close'].pct_change(3)
        group['return_7d'] = group['close'].pct_change(7)
        
        rolling_std = group['return_1d'].rolling(window=30).std()
        group['target_z'] = group['return_1d'].shift(-1) / rolling_std
        group['target_raw'] = group['return_1d'].shift(-1)
        processed_dfs.append(group)
    
    df = pd.concat(processed_dfs)
    
    # Global Signals
    market_sent = df[df['ticker'] == 'SPY'][['date', 'sentiment_score']].rename(columns={'sentiment_score': 'market_sentiment'})
    df = pd.merge(df, market_sent, on='date', how='left')
    df['market_sentiment'] = df['market_sentiment'].fillna(0)

    def adjust_sentiment(row):
        if row['market'] in ['CHN', 'HK']: return row['sentiment_score']
        if row['ticker'] in SAFE_HAVENS and row['market_sentiment'] < 0:
            return abs(row['market_sentiment'])
        return row['sentiment_score']
    df['adjusted_sentiment'] = df.apply(adjust_sentiment, axis=1)
    
    vix_data = df[df['ticker'] == '^VIX'][['date', 'return_1d']].rename(columns={'return_1d': 'vix_return'})
    gold_data = df[df['ticker'] == 'GLD'][['date', 'return_1d']].rename(columns={'return_1d': 'gold_return'})
    df = pd.merge(df, vix_data, on='date', how='left')
    df = pd.merge(df, gold_data, on='date', how='left')
    
    return df.dropna()

def run_proof_validation():
    if not os.path.exists(DASHBOARD_DATA): return

    df = pd.read_csv(DASHBOARD_DATA)
    df_features = prepare_proof_features(df)
    
    features = [
        'return_1d', 'return_3d', 'return_7d', 'rsi', 'macd_diff',
        'volatility_7d', 'sentiment_score', 'adjusted_sentiment', 
        'sector_sentiment', 'market_sentiment', 'market_encoded',
        'vix_return', 'gold_return'
    ]
    
    # SPLIT POINT: Exactly 15 days ago
    last_date = df_features['date'].max()
    split_date = last_date - timedelta(days=15)
    
    # Train on history up until 15 days ago
    train_df = df_features[df_features['date'] < split_date]
    # Test on the last 15 days
    test_df = df_features[df_features['date'] >= split_date]
    
    print(f"--- 15-DAY PREDICTION PROOF ---")
    print(f"Training on all data up to {split_date.date()}")
    print(f"Verifying predictions from {split_date.date()} to {last_date.date()}")
    
    model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.02, max_depth=6, random_state=42)
    model.fit(train_df[features].astype(float), train_df['target_z'].astype(float))
    
    test_df['pred_z'] = model.predict(test_df[features].astype(float))
    
    def grade(row):
        # 0.15 threshold for High Confidence
        if row['pred_z'] > 0.15 and row['target_raw'] > 0: return "CORRECT (Bull)"
        if row['pred_z'] < -0.15 and row['target_raw'] < 0: return "CORRECT (Bear)"
        if abs(row['pred_z']) <= 0.15: return "NEUTRAL (Filtered)"
        return "WRONG"

    test_df['result'] = test_df.apply(grade, axis=1)
    
    # Final Table Output
    print("\n--- Verification Table (Sample of decisive signals) ---")
    decisive = test_df[test_df['result'] != "NEUTRAL (Filtered)"]
    if decisive.empty:
        print("No high-conviction signals generated in the last 15 days.")
    else:
        # Show breakdown by market
        market_acc = decisive.groupby('market').apply(
            lambda x: len(x[x['result'].str.startswith("CORRECT")]) / len(x)
        ).sort_values(ascending=False)

        print("\nAccuracy Breakdown by Market (Last 15 Days):")
        for market, acc in market_acc.items():
            count = len(decisive[decisive['market'] == market])
            print(f" - {market: <12}: {acc:.1%} ({count} signals)")

        print("\nSample of Signals:")
        print(decisive[['date', 'ticker', 'pred_z', 'target_raw', 'result']].tail(15).to_string(index=False))
        
        correct = len(decisive[decisive['result'].str.startswith("CORRECT")])
        print(f"\nFinal Overall Accuracy (Last 15 Days): {(correct/len(decisive)):.1%}")
        print(f"Total Decisive Signals: {len(decisive)}")

if __name__ == "__main__":
    run_proof_validation()
