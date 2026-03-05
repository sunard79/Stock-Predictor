import sqlite3
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

DB_PATH = "database/stocks.db"
DASHBOARD_DATA = "data/processed/multi_asset_dashboard.csv"

SAFE_HAVENS = ['GLD', 'TLT', '^VIX']

def prepare_features(df):
    df = df.sort_values(['ticker', 'date'])
    groups = df.groupby('ticker')
    
    # 1. Price & Technical Lags
    df['return_1d'] = groups['close'].pct_change(1)
    df['return_3d'] = groups['close'].pct_change(3)
    df['return_7d'] = groups['close'].pct_change(7)
    df['ma_diff'] = (df['ma_7'] / df['ma_30']) - 1
    
    # 2. Extract Market Mood (SPY Sentiment)
    market_sent = df[df['ticker'] == 'SPY'][['date', 'sentiment_score']].rename(columns={'sentiment_score': 'market_sentiment'})
    df = pd.merge(df, market_sent, on='date', how='left')
    df['market_sentiment'] = df['market_sentiment'].fillna(0)

    # 3. Inversion Logic for Safe Havens
    # If SPY is bearish (<0) and asset is GLD/TLT/VIX, we flip the sentiment impact
    def adjust_sentiment(row):
        if row['ticker'] in SAFE_HAVENS and row['market_sentiment'] < 0:
            return abs(row['market_sentiment']) # Bearish market = Bullish safe haven
        return row['sentiment_score']

    df['adjusted_sentiment'] = df.apply(adjust_sentiment, axis=1)
    
    # 4. Cross-Asset Signals
    vix_data = df[df['ticker'] == '^VIX'][['date', 'return_1d']].rename(columns={'return_1d': 'vix_return'})
    gold_data = df[df['ticker'] == 'GLD'][['date', 'return_1d']].rename(columns={'return_1d': 'gold_return'})
    df = pd.merge(df, vix_data, on='date', how='left')
    df = pd.merge(df, gold_data, on='date', how='left')
    
    # 5. Target Variable
    df['target_next_return'] = groups['return_1d'].shift(-1)
    
    return df.dropna()

def train_and_predict():
    if not os.path.exists(DASHBOARD_DATA):
        print(f"Error: Dashboard data not found at {DASHBOARD_DATA}")
        return

    df = pd.read_csv(DASHBOARD_DATA)
    df['date'] = pd.to_datetime(df['date'])
    
    print("Preparing features with Inversion Logic...")
    df_features = prepare_features(df)
    
    features = [
        'return_1d', 'return_3d', 'return_7d', 
        'ma_diff', 'volatility_7d', 'sentiment_score', 
        'adjusted_sentiment', 'market_sentiment',
        'vix_return', 'gold_return'
    ]
    
    X = df_features[features]
    y = df_features['target_next_return']
    
    split_idx = int(len(df_features) * 0.9)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training updated model on {len(X_train)} samples...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    mae = mean_absolute_error(y_test, model.predict(X_test))
    
    print()
    print("--- Model Performance (With Inversion Logic) ---")
    print(f"Mean Absolute Error: {mae:.4f}")
    
    latest_indices = df_features.groupby('ticker')['date'].idxmax()
    latest_data = df_features.loc[latest_indices]
    
    print()
    print(f"--- Next Day Predictions (Updated: {latest_data['date'].max().strftime('%Y-%m-%d')}) ---")
    
    results = []
    for _, row in latest_data.iterrows():
        pred_return = model.predict(row[features].values.reshape(1, -1))[0]
        direction = "Bullish" if pred_return > 0 else "Bearish"
        results.append({
            "Ticker": row['ticker'],
            "Last_Price": f"{row['close']:.2f}",
            "Pred_Move": f"{pred_return:+.2%}",
            "Signal": direction
        })
    
    results_df = pd.DataFrame(results).sort_values("Ticker")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    train_and_predict()
