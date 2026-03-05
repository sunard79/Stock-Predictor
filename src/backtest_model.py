import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

DASHBOARD_DATA = "data/processed/multi_asset_dashboard.csv"
SAFE_HAVENS = ['GLD', 'TLT', '^VIX']

def prepare_features(df):
    df = df.sort_values(['ticker', 'date'])
    groups = df.groupby('ticker')
    
    # 1. Basics
    df['return_1d'] = groups['close'].pct_change(1)
    df['return_3d'] = groups['close'].pct_change(3)
    df['return_7d'] = groups['close'].pct_change(7)
    df['ma_diff'] = (df['ma_7'] / df['ma_30']) - 1
    
    # 2. Market Sentiment (Inversion Logic)
    market_sent = df[df['ticker'] == 'SPY'][['date', 'sentiment_score']].rename(columns={'sentiment_score': 'market_sentiment'})
    df = pd.merge(df, market_sent, on='date', how='left')
    df['market_sentiment'] = df['market_sentiment'].fillna(0)

    def adjust_sentiment(row):
        if row['ticker'] in SAFE_HAVENS and row['market_sentiment'] < 0:
            return abs(row['market_sentiment'])
        return row['sentiment_score']
    df['adjusted_sentiment'] = df.apply(adjust_sentiment, axis=1)
    
    # 3. Cross-Asset
    vix_data = df[df['ticker'] == '^VIX'][['date', 'return_1d']].rename(columns={'return_1d': 'vix_return'})
    gold_data = df[df['ticker'] == 'GLD'][['date', 'return_1d']].rename(columns={'return_1d': 'gold_return'})
    df = pd.merge(df, vix_data, on='date', how='left')
    df = pd.merge(df, gold_data, on='date', how='left')
    
    df['target_next_return'] = groups['return_1d'].shift(-1)
    return df.dropna()

def run_backtest():
    if not os.path.exists(DASHBOARD_DATA): return

    df = pd.read_csv(DASHBOARD_DATA)
    df['date'] = pd.to_datetime(df['date'])
    df_features = prepare_features(df)
    
    features = ['return_1d', 'return_3d', 'return_7d', 'ma_diff', 'volatility_7d', 
                'sentiment_score', 'adjusted_sentiment', 'market_sentiment', 
                'vix_return', 'gold_return']
    
    X = df_features[features]
    y = df_features['target_next_return']
    
    split_idx = int(len(df_features) * 0.9)
    train_X, train_y = X.iloc[:split_idx], y.iloc[:split_idx]
    test_df = df_features.iloc[split_idx:].copy()
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(train_X, train_y)
    
    test_df['predicted_return'] = model.predict(test_df[features])
    test_df['target_next_return'] = test_df['target_next_return'].clip(-0.1, 0.1)
    test_df['signal'] = np.where(test_df['predicted_return'] > 0, 1, 0)
    test_df['strategy_return'] = test_df['signal'] * test_df['target_next_return']
    
    ticker_results = []
    for ticker in test_df['ticker'].unique():
        t_data = test_df[test_df['ticker'] == ticker]
        bh = t_data['target_next_return'].sum()
        strat = t_data['strategy_return'].sum()
        win = (t_data[t_data['signal'] == 1]['target_next_return'] > 0).sum() / (t_data['signal'] == 1).sum() if (t_data['signal'] == 1).sum() > 0 else 0
        
        ticker_results.append({
            "Ticker": ticker,
            "BH_Total": f"{bh:+.2%}",
            "Strat_Total": f"{strat:+.2%}",
            "Win_Rate": f"{win:.1%}",
            "Alpha": f"{(strat - bh):+.2%}"
        })
    
    print()
    print("--- Backtest Results (With Inversion Logic) ---")
    print(pd.DataFrame(ticker_results).sort_values("Alpha", ascending=False).to_string(index=False))
    
    total_strat = test_df.groupby('date')['strategy_return'].mean().sum()
    total_bh = test_df.groupby('date')['target_next_return'].mean().sum()
    
    print()
    print(f"Portfolio Strategy Return: {total_strat:+.2%}")
    print(f"Portfolio Buy & Hold: {total_bh:+.2%}")

if __name__ == "__main__":
    run_backtest()
