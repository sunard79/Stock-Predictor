import pandas as pd
import sqlite3
import os
from ta.momentum import RSIIndicator
from ta.trend import MACD
import warnings
warnings.filterwarnings('ignore')

DB_PATH = "database/stocks.db"
DASHBOARD_DATA = "data/processed/multi_asset_dashboard.csv"

def analyze_failures():
    df = pd.read_csv(DASHBOARD_DATA)
    df['date'] = pd.to_datetime(df['date'])
    
    # Re-calculate indicators since they aren't in the raw CSV
    processed_dfs = []
    for ticker, group in df.groupby('ticker'):
        group = group.copy().sort_values('date')
        group['rsi'] = RSIIndicator(close=group['close'], window=14).rsi()
        processed_dfs.append(group)
    df = pd.concat(processed_dfs)

    # Convert target dates to datetime objects for reliable comparison
    target_dates = pd.to_datetime(['2026-02-24', '2026-02-26', '2026-02-13', '2026-02-17', '2026-02-25'])
    target_tickers = ['^N225', '^VIX']
    
    analysis_df = df[(df['date'].isin(target_dates)) & (df['ticker'].isin(target_tickers))]
    
    print("--- Detailed Feature Analysis for WRONG Predictions ---")
    
    for _, row in analysis_df.iterrows():
        print(f"Date: {row['date'].date()} | Ticker: {row['ticker']}")
        rsi_val = f"{row['rsi']:.1f}" if not pd.isna(row['rsi']) else "N/A"
        print(f" - Technicals: RSI: {rsi_val} | MA7/MA30: {row['ma_7']:.2f}/{row['ma_30']:.2f}")
        print(f" - Momentum: 1d Return: {row['daily_return']:.2%}")
        print(f" - Sentiment: Ticker Score: {row['sentiment_score']:.2f}")
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        query = """
            SELECT na.title, res.sentiment_label 
            FROM news_articles na
            JOIN news_analysis res ON na.id = res.article_id
            JOIN news_to_tickers nt ON na.id = nt.news_id
            WHERE nt.ticker = ? AND date(na.published_date) = ?
        """
        cursor.execute(query, (row['ticker'], str(row['date'].date())))
        news = cursor.fetchall()
        conn.close()
        
        if news:
            print(" - Associated News:")
            for title, sent in news:
                print(f"   * [{sent.upper()}] {title[:80]}...")
        else:
            print(" - No specific news found for this ticker/day.")
        print("-" * 50)

if __name__ == "__main__":
    analyze_failures()
