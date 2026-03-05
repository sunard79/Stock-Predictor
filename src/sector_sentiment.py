import sqlite3
import pandas as pd
import numpy as np
import os
from datetime import datetime

DB_PATH = "database/stocks.db"

# Sector Definitions
SECTOR_MAP = {
    "us_markets": ["SPY", "QQQ", "DIA", "IWM"],
    "australia": ["^AXJO", "EWA"],
    "japan": ["EWJ", "^N225"],
    "china": ["MCHI", "FXI", "KWEB", "^HSI"],
    "commodities": ["GLD", "SLV", "USO"],
    "bonds": ["TLT"]
}

def calculate_sector_sentiment():
    if not os.path.exists(DB_PATH): return
    conn = sqlite3.connect(DB_PATH)
    
    # 1. Fetch Ticker-Level Sentiment
    query = """
        SELECT date(na.published_date) as date, nt.ticker, 
               AVG(COALESCE(res.sentiment_score, res.weighted_sentiment)) as sentiment_score
        FROM news_articles na
        JOIN news_analysis res ON na.id = res.article_id
        JOIN news_to_tickers nt ON na.id = nt.news_id
        GROUP BY date, nt.ticker
    """
    df = pd.read_sql_query(query, conn)
    df['date'] = pd.to_datetime(df['date'])
    
    # 2. Map tickers to sectors
    reverse_map = {}
    for sector, tickers in SECTOR_MAP.items():
        for t in tickers:
            reverse_map[t] = sector
            
    df['sector'] = df['ticker'].map(reverse_map)
    df = df.dropna(subset=['sector'])

    # 3. Calculate Sector-Level Mean Sentiment
    sector_daily = df.groupby(['date', 'sector'])['sentiment_score'].mean().reset_index()
    sector_daily = sector_daily.rename(columns={'sentiment_score': 'sector_avg_sentiment'})

    # 4. Calculate Sentiment Momentum (7-day trend)
    sector_daily = sector_daily.sort_values(['sector', 'date'])
    sector_daily['sector_sentiment_momentum'] = sector_daily.groupby('sector')['sector_avg_sentiment'].transform(lambda x: x.rolling(7).mean())

    # 5. Merge back to get Divergence
    df = pd.merge(df, sector_daily, on=['date', 'sector'], how='left')
    df['sentiment_divergence'] = df['sentiment_score'] - df['sector_avg_sentiment']

    # 6. Pivot for a "wide" version (one row per date with all sector scores)
    sector_pivot = sector_daily.pivot(index='date', columns='sector', values='sector_avg_sentiment').fillna(0)
    sector_pivot.columns = [f"{col}_sector_sentiment" for col in sector_pivot.columns]
    
    # 7. Merge everything into a final table
    # We join with the original ticker-level data
    final_df = pd.merge(df, sector_pivot, on='date', how='left')

    print("Saving sector sentiment features to 'sector_sentiment_daily'...")
    final_df.to_sql("sector_sentiment_daily", conn, if_exists='replace', index=False)
    
    conn.close()
    print()
    print(f"Sector sentiment calculation complete. Processed {len(final_df)} records.")
    print(f"Sectors tracked: {list(SECTOR_MAP.keys())}")

if __name__ == "__main__":
    calculate_sector_sentiment()
