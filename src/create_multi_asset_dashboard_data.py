import sqlite3
import pandas as pd
import os
import json

DB_PATH = "database/stocks.db"
OUTPUT_PATH = "data/processed/multi_asset_dashboard.csv"

def create_multi_asset_dashboard():
    """
    Processes multi-asset prices and enhanced news analysis into a unified dashboard dataset.
    """
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at {DB_PATH}")
        return

    try:
        conn = sqlite3.connect(DB_PATH)
        
        # 1. Load Prices
        print("Loading multi-asset prices...")
        prices_df = pd.read_sql_query("SELECT * FROM stock_prices", conn)
        prices_df['date'] = pd.to_datetime(prices_df['date'])
        
        # 2. Load Enhanced News Analysis
        print("Loading enhanced news analysis...")
        query = """
            SELECT na.published_date, res.sentiment_label, res.affected_assets, res.geographic_focus
            FROM news_articles na
            JOIN news_analysis res ON na.id = res.article_id
        """
        news_df = pd.read_sql_query(query, conn)
        news_df['date'] = pd.to_datetime(news_df['published_date']).dt.normalize()
        
        conn.close()

        if prices_df.empty:
            print("Error: stock_prices table is empty.")
            return

        # 3. Process Ticker-Specific Sentiment
        # Map labels to scores
        sentiment_map = {'bullish': 1, 'neutral': 0, 'bearish': -1}
        news_df['score'] = news_df['sentiment_label'].map(sentiment_map)
        
        # Expand 'affected_assets' from JSON string to individual rows
        print("Calculating asset-specific sentiment scores...")
        news_expanded = []
        for _, row in news_df.iterrows():
            try:
                affected = json.loads(row['affected_assets'])
                for ticker in affected:
                    news_expanded.append({
                        'date': row['date'],
                        'ticker': ticker,
                        'score': row['score'],
                        'geo': row['geographic_focus']
                    })
            except:
                continue
        
        sentiment_df = pd.DataFrame(news_expanded)
        
        # Aggregate sentiment by date and ticker
        daily_sentiment = sentiment_df.groupby(['date', 'ticker']).agg(
            sentiment_score=('score', 'mean'),
            sentiment_count=('score', 'count')
        ).reset_index()

        # 4. Merge Prices and Sentiment
        print("Merging prices with sentiment...")
        final_df = pd.merge(prices_df, daily_sentiment, on=['date', 'ticker'], how='left')
        
        # Fill missing values
        final_df['sentiment_score'] = final_df['sentiment_score'].fillna(0)
        final_df['sentiment_count'] = final_df['sentiment_count'].fillna(0)

        # 5. Add Technical Indicators per Ticker
        print("Calculating technical indicators...")
        final_df = final_df.sort_values(['ticker', 'date'])
        
        # Group by ticker for indicators
        groups = final_df.groupby('ticker')
        final_df['daily_return'] = groups['close'].pct_change()
        final_df['ma_7'] = groups['close'].transform(lambda x: x.rolling(7).mean())
        final_df['ma_30'] = groups['close'].transform(lambda x: x.rolling(30).mean())
        
        # 6. Add Volatility (Standard Deviation of returns)
        final_df['volatility_7d'] = groups['daily_return'].transform(lambda x: x.rolling(7).std())

        # 7. Export
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        final_df.to_csv(OUTPUT_PATH, index=False)
        
        print()
        print(f"Successfully created Multi-Asset Dashboard data at: {OUTPUT_PATH}")
        print(f"Total Rows: {len(final_df)}")
        print(f"Tickers Covered: {final_df['ticker'].nunique()}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    create_multi_asset_dashboard()
