import sqlite3
import pandas as pd
import os
import json

DB_PATH = "database/stocks.db"
OUTPUT_PATH = "data/processed/multi_asset_dashboard.csv"

def create_multi_asset_dashboard():
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at {DB_PATH}")
        return

    try:
        conn = sqlite3.connect(DB_PATH)
        prices_df = pd.read_sql_query("SELECT * FROM stock_prices", conn)
        prices_df['date'] = pd.to_datetime(prices_df['date'])

        # Group tickers into markets based on name and category
        def get_market(row):
            name = row['asset_name'].lower()
            ticker = row['ticker']
            if row['asset_type'] == "International":
                if "australia" in name or "asx 200" in name: return "AUS"
                if "japan" in name or ticker == "^N225": return "JPN"
                if "china" in name or ticker in ["FXI", "KWEB"]: return "CHN"
                if "hong kong" in name or ticker == "^HSI": return "HK"
                if "south korea" in name or ticker == "EWY": return "KOR"
                if "india" in name or ticker == "EPI": return "IND"
                if "germany" in name: return "GER"
                if "united kingdom" in name: return "UK"
                if "brazil" in name: return "BRA"
            if row['asset_type'] == "Risk":
                if ticker == "^VIX": return "Volatility"
                if ticker == "TLT": return "Bonds"
            return row['asset_type']

        prices_df['market'] = prices_df.apply(get_market, axis=1)
        prices_df.drop(columns=['asset_type'], inplace=True)

        # Sentiment query joining articles, analysis, and ticker mappings
        query = """
            SELECT na.published_date, res.weighted_sentiment as score, nt.ticker, res.relevance_score
            FROM news_articles na
            JOIN news_analysis res ON na.id = res.article_id
            JOIN news_to_tickers nt ON na.id = nt.news_id
        """
        mapped_news_df = pd.read_sql_query(query, conn)
        mapped_news_df['date'] = pd.to_datetime(mapped_news_df['published_date'], format='mixed', utc=True).dt.normalize()
        mapped_news_df['date'] = mapped_news_df['date'].dt.tz_localize(None)
        conn.close()

        # Aggregate daily sentiment per ticker using the weighted scores
        daily_ticker_sentiment = mapped_news_df.groupby(['date', 'ticker']).agg(
            sentiment_score=('score', 'mean'),
            relevance_avg=('relevance_score', 'mean'),
            sentiment_count=('ticker', 'count')
        ).reset_index()

        # Use SPY as market-wide sentiment fallback
        market_mood = daily_ticker_sentiment[daily_ticker_sentiment['ticker'] == 'SPY'].copy()
        market_mood = market_mood.rename(columns={
            'sentiment_score': 'market_score', 
            'sentiment_count': 'market_count'
        }).drop(columns=['ticker'])

        # Join and apply SPY sentiment where specific ticker news is missing
        df = pd.merge(prices_df, daily_ticker_sentiment, on=['date', 'ticker'], how='left')
        df = pd.merge(df, market_mood, on=['date'], how='left')
        
        df['sentiment_score'] = df['sentiment_score'].combine_first(df['market_score']).fillna(0)
        df['sentiment_count'] = df['sentiment_count'].combine_first(df['market_count']).fillna(0)
        df.drop(columns=['market_score', 'market_count'], inplace=True)

        # Technical indicators grouped by ticker
        df = df.sort_values(['ticker', 'date'])
        groups = df.groupby('ticker')
        df['daily_return'] = groups['close'].pct_change()
        df['ma_7'] = groups['close'].transform(lambda x: x.rolling(7).mean())
        df['ma_30'] = groups['close'].transform(lambda x: x.rolling(30).mean())
        df['volatility_7d'] = groups['daily_return'].transform(lambda x: x.rolling(7).std())

        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        df.to_csv(OUTPUT_PATH, index=False)
        
        print()
        print(f"Successfully created Multi-Asset Dashboard data at: {OUTPUT_PATH}")
        print(f"Total Rows: {len(df)}")
        print(f"Tickers Covered: {df['ticker'].nunique()}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    create_multi_asset_dashboard()
