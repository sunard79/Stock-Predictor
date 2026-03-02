import sqlite3
import pandas as pd
import os

DB_PATH = "database/stocks.db"
OUTPUT_PATH = "data/processed/dashboard_data.csv"

def create_dashboard_data():
    """
    Processes stock prices and news sentiments to create a unified dataset for the dashboard.
    """
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at {DB_PATH}")
        return

    try:
        # 1. Connect and Load Data
        conn = sqlite3.connect(DB_PATH)
        
        print("Loading stock prices...")
        stocks_df = pd.read_sql_query("SELECT * FROM stock_prices", conn)
        
        print("Loading news articles...")
        news_df = pd.read_sql_query("SELECT * FROM news_articles", conn)
        
        conn.close()

        if stocks_df.empty:
            print("Error: stock_prices table is empty.")
            return

        # 2. Preprocess Dates
        # Based on previous db_explorer output, stock_prices columns might be multi-index like
        # Handling the specific format seen in db_explorer: ('Date', ''), ('Close', 'SPY'), etc.
        # Actually, if read directly from SQLite without special handling, they might be strings.
        
        # Mapping for the column names based on what we saw in db_explorer
        # It showed columns like "('Date', '')", "('Close', 'SPY')"
        # Let's clean up column names if they are in that tuple string format
        stocks_df.columns = [c.replace("('", "").replace("', '')", "").replace("', 'SPY')", "") for c in stocks_df.columns]
        
        stocks_df['Date'] = pd.to_datetime(stocks_df['Date'])
        news_df['published_date'] = pd.to_datetime(news_df['published_date']).dt.date
        news_df['published_date'] = pd.to_datetime(news_df['published_date']) # Standardize back to datetime for merge

        # 3. Calculate Daily Sentiment Score
        # Map labels to numeric values
        sentiment_map = {'bullish': 1, 'neutral': 0, 'bearish': -1}
        news_df['score'] = news_df['sentiment_label'].map(sentiment_map)
        
        # Filter for analyzed articles
        analyzed_news = news_df[news_df['sentiment_label'].notnull()].copy()
        
        if analyzed_news.empty:
            print("Warning: No sentiment data found. sentiment_score will be 0.")
            daily_sentiment = pd.DataFrame(columns=['Date', 'sentiment_score', 'sentiment_count'])
        else:
            daily_sentiment = analyzed_news.groupby('published_date').agg(
                sentiment_score=('score', 'mean'),
                sentiment_count=('score', 'count')
            ).reset_index()
            daily_sentiment.rename(columns={'published_date': 'Date'}, inplace=True)

        # 4. Join Data
        print("Joining stock and sentiment data...")
        df = pd.merge(stocks_df, daily_sentiment, on='Date', how='left')
        df['sentiment_score'] = df['sentiment_score'].fillna(0)
        df['sentiment_count'] = df['sentiment_count'].fillna(0)

        # 5. Add Technical Indicators
        print("Calculating technical indicators...")
        df = df.sort_values('Date')
        
        # Daily Return
        df['daily_return'] = df['Close'].pct_change()
        
        # Moving Averages
        df['ma_7'] = df['Close'].rolling(window=7).mean()
        df['ma_30'] = df['Close'].rolling(window=30).mean()

        # 6. Final Selection and Export
        # Expected columns: date, close_price, volume, daily_return, sentiment_score, sentiment_count, ma_7, ma_30
        final_df = df[[
            'Date', 'Close', 'Volume', 'daily_return', 
            'sentiment_score', 'sentiment_count', 'ma_7', 'ma_30'
        ]].copy()
        
        final_df.rename(columns={'Date': 'date', 'Close': 'close_price', 'Volume': 'volume'}, inplace=True)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        
        final_df.to_csv(OUTPUT_PATH, index=False)
        print(f"Successfully created dashboard data at {OUTPUT_PATH}")
        print(f"Total rows: {len(final_df)}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    create_dashboard_data()
