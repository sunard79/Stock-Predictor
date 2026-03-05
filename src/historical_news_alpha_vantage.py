import requests
import sqlite3
import os
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

ENV_PATH = "config/.env"
DB_PATH = "database/stocks.db"

def fetch_alpha_vantage_history(api_key, tickers, start_date, end_date):
    """
    Fetches news sentiment from Alpha Vantage for a specific date range.
    Alpha Vantage format: YYYYMMDDTHHMM
    """
    print(f"--- Fetching Alpha Vantage: {start_date} to {end_date} ---")
    
    # Format dates for API
    time_from = start_date.strftime("%Y%m%dT%H%M")
    time_to = end_date.strftime("%Y%m%dT%H%M")
    
    # We query for broad market terms to get general sentiment if no specific ticker matches
    tickers_str = ",".join(tickers[:5]) # Alpha Vantage allows multiple tickers
    
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={tickers_str}&time_from={time_from}&time_to={time_to}&limit=200&apikey={api_key}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if "feed" not in data:
            print(f"Error or No Data: {data.get('Note', data.get('Information', 'Unknown error'))}")
            return 0

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        added = 0

        for item in data["feed"]:
            title = item.get("title")
            url_link = item.get("url")
            summary = item.get("summary")
            published_str = item.get("time_published") # Format: 20230310T143000
            
            # Convert AV time to DB format
            try:
                dt = datetime.strptime(published_str, "%Y%m%dT%H%M%S")
                published_date = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                published_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            source = item.get("source")
            
            # Sentiment from Alpha Vantage
            score = float(item.get("overall_sentiment_score", 0))
            label = item.get("overall_sentiment_label", "Neutral")

            try:
                # 1. Insert into news_articles
                cursor.execute("""
                    INSERT INTO news_articles (title, description, url, published_date, source)
                    VALUES (?, ?, ?, ?, ?)
                """, (title, summary, url_link, published_date, source))
                article_id = cursor.lastrowid
                
                # 2. Insert into news_analysis (so we don't have to re-run LLM)
                cursor.execute("""
                    INSERT INTO news_analysis (article_id, sentiment_label, sentiment_score, provider)
                    VALUES (?, ?, ?, ?)
                """, (article_id, label.lower(), score, 'alphavantage'))
                
                # 3. Map to tickers (optional but helpful)
                for t_sent in item.get("ticker_sentiment", []):
                    ticker = t_sent.get("ticker")
                    if ticker in tickers:
                        cursor.execute("INSERT OR IGNORE INTO news_to_tickers (news_id, ticker) VALUES (?, ?)", (article_id, ticker))
                
                added += 1
            except sqlite3.IntegrityError:
                continue # Duplicate URL

        conn.commit()
        conn.close()
        return added

    except Exception as e:
        print(f"Request failed: {e}")
        return 0

def main():
    load_dotenv(ENV_PATH)
    api_key = os.getenv("ALPHA_VANTAGE_KEY")
    if not api_key:
        print("Error: ALPHA_VANTAGE_KEY not found in config/.env")
        return

    # Key historical periods to backfill
    # 1. SVB Collapse (March 2023)
    # 2. Fed Hikes Start (May 2022)
    # 3. AI Boom (Feb 2024)
    
    periods = [
        (datetime(2023, 3, 8), datetime(2023, 3, 15), "SVB Crisis"),
        (datetime(2022, 5, 1), datetime(2022, 5, 10), "Fed Hike Acceleration"),
        (datetime(2024, 2, 20), datetime(2024, 2, 28), "Nvidia AI Breakout")
    ]
    
    tickers = ["SPY", "QQQ", "GLD", "^AXJO"]
    
    for start, end, name in periods:
        print(f"\nTargeting Period: {name}")
        count = fetch_alpha_vantage_history(api_key, tickers, start, end)
        print(f"Added {count} records for {name}.")
        
        # Respect API limits (Free tier: 5 calls/min or 25/day)
        if count > 0:
            print("Waiting 15s to respect API rate limits...")
            time.sleep(15)

if __name__ == "__main__":
    main()
