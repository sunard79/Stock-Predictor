import feedparser
import sqlite3
from datetime import datetime, timedelta
import time

def create_news_table(db_path="database/stocks.db"):
    """Creates the news_articles table if it doesn't exist."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS news_articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                url TEXT UNIQUE NOT NULL,
                published_date DATETIME NOT NULL,
                source TEXT NOT NULL,
                sentiment_label TEXT,
                sentiment_confidence REAL,
                sentiment_reasoning TEXT
            );
        """)
        conn.commit()
    print("Table 'news_articles' is ready.")

def is_duplicate(url, db_path="database/stocks.db"):
    """Checks if a URL already exists in the database."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM news_articles WHERE url = ?", (url,))
        return cursor.fetchone() is not None

def fetch_and_store_news():
    """Fetches news from RSS feeds, filters, and stores in the database."""
    db_path = "database/stocks.db"
    create_news_table(db_path)

    feeds = {
        "CNBC": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "Wall Street Journal": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "Investing.com": "https://www.investing.com/rss/news_25.rss"
    }
    keywords = {'stock', 'market', 's&p', 'spy', 'trading', 'finance', 'economy', 'business', 'earnings'}
    seven_days_ago = datetime.now() - timedelta(days=7)
    articles_added = 0

    print(f"Looking for articles from the last 7 days containing the keywords: {keywords}")

    for source, url in feeds.items():
        print(f"\\nFetching news from {source}...")
        feed = feedparser.parse(url)
        
        if not feed.entries:
            print(f"  - No articles found in feed.")
            continue

        for entry in feed.entries:
            title = entry.get("title", "")
            
            published_struct = entry.get("published_parsed", entry.get("updated_parsed"))
            if not published_struct:
                print(f"  - Skipped (no date): {title[:60]}...")
                continue

            published_date = datetime.fromtimestamp(time.mktime(published_struct))
            if published_date < seven_days_ago:
                # This can be noisy, so we'll comment it out unless debugging is needed.
                # print(f"  - Skipped (old date): {title[:60]}...")
                continue

            description = entry.get("summary", "")
            link = entry.get("link", "")
            content = (title + " " + description).lower()
            
            if not any(keyword in content for keyword in keywords):
                # This can also be very noisy.
                # print(f"  - Skipped (no keywords): {title[:60]}...")
                continue

            if not link or is_duplicate(link, db_path):
                # print(f"  - Skipped (duplicate): {title[:60]}...")
                continue
            
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute("""
                        INSERT INTO news_articles (title, description, url, published_date, source)
                        VALUES (?, ?, ?, ?, ?)
                    """, (title, description, link, published_date, source))
                    conn.commit()
                    articles_added += 1
                    print(f"  + Added: {title[:60]}...")
                except sqlite3.IntegrityError:
                    print(f"  - Skipped (duplicate on insert): {title[:60]}...")
                except Exception as e:
                    print(f"An error occurred while inserting article: {e}")
    
    print(f"\\nFinished. Total articles added: {articles_added}")

if __name__ == "__main__":
    fetch_and_store_news()
