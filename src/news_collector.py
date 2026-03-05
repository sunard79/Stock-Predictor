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
    db_path = "database/stocks.db"
    create_news_table(db_path)

    feeds = {
        "CNBC": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "Wall Street Journal": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "Investing.com": "https://www.investing.com/rss/news_25.rss"
    }
    keywords = {'stock', 'market', 's&p', 'spy', 'trading', 'finance', 'economy', 'business', 'earnings', 'nvidia', 'iran', 'rba', 'fed', 'middle east'}
    # Extend to 30 days to fetch historical news for the full month
    search_limit_days = 30
    time_threshold = datetime.now() - timedelta(days=search_limit_days)
    articles_added = 0

    print()
    print(f"Fetching news from the last {search_limit_days} days containing keywords: {keywords}")

    for source, url in feeds.items():
        print(f"Fetching from {source}...")
        feed = feedparser.parse(url)
        
        for entry in feed.entries:
            title = entry.get("title", "")
            published_struct = entry.get("published_parsed", entry.get("updated_parsed"))
            if not published_struct: continue

            published_date = datetime.fromtimestamp(time.mktime(published_struct))
            if published_date < time_threshold: continue

            description = entry.get("summary", "")
            link = entry.get("link", "")
            content = (title + " " + description).lower()
            
            if not any(keyword in content for keyword in keywords): continue
            if not link or is_duplicate(link, db_path): continue
            
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute("""
                        INSERT INTO news_articles (title, description, url, published_date, source)
                        VALUES (?, ?, ?, ?, ?)
                    """, (title, description, link, published_date, source))
                    conn.commit()
                    articles_added += 1
                except sqlite3.IntegrityError:
                    continue
    
    print()
    print(f"Finished. Total articles added: {articles_added}")

if __name__ == "__main__":
    fetch_and_store_news()
