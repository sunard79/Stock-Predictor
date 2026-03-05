import sqlite3
import os

DB_PATH = "database/stocks.db"

def check_chinese_content():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check for Chinese characters in title or description
    cursor.execute("SELECT id, title FROM news_articles WHERE title GLOB '*[一-龥]*' LIMIT 5")
    rows = cursor.fetchall()
    
    if rows:
        print(f"Found {len(rows)} untranslated Chinese articles. (Sample IDs: {[r[0] for r in rows]})")
        print("You must run 'python src/translate_news.py' before analysis.")
    else:
        # Check if they WERE translated (look for specific sources)
        cursor.execute("SELECT COUNT(*) FROM news_articles WHERE source IN ('Caixin', 'Wall Street CN', 'Sina Finance', 'East Money')")
        count = cursor.fetchone()[0]
        print(f"Total articles from Chinese sources: {count}")
        
    conn.close()

if __name__ == "__main__":
    check_chinese_content()
