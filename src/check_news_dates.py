import sqlite3
import os

DB_PATH = os.path.join("database", "stocks.db")

if os.path.exists(DB_PATH):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT MIN(published_date), MAX(published_date), COUNT(*) FROM news_articles")
        result = cursor.fetchone()
        print(f"News Article Range: {result[0]} to {result[1]}")
        print(f"Total Articles: {result[2]}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()
else:
    print(f"Database file {DB_PATH} not found.")
