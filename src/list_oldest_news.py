import sqlite3
import os

DB_PATH = os.path.join("database", "stocks.db")

def list_oldest_news():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("--- 5 Oldest News Articles in DB ---")
    query = "SELECT id, published_date, title FROM news_articles ORDER BY published_date ASC LIMIT 5"
    cursor.execute(query)
    for row in cursor.fetchall():
        print(f"ID: {row[0]} | Date: {row[1]} | {row[2][:60]}...")
        
    conn.close()

if __name__ == "__main__":
    list_oldest_news()
