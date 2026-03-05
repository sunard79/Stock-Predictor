import sqlite3
import os

DB_PATH = "database/stocks.db"

def inspect_unmatched_articles():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Find articles that ARE NOT in news_to_tickers
    query = """
        SELECT id, title FROM news_articles 
        WHERE id NOT IN (SELECT DISTINCT news_id FROM news_to_tickers)
    """
    cursor.execute(query)
    articles = cursor.fetchall()
    
    print(f"--- Inspecting {len(articles)} Unmatched Articles ---")
    for article_id, title in articles:
        print(f"ID: {article_id} | Title: {title}")
        
    conn.close()

if __name__ == "__main__":
    inspect_unmatched_articles()
