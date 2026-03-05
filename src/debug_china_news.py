import sqlite3
import os

DB_PATH = "database/stocks.db"

def debug_china_content():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("--- Articles mentioning 'China' but NOT matched to CHN tickers ---")
    query = """
        SELECT na.id, na.title, na.source 
        FROM news_articles na
        WHERE (na.title LIKE '%China%' OR na.description LIKE '%China%')
        AND na.id NOT IN (SELECT news_id FROM news_to_tickers WHERE ticker IN ('MCHI', 'FXI', 'KWEB'))
        LIMIT 10
    """
    cursor.execute(query)
    for row in cursor.fetchall():
        print(f"ID: {row[0]} | Source: {row[2]} | Title: {row[1]}")

    print()
    print("--- Sample of 'No Match' articles ---")
    query = """
        SELECT id, title, description, source FROM news_articles 
        WHERE id NOT IN (SELECT news_id FROM news_to_tickers)
        LIMIT 5
    """
    cursor.execute(query)
    for row in cursor.fetchall():
        print(f"ID: {row[0]} | Source: {row[3]} | Title: {row[1]}")
        desc_snippet = row[2][:100] if row[2] else 'NO DESCRIPTION'
        print(f"Snippet: {desc_snippet}")
        print()
        
    conn.close()

if __name__ == "__main__":
    debug_china_content()
