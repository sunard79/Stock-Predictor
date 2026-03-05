import sqlite3
import pandas as pd
import os

DB_PATH = "database/stocks.db"

def verify_china_data():
    conn = sqlite3.connect(DB_PATH)
    
    print("--- February News Volume ---")
    query = """
        SELECT date(published_date) as day, COUNT(*) as count 
        FROM news_articles 
        WHERE published_date BETWEEN '2026-02-01' AND '2026-02-28'
        GROUP BY day
    """
    df_vol = pd.read_sql(query, conn)
    print(df_vol.to_string(index=False))

    print()
    print("--- Search for 'China' in all historical news ---")
    query = "SELECT COUNT(*) FROM news_articles WHERE title LIKE '%China%' OR description LIKE '%China%'"
    cursor = conn.cursor()
    cursor.execute(query)
    print(f"Total articles mentioning 'China': {cursor.fetchone()[0]}")

    print()
    print("--- Top News Sources in DB ---")
    query = "SELECT source, COUNT(*) as count FROM news_articles GROUP BY source ORDER BY count DESC LIMIT 10"
    df_src = pd.read_sql(query, conn)
    print(df_src.to_string(index=False))
    
    conn.close()

if __name__ == "__main__":
    verify_china_data()
