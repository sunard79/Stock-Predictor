import sqlite3
import pandas as pd
import os

DB_PATH = "database/stocks.db"

def check_history():
    if not os.path.exists(DB_PATH):
        print("Database not found.")
        return

    conn = sqlite3.connect(DB_PATH)
    
    print("--- News History ---")
    news_df = pd.read_sql_query("""
        SELECT MIN(date(published_date)) as earliest, 
               MAX(date(published_date)) as latest, 
               COUNT(*) as total_articles 
        FROM news_articles
    """, conn)
    print(news_df.to_string(index=False))
    
    print("\n--- Feature Table Coverage (News vs Price) ---")
    coverage_df = pd.read_sql_query("""
        SELECT COUNT(*) as total_rows,
               SUM(CASE WHEN sentiment_score != 0 AND sentiment_score IS NOT NULL THEN 1 ELSE 0 END) as rows_with_sentiment
        FROM features_data_robust
    """, conn)
    coverage_df['percentage'] = (coverage_df['rows_with_sentiment'] / coverage_df['total_rows']) * 100
    print(coverage_df.to_string(index=False))
    
    # Check for specific "Shock" events
    print("\n--- Events with High News Volume ---")
    shocks_df = pd.read_sql_query("""
        SELECT date(published_date) as d, COUNT(*) as count 
        FROM news_articles 
        GROUP BY d 
        ORDER BY count DESC 
        LIMIT 5
    """, conn)
    print(shocks_df.to_string(index=False))
    
    conn.close()

if __name__ == "__main__":
    check_history()
