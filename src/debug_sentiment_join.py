import sqlite3
import pandas as pd
import os

DB_PATH = "database/stocks.db"

def debug_join():
    conn = sqlite3.connect(DB_PATH)
    
    print("--- Table Counts ---")
    counts = {
        "news_articles": conn.execute("SELECT COUNT(*) FROM news_articles").fetchone()[0],
        "news_analysis": conn.execute("SELECT COUNT(*) FROM news_analysis").fetchone()[0],
        "news_to_tickers": conn.execute("SELECT COUNT(*) FROM news_to_tickers").fetchone()[0],
    }
    for table, count in counts.items():
        print(f"{table}: {count}")

    print("\n--- Sample News (2023) ---")
    query_news = "SELECT id, published_date, title FROM news_articles WHERE published_date LIKE '2023%' LIMIT 3"
    print(pd.read_sql_query(query_news, conn))

    print("\n--- Sample Analysis for those IDs ---")
    query_analysis = """
        SELECT res.article_id, res.sentiment_score, res.macro_stress_level 
        FROM news_analysis res
        JOIN news_articles na ON res.article_id = na.id
        WHERE na.published_date LIKE '2023%' LIMIT 3
    """
    print(pd.read_sql_query(query_analysis, conn))

    print("\n--- Sample Mapping for those IDs ---")
    query_mapping = """
        SELECT nt.news_id, nt.ticker 
        FROM news_to_tickers nt
        JOIN news_articles na ON nt.news_id = na.id
        WHERE na.published_date LIKE '2023%' LIMIT 3
    """
    print(pd.read_sql_query(query_mapping, conn))

    print("\n--- Testing the Full Join used in Feature Engineering ---")
    sentiment_query = """
        SELECT date(na.published_date) as date, nt.ticker, 
               AVG(res.sentiment_score) as sentiment_score,
               COUNT(na.id) as news_count
        FROM news_articles na
        JOIN news_analysis res ON na.id = res.article_id
        JOIN news_to_tickers nt ON na.id = nt.news_id
        WHERE na.published_date LIKE '2023%'
        GROUP BY date, nt.ticker
        LIMIT 5
    """
    try:
        df = pd.read_sql_query(sentiment_query, conn)
        if df.empty:
            print("FULL JOIN RETURNED NO DATA FOR 2023.")
        else:
            print("Join Success! Sample results:")
            print(df)
    except Exception as e:
        print(f"Join Error: {e}")

    conn.close()

if __name__ == "__main__":
    debug_join()
