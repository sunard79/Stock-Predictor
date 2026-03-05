import sqlite3
import pandas as pd
import json

DB_PATH = "database/stocks.db"

def check_analysis_coverage():
    conn = sqlite3.connect(DB_PATH)
    
    # Check total articles vs analyzed
    total_news = pd.read_sql("SELECT COUNT(*) FROM news_articles", conn).iloc[0,0]
    analyzed_news = pd.read_sql("SELECT COUNT(*) FROM news_analysis", conn).iloc[0,0]
    
    print(f"Total Articles: {total_news}")
    print(f"Analyzed Articles: {analyzed_news}")
    
    # Check ticker coverage in analysis
    df = pd.read_sql("SELECT affected_assets FROM news_analysis", conn)
    ticker_counts = {}
    
    for assets_json in df['affected_assets']:
        try:
            assets = json.loads(assets_json)
            for t in assets:
                ticker_counts[t] = ticker_counts.get(t, 0) + 1
        except:
            continue
            
    print()
    print("Sentiment Coverage by Ticker:")
    for ticker, count in sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True):
        print(f" - {ticker}: {count} articles")
        
    conn.close()

if __name__ == "__main__":
    check_analysis_coverage()
