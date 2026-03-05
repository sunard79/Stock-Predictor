import sqlite3
import pandas as pd
import numpy as np
import os

DB_PATH = "database/stocks.db"

def verify_previous_month():
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    
    # Target: Full Month of February 2026
    start_date = "2026-02-01"
    end_date = "2026-02-28"
    
    print()
    print(f"--- Full Month Validation: {start_date} to {end_date} ---")
    
    # 1. Price Data
    p_query = "SELECT ticker, date, close FROM stock_prices WHERE date BETWEEN ? AND ?"
    prices = pd.read_sql_query(p_query, conn, params=(start_date, end_date))
    
    # 2. News/Sentiment Data
    n_query = """
        SELECT na.published_date, res.sentiment_label, nt.ticker
        FROM news_articles na
        JOIN news_analysis res ON na.id = res.article_id
        JOIN news_to_tickers nt ON na.id = nt.news_id
        WHERE na.published_date BETWEEN ? AND ?
    """
    sentiment = pd.read_sql_query(n_query, conn, params=(start_date + " 00:00:00", end_date + " 23:59:59"))
    conn.close()

    if prices.empty or sentiment.empty:
        print("No overlapping data found for the cross-check period.")
        return

    # Process prices
    prices['date'] = pd.to_datetime(prices['date']).dt.date
    prices = prices.sort_values(['ticker', 'date'])
    prices['actual_return'] = prices.groupby('ticker')['close'].pct_change()
    
    # Process sentiment with robust mixed format parsing
    sentiment['published_date'] = pd.to_datetime(sentiment['published_date'], format='mixed', utc=True)
    sentiment['date'] = sentiment['published_date'].dt.date
    s_map = {'bullish': 1, 'neutral': 0, 'bearish': -1}
    sentiment['score'] = sentiment['sentiment_label'].map(s_map)
    daily_sent = sentiment.groupby(['date', 'ticker']).agg({'score': 'mean', 'sentiment_label': 'first'}).reset_index()

    # Merge
    val = pd.merge(daily_sent, prices, on=['date', 'ticker'], how='inner')
    val = val.dropna(subset=['actual_return']) # Remove first day of week (NaN return)

    def grade(row):
        if row['score'] > 0 and row['actual_return'] > 0: return "CORRECT (Bullish)"
        if row['score'] < 0 and row['actual_return'] < 0: return "CORRECT (Bearish)"
        if row['score'] == 0: return "NEUTRAL"
        return "WRONG"

    val['result'] = val.apply(grade, axis=1)
    
    print()
    print(val[['date', 'ticker', 'sentiment_label', 'actual_return', 'result']].to_string(index=False))
    
    # Stats
    correct = len(val[val['result'].str.startswith("CORRECT")])
    wrong = len(val[val['result'] == "WRONG"])
    total = correct + wrong
    
    print()
    print("--- Verification Summary ---")
    print(f"Decisive Signals: {total}")
    print(f"Correct Predictions: {correct}")
    print(f"Accuracy: {(correct/total if total > 0 else 0):.1%}")

if __name__ == "__main__":
    verify_previous_month()
