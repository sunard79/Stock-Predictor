import sqlite3
import os

DB_PATH = os.path.join("database", "stocks.db")

def check_data_availability():
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("--- Data Availability Check ---")
    
    # Check Price Range
    cursor.execute("SELECT MIN(date), MAX(date) FROM stock_prices")
    price_range = cursor.fetchone()
    print(f"Stock Prices Range: {price_range[0]} to {price_range[1]}")
    
    # Check News Range
    cursor.execute("SELECT MIN(published_date), MAX(published_date), COUNT(*) FROM news_articles")
    news_range = cursor.fetchone()
    print(f"News Articles Range: {news_range[0]} to {news_range[1]} (Total: {news_range[2]})")
    
    conn.close()

if __name__ == "__main__":
    check_data_availability()
