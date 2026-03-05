import sqlite3
import os

DB_PATH = "database/stocks.db"

def check_data_availability():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("--- Data Availability Check ---")
    
    # Check Price Range
    cursor.execute("SELECT MIN(date), MAX(date) FROM stock_prices")
    price_range = cursor.fetchone()
    print(f"Stock Prices: {price_range[0]} to {price_range[1]}")
    
    # Check News Range
    cursor.execute("SELECT MIN(published_date), MAX(published_date), COUNT(*) FROM news_articles")
    news_range = cursor.fetchone()
    print(f"News Articles: {news_range[0]} to {news_range[1]} (Total: {news_range[2]})")
    
    # Check Analysis Range
    cursor.execute("SELECT COUNT(*) FROM news_analysis")
    analysis_count = cursor.fetchone()
    print(f"Analyzed Articles: {analysis_count[0]}")
    
    conn.close()

if __name__ == "__main__":
    check_data_availability()
