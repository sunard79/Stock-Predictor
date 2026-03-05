import sqlite3
import os

DB_PATH = "database/stocks.db"

def check_schema():
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("PRAGMA table_info(stock_prices)")
    columns = cursor.fetchall()
    print("Table: stock_prices")
    for col in columns:
        print(col)
    
    conn.close()

if __name__ == "__main__":
    check_schema()
