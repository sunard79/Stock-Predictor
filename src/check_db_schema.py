import sqlite3
import os

DB_PATH = os.path.join("database", "stocks.db")

if os.path.exists(DB_PATH):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("PRAGMA table_info(stock_prices)")
        columns = cursor.fetchall()
        if columns:
            print("Current columns in 'stock_prices':")
            for col in columns:
                print(f" - {col[1]} ({col[2]})")
        else:
            print("Table 'stock_prices' does not exist.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()
else:
    print(f"Database file {DB_PATH} not found.")
