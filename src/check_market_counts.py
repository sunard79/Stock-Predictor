import sqlite3
import pandas as pd

DB_PATH = "database/stocks.db"

def check_counts():
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT market, COUNT(*) as count FROM features_data GROUP BY market"
    df = pd.read_sql(query, conn)
    print("Record counts per market:")
    print(df.to_string(index=False))
    
    # Also check date range for AUS
    query_aus = "SELECT MIN(date), MAX(date) FROM features_data WHERE market='AUS'"
    aus_range = pd.read_sql(query_aus, conn)
    print()
    print("AUS Date Range:")
    print(aus_range.to_string(index=False))
    conn.close()

if __name__ == "__main__":
    check_counts()
