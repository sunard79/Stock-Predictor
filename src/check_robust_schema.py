import sqlite3
import pandas as pd

DB_PATH = "database/stocks.db"

def check_robust_schema():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM features_data_robust LIMIT 1", conn)
        print("Columns in features_data_robust:")
        print(df.columns.tolist())
    except Exception as e:
        print(f"Error: {e}")
    conn.close()

if __name__ == "__main__":
    check_robust_schema()
