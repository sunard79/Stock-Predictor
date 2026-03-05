import sqlite3
import pandas as pd
import os

DB_PATH = "database/stocks.db"

def check_cols():
    if not os.path.exists(DB_PATH):
        print("DB not found.")
        return
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM features_data_robust LIMIT 1", conn)
    print("Columns in features_data_robust:")
    print(df.columns.tolist())
    conn.close()

if __name__ == "__main__":
    check_cols()
