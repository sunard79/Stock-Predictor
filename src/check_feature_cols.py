import sqlite3
import pandas as pd

DB_PATH = "database/stocks.db"

def check_columns():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM features_data LIMIT 1", conn)
    print("Columns in features_data:")
    print(df.columns.tolist())
    conn.close()

if __name__ == "__main__":
    check_columns()
