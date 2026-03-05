import sqlite3
import os

DB_PATH = "database/stocks.db"

def check_schema():
    if not os.path.exists(DB_PATH):
        print("DB not found.")
        return
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(news_analysis)")
    cols = [info[1] for info in cursor.fetchall()]
    print("Columns in news_analysis:", cols)
    conn.close()

if __name__ == "__main__":
    check_schema()
