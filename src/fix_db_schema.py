import sqlite3
import os

DB_PATH = "database/stocks.db"

def fix_schema():
    if not os.path.exists(DB_PATH):
        print("Database not found.")
        return

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        
        # Check if 'provider' column exists
        cursor.execute("PRAGMA table_info(news_analysis)")
        columns = [info[1] for info in cursor.fetchall()]
        
        if 'provider' not in columns:
            print("Adding 'provider' column to news_analysis table...")
            try:
                cursor.execute("ALTER TABLE news_analysis ADD COLUMN provider TEXT DEFAULT 'gemini'")
                conn.commit()
                print("Column added successfully.")
            except Exception as e:
                print(f"Error adding column: {e}")
        else:
            print("'provider' column already exists.")

        # Also ensure macro_stress_level and sentiment_score exist
        if 'macro_stress_level' not in columns:
            print("Adding 'macro_stress_level' column...")
            cursor.execute("ALTER TABLE news_analysis ADD COLUMN macro_stress_level REAL DEFAULT 0.0")
        
        if 'sentiment_score' not in columns:
            print("Adding 'sentiment_score' column...")
            cursor.execute("ALTER TABLE news_analysis ADD COLUMN sentiment_score REAL DEFAULT 0.0")
            
        conn.commit()
        print("Schema update complete.")

if __name__ == "__main__":
    fix_schema()
