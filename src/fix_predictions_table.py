import sqlite3
import os

DB_PATH = "database/stocks.db"

def fix_table():
    if not os.path.exists(DB_PATH):
        print("Database not found.")
        return

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        
        # Check existing columns
        cursor.execute("PRAGMA table_info(daily_predictions_history)")
        columns = [info[1] for info in cursor.fetchall()]
        
        print(f"Current columns: {columns}")
        
        # Columns to add if they are missing
        needed_columns = {
            'XGB_Z': 'TEXT',
            'LGB_Z': 'TEXT',
            'Price': 'TEXT',
            'Horizon': 'TEXT',
            'blend_mode': 'TEXT'
        }
        
        for col, col_type in needed_columns.items():
            if col not in columns:
                print(f"Adding missing column: {col}")
                try:
                    cursor.execute(f"ALTER TABLE daily_predictions_history ADD COLUMN {col} {col_type}")
                except Exception as e:
                    print(f"Error adding {col}: {e}")
        
        conn.commit()
        print("Table schema updated successfully.")

if __name__ == "__main__":
    fix_table()
