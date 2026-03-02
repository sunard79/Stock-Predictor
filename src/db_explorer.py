import sqlite3
import pandas as pd

DB_PATH = "database/stocks.db"

def explore_database():
    """
    Connects to the database, lists all tables, and prints the first 5 rows of each table.
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            print("--- Exploring Database: database/stocks.db ---")

            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()

            if not tables:
                print()
                print("Database is empty or contains no tables.")
                return

            print()
            print(f"Found tables: {[table[0] for table in tables]}")

            for table_name in tables:
                table_name = table_name[0]
                print()
                print(f"--- Table: {table_name} ---")
                try:
                    # Corrected the function name from read_sql_ to read_sql_query
                    df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 5", conn)
                    if df.empty:
                        print("Table is empty.")
                    else:
                        print(df.to_string())
                except pd.io.sql.DatabaseError as e:
                    print(f"Could not read table '{table_name}': {e}")
        
        print()
        print("--- End of Exploration ---")

    except sqlite3.OperationalError as e:
        print(f"Database error: {e}. Ensure '{DB_PATH}' exists.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    explore_database()
