import sqlite3

DB_PATH = "database/stocks.db"

def list_tables_and_dates():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # List all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Tables in database:", [t[0] for t in tables])
    
    # Check for specific prediction tables
    prediction_tables = ['predictions', 'daily_report', 'daily_prediction_validation', 'features_data_robust']
    for table in prediction_tables:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"\nTable '{table}' has {count} rows.")
            
            if table == 'features_data_robust':
                cursor.execute(f"SELECT MAX(date) FROM {table}")
                max_date = cursor.fetchone()[0]
                print(f"  Latest date in {table}: {max_date}")
            elif table == 'daily_prediction_validation':
                cursor.execute(f"SELECT DISTINCT target_date FROM {table}")
                dates = cursor.fetchall()
                print(f"  Validated dates: {[d[0] for d in dates]}")
        except:
            pass
            
    conn.close()

if __name__ == "__main__":
    list_tables_and_dates()
