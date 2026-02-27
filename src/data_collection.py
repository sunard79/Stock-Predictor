import yfinance as yf
import sqlite3
import pandas as pd
from datetime import datetime

def collect_stock_data(ticker="SPY", start_date="2015-01-01", db_path="database/stocks.db"):
    """
    Downloads historical stock data using yfinance and saves it to an SQLite database.

    Args:
        ticker (str): The stock ticker symbol (default: "SPY").
        start_date (str): The start date for data collection in "YYYY-MM-DD" format.
        db_path (str): The path to the SQLite database file.
    """
    end_date = datetime.now().strftime("%Y-%m-%d")
    print(f"Collecting {ticker} data from {start_date} to {end_date}...")

    try:
        # Download data
        data = yf.download(ticker, start=start_date, end=end_date)

        if data.empty:
            print(f"No data downloaded for {ticker}.")
            return

        # Prepare data for database
        data.reset_index(inplace=True)
        data['Date'] = pd.to_datetime(data['Date'])
        data['Date'] = data['Date'].dt.date # Store date without time for simplicity

        # Connect to SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create table if it doesn't exist
        table_name = "stock_prices"
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                Date DATE PRIMARY KEY,
                Open REAL,
                High REAL,
                Low REAL,
                Close REAL,
                AdjClose REAL,
                Volume INTEGER
            );
        """)
        conn.commit()

        # Insert data into the table
        # Using executemany for efficiency and proper handling of dates
        data.to_sql(table_name, conn, if_exists='replace', index=False, dtype={'Date': 'DATE'})
        
        print(f"Successfully saved {len(data)} rows of {ticker} data to {db_path} in table {table_name}.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()

if __name__ == "__main__":
    # Ensure the database directory exists
    import os
    os.makedirs("database", exist_ok=True)
    collect_stock_data()
