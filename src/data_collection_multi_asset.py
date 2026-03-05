import yfinance as yf
import pandas as pd
import sqlite3
import datetime
import os

# Tickers and metadata configuration
ASSET_CONFIG = {
    "US": [
        {"ticker": "SPY", "name": "S&P 500 ETF"},
        {"ticker": "QQQ", "name": "Nasdaq 100 ETF"},
        {"ticker": "DIA", "name": "Dow Jones ETF"},
        {"ticker": "IWM", "name": "Russell 2000 ETF"}
    ],
    "International": [
        {"ticker": "^AXJO", "name": "S&P/ASX 200 Index"},
        {"ticker": "EWA", "name": "MSCI Australia ETF"},
        # {"ticker": "EWJ", "name": "MSCI Japan ETF"},
        # {"ticker": "^N225", "name": "Nikkei 225"},
        {"ticker": "EWU", "name": "MSCI United Kingdom ETF"},
        {"ticker": "EWG", "name": "MSCI Germany ETF"},
        # {"ticker": "MCHI", "name": "MSCI China ETF"},
        # {"ticker": "FXI", "name": "iShares China Large-Cap ETF"},
        # {"ticker": "KWEB", "name": "KraneShares China Internet ETF"},
        # {"ticker": "^HSI", "name": "Hang Seng Index"},
        # {"ticker": "EWY", "name": "MSCI South Korea ETF"},
        # {"ticker": "EPI", "name": "WisdomTree India Earnings ETF"},
        {"ticker": "EWZ", "name": "MSCI Brazil ETF"}
    ],
    "Commodities": [
        {"ticker": "GLD", "name": "Gold Trust"},
        {"ticker": "SLV", "name": "Silver Trust"},
        {"ticker": "USO", "name": "United States Oil Fund"}
    ],
    "Risk": [
        {"ticker": "^VIX", "name": "Volatility Index"},
        {"ticker": "TLT", "name": "20+ Yr Treasury Bond ETF"}
    ],
    "Macro": [
        {"ticker": "^VIX3M", "name": "VIX 3-Month Index"},
        {"ticker": "HYG", "name": "High Yield Corporate Bond ETF"},
        {"ticker": "IEF", "name": "7-10 Year Treasury Bond ETF"},
        {"ticker": "^TNX", "name": "10-Year Treasury Yield"},
        {"ticker": "^IRX", "name": "13-Week Treasury Bill Rate"},
        {"ticker": "EEM", "name": "Emerging Markets ETF"},
        {"ticker": "TIP", "name": "TIPS Bond ETF"},
        {"ticker": "UUP", "name": "US Dollar Index Bullish Fund"},
        {"ticker": "GDX", "name": "Gold Miners ETF"},
    ]
}

DB_PATH = os.path.join("database", "stocks.db")
TABLE_NAME = "stock_prices"
START_DATE = "2015-01-01"

def setup_database():
    """Initializes the database and table with the correct schema."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if table exists and has the correct schema
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (TABLE_NAME,))
    table_exists = cursor.fetchone()
    
    if table_exists:
        # Check for the 'ticker' column specifically
        cursor.execute(f"PRAGMA table_info({TABLE_NAME})")
        columns = [col[1] for col in cursor.fetchall()]
        if "ticker" not in columns:
            print(f"Old schema detected in {TABLE_NAME}. Recreating table with new schema...")
            cursor.execute(f"DROP TABLE {TABLE_NAME}")
            conn.commit()

    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            ticker TEXT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            asset_type TEXT,
            asset_name TEXT,
            PRIMARY KEY (ticker, date)
        )
    """)
    conn.commit()
    return conn

def download_ticker_data(ticker, asset_type, asset_name):
    """Downloads historical data for a given ticker."""
    try:
        data = yf.download(ticker, start=START_DATE, progress=False)
        if data.empty:
            print(f"Warning: No data found for {ticker}")
            return None
        
        # Reset index to get the Date column
        data = data.reset_index()
        
        # Rename columns to match requirements and handle yfinance MultiIndex or single index
        data.columns = [col.lower() if isinstance(col, str) else col[0].lower() for col in data.columns]
        
        # Add metadata columns
        data['ticker'] = ticker
        data['asset_type'] = asset_type
        data['asset_name'] = asset_name
        
        # Select and order columns
        cols = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'asset_type', 'asset_name']
        data = data[cols]
        
        # Format date as string (YYYY-MM-DD)
        data['date'] = data['date'].dt.strftime('%Y-%m-%d')
        
        return data
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        return None

def save_to_db(conn, data, ticker):
    """Deletes existing data for a ticker and inserts new data."""
    try:
        cursor = conn.cursor()
        cursor.execute(f"DELETE FROM {TABLE_NAME} WHERE ticker = ?", (ticker,))
        data.to_sql(TABLE_NAME, conn, if_exists='append', index=False)
        conn.commit()
        return True
    except Exception as e:
        print(f"Error saving {ticker} to database: {e}")
        return False

def validate_data(conn):
    """Validates the database for missing values and potential data gaps."""
    print()
    print("--- Running Database Validation ---")
    df = pd.read_sql(f"SELECT ticker, date FROM {TABLE_NAME} ORDER BY ticker, date", conn)
    df['date'] = pd.to_datetime(df['date'])
    
    # Check for null values
    null_counts = pd.read_sql(f"SELECT * FROM {TABLE_NAME} WHERE ticker IS NULL OR date IS NULL OR close IS NULL", conn)
    if not null_counts.empty:
        print(f"FAILED: Found {len(null_counts)} records with NULL values.")
    else:
        print("PASSED: No NULL values in critical columns.")
    
    # Check for data gaps (more than 5 days between records)
    gaps_found = False
    for ticker in df['ticker'].unique():
        ticker_dates = df[df['ticker'] == ticker]['date'].sort_values()
        diffs = ticker_dates.diff().dt.days
        gaps = diffs[diffs > 5]  # 5 days threshold to account for holidays/long weekends
        if not gaps.empty:
            gaps_found = True
            print(f"Warning: {ticker} has {len(gaps)} gaps larger than 5 days.")
    
    if not gaps_found:
        print("PASSED: No significant date gaps found.")

def main():
    conn = setup_database()
    summary_list = []
    
    print(f"--- Starting Multi-Asset Data Collection (from {START_DATE}) ---")
    
    # Flatten the config for processing
    all_tasks = []
    for asset_type, tickers in ASSET_CONFIG.items():
        for t in tickers:
            all_tasks.append((t['ticker'], asset_type, t['name']))
            
    for ticker, asset_type, asset_name in all_tasks:
        print(f"Processing {ticker} ({asset_name})...")
        df = download_ticker_data(ticker, asset_type, asset_name)
        
        if df is not None:
            if save_to_db(conn, df, ticker):
                n_rows = len(df)
                min_date = df['date'].min()
                max_date = df['date'].max()
                latest_close = df.iloc[-1]['close']
                
                print(f"Success: {ticker: <6} | {n_rows: >4} rows | {min_date} to {max_date} | Latest: {latest_close:.2f}")
                
                summary_list.append({
                    "ticker": ticker,
                    "records": n_rows,
                    "date_range": f"{min_date} to {max_date}",
                    "avg_price": df['close'].mean()
                })
            else:
                print(f"Failed:  {ticker: <6} | Database save failed.")
        else:
            print(f"Failed:  {ticker: <6} | Data download failed.")
            
    # Final Summary Table
    if summary_list:
        summary_df = pd.DataFrame(summary_list)
        print()
        print("--- Collection Summary ---")
        print(summary_df.to_string(index=False, formatters={'avg_price': '{:,.2f}'.format}))
    
    # Run Validation
    validate_data(conn)
    
    conn.close()
    print()
    print("Process completed successfully.")

if __name__ == "__main__":
    main()
