import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join("database", "stocks.db")

def analyze_2023_impact():
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    # Tickers to watch: SPY (US), ^AXJO (AUS), EWJ (JPN), GLD (Gold), USO (Oil), ^VIX (Fear)
    tickers = ["SPY", "^AXJO", "EWJ", "GLD", "USO", "^VIX"]
    
    # Oct 7, 2023 was a Saturday. Let's look at Oct 2 to Oct 20.
    query = """
        SELECT ticker, date, close 
        FROM stock_prices 
        WHERE date BETWEEN '2023-10-02' AND '2023-10-20'
        AND ticker IN ({})
    """.format(','.join(['?']*len(tickers)))
    
    df = pd.read_sql_query(query, conn, params=tickers)
    conn.close()

    if df.empty:
        print("No price data found for October 2023.")
        return

    df['date'] = pd.to_datetime(df['date'])
    pivot_df = df.pivot(index='date', columns='ticker', values='close')
    
    # Calculate % change from Oct 6 (Friday before the attack)
    oct_6_prices = pivot_df.loc['2023-10-06']
    pct_change = (pivot_df / oct_6_prices - 1) * 100
    
    print("--- Market Response Post-Oct 7, 2023 (% Change from Oct 6) ---")
    print(pct_change.to_string())

if __name__ == "__main__":
    analyze_2023_impact()
