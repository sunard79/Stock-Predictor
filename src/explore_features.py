import sqlite3
import pandas as pd
import os

DB_PATH = "database/stocks.db"

def explore_engineered_features():
    if not os.path.exists(DB_PATH):
        print("Database not found.")
        return

    conn = sqlite3.connect(DB_PATH)
    
    print("Loading data from 'features_data' table...")
    df = pd.read_sql_query("SELECT * FROM features_data", conn)
    conn.close()

    if df.empty:
        print("Table is empty.")
        return

    df['date'] = pd.to_datetime(df['date'])
    
    print()
    print("--- Available Features in Database ---")
    cols = df.columns.tolist()
    print(f"Total Columns: {len(cols)}")
    print(f"Sample Features: {cols[-15:]}")

    print()
    print("--- Latest Signal Snapshots (by Market) ---")
    latest_indices = df.groupby('market')['date'].idxmax()
    latest_df = df.loc[latest_indices]
    
    display_cols = [
        'date', 'ticker', 'market', 'rsi', 'bb_high_dist', 'bb_low_dist', 
        'atr_14', 'corr_30d', 'vix_rank', 'rel_strength_spy'
    ]
    
    subset = latest_df[display_cols].copy()
    for col in subset.columns:
        if subset[col].dtype == 'float64':
            subset[col] = subset[col].round(4)
            
    print(subset.to_string(index=False))

    print()
    print("--- Feature Correlation with Next-Day Return (Sample) ---")
    df['target'] = df.groupby('ticker')['return_1d'].shift(-1)
    # Filter for numeric columns for correlation
    corrs = df[['target', 'rsi', 'vix_rank', 'corr_30d', 'sentiment_score', 'rel_strength_spy']].corr()['target']
    print(corrs.to_string())

if __name__ == "__main__":
    explore_engineered_features()
