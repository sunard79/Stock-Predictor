import pandas as pd
import sqlite3
import os
from tqdm import tqdm

CSV_PATH = "data/archive/sp500_headlines_2008_2024.csv"
DB_PATH = "database/stocks.db"

def import_csv():
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
        return

    print("Loading CSV data...")
    df = pd.read_csv(CSV_PATH)
    
    # 1. Prepare and filter dates
    df['Date'] = pd.to_datetime(df['Date'])
    mask = (df['Date'] >= '2022-01-01') & (df['Date'] <= '2024-12-31')
    historical_df = df.loc[mask].copy()
    
    if historical_df.empty:
        print("No records found for 2022-2024 in CSV.")
        return

    print(f"Found {len(historical_df)} headlines from 2022-2024. Importing...")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    added_count = 0
    mapping_count = 0

    for _, row in tqdm(historical_df.iterrows(), total=len(historical_df)):
        title = str(row['Title'])
        # Standardizing date format for DB
        pub_date = row['Date'].strftime('%Y-%m-%d 00:00:00')
        source = "Kaggle Archive"
        url = f"archive_{hash(title)}_{pub_date[:10]}" # Generate unique fake URL for mapping

        try:
            # 1. Insert into news_articles
            cursor.execute("""
                INSERT INTO news_articles (title, description, url, published_date, source)
                VALUES (?, ?, ?, ?, ?)
            """, (title, "", url, pub_date, source))
            
            news_id = cursor.lastrowid
            
            # 2. Map to SPY so feature engineering can find it
            cursor.execute("""
                INSERT OR IGNORE INTO news_to_tickers (news_id, ticker)
                VALUES (?, ?)
            """, (news_id, "SPY"))
            
            added_count += 1
            mapping_count += 1
            
        except sqlite3.IntegrityError:
            # Duplicate URL/Title
            continue

    conn.commit()
    conn.close()
    print(f"\nImport Complete: Added {added_count} articles and mapped them to SPY.")

if __name__ == "__main__":
    import_csv()
