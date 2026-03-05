import sqlite3
import pandas as pd
import os

DB_PATH = "database/stocks.db"

def run_diagnostics():
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    
    print("--- 1. Table Row Counts ---")
    tables = ['stock_prices', 'features_data', 'features_data_robust', 'news_analysis', 'sector_sentiment_daily']
    counts = {}
    for table in tables:
        try:
            count = pd.read_sql_query(f"SELECT COUNT(*) as count FROM {table}", conn).iloc[0]['count']
            counts[table] = count
            print(f"{table: <25}: {count:,} rows")
        except Exception:
            counts[table] = 0
            print(f"{table: <25}: Table not found or empty")

    print("\n--- 2. Features Data Deep Dive ---")
    if counts.get('features_data', 0) > 0:
        df_features = pd.read_sql_query("SELECT ticker, date FROM features_data", conn)
        df_features['date'] = pd.to_datetime(df_features['date'])
        
        total_rows = len(df_features)
        ticker_counts = df_features['ticker'].value_counts()
        min_date = df_features['date'].min()
        max_date = df_features['date'].max()
        
        print(f"Total Feature Rows: {total_rows:,}")
        print(f"Date Range: {min_date.date()} to {max_date.date()}")
        print("\nRows per Ticker (Top 5):")
        print(ticker_counts.head(5))
        
        low_data_tickers = ticker_counts[ticker_counts < 100]
        if not low_data_tickers.empty:
            print(f"\nTickers with < 100 rows ({len(low_data_tickers)}):")
            print(low_data_tickers.index.tolist())
        else:
            print("\nAll tickers have at least 100 rows.")
    else:
        print("No features found to analyze.")

    print("\n--- 3. Missing Data & Joins ---")
    try:
        # Check stock_prices vs features
        sp_count = pd.read_sql_query("SELECT COUNT(*) as count FROM stock_prices", conn).iloc[0]['count']
        feat_count = pd.read_sql_query("SELECT COUNT(*) as count FROM features_data", conn).iloc[0]['count']
        
        # Check for NULLs in features
        df_sample = pd.read_sql_query("SELECT * FROM features_data LIMIT 1000", conn)
        null_counts = df_sample.isnull().sum().sort_values(ascending=False)
        
        print(f"Stock Prices total: {sp_count:,}")
        print(f"Features total    : {feat_count:,}")
        print(f"Gap (Missing)     : {sp_count - feat_count:,} rows")
        
        print("\nFeatures with most NULLs (top 5):")
        print(null_counts.head(5))
    except Exception as e:
        print(f"Could not complete join analysis: {e}")

    print("\n--- 4. Diagnostic Report ---")
    expected = 55000
    actual = counts.get('features_data', 0)
    gap = expected - actual
    
    print(f"Expected features: ~{expected:,} (2,500 days × 22 tickers)")
    print(f"Actual features:   {actual:,}")
    print(f"Gap:               {gap:,}")
    
    print("\nLikely cause:")
    # Heuristics for diagnosis
    cause_sentiment = "[X]" if counts.get('news_analysis', 0) < 100 else "[ ]"
    cause_indicators = "[X]" if gap > 0 and gap < 5000 else "[ ]"
    cause_joins = "[X]" if gap > 10000 else "[ ]"
    
    print(f"{cause_sentiment} Sentiment data missing for most dates")
    print(f"{cause_indicators} Feature calculation loss (lookback periods like RSI/MACD)")
    print(f"{cause_joins} Join conditions too strict (dropping rows in merge)")
    
    print("\n--- 5. Recommendations ---")
    if actual == 0:
        print("1. Run 'src/data_collection_multi_asset.py' to ensure base data exists.")
        print("2. Run 'src/feature_engineering.py' to generate features.")
    elif gap > 20000:
        print("CRITICAL: Massive data loss. Check 'src/feature_engineering.py' for INNER JOINs that should be LEFT JOINs.")
        print("Check if SPY or VIX data is missing, as these are often used as base join keys.")
    elif counts.get('news_analysis', 0) == 0:
        print("1. Run news collection and sentiment analysis to populate sentiment features.")
    else:
        print("Data looks relatively healthy. Ensure 'SPY' and '^VIX' have full history.")

    conn.close()

if __name__ == "__main__":
    run_diagnostics()
