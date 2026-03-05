import sqlite3
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

DB_PATH = "database/stocks.db"
OUTPUT_DIR = "docs"

def calculate_market_correlations():
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    
    # Fetch all 2023 data
    query = "SELECT ticker, date, close FROM stock_prices WHERE date BETWEEN '2023-01-01' AND '2023-12-31'"
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print("No data found for 2023.")
        return

    # Pivot and calculate daily returns
    df['date'] = pd.to_datetime(df['date'])
    pivot_df = df.pivot(index='date', columns='ticker', values='close')
    returns_df = pivot_df.pct_change().dropna()

    # Calculate Correlation Matrix
    corr_matrix = returns_df.corr()

    # Identify Key Relationships
    print("--- 2023 Correlation Analysis ---")
    
    # 1. Top 5 correlations with SPY (US Market)
    spy_corr = corr_matrix['SPY'].sort_values(ascending=False)
    print()
    print("Top Assets Moving WITH SPY (US Market):")
    print(spy_corr.iloc[1:6].to_string())
    
    # 2. Top inverse correlations with SPY (Hedges)
    print()
    print("Top Assets Moving AGAINST SPY (Hedges):")
    print(spy_corr.iloc[-3:].to_string())

    # 3. Lead/Lag: Does VIX today affect SPY tomorrow?
    vix_lead = returns_df['^VIX'].corr(returns_df['SPY'].shift(-1))
    print()
    print(f"Lead Indicator Check: VIX today vs SPY tomorrow Correlation: {vix_lead:.4f}")
    
    # 4. Global Sensitivity: Which international market followed SPY most closely?
    intl_tickers = ["^AXJO", "EWJ", "EWU", "EWG", "MCHI", "EWZ"]
    intl_corr = spy_corr[spy_corr.index.isin(intl_tickers)]
    print()
    print("International Sensitivity to US Market (Beta):")
    print(intl_corr.to_string())

    # Generate Heatmap
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Multi-Asset Correlation Matrix (2023)")
    plt.savefig(os.path.join(OUTPUT_DIR, "market_correlations_2023.png"))
    print()
    print(f"Correlation heatmap saved to {OUTPUT_DIR}/market_correlations_2023.png")

if __name__ == "__main__":
    calculate_market_correlations()
