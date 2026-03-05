import sqlite3
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os

DB_PATH = "database/stocks.db"

def validate_history():
    if not os.path.exists(DB_PATH):
        print("Database not found.")
        return

    conn = sqlite3.connect(DB_PATH)
    try:
        # Load saved predictions
        query = "SELECT * FROM daily_predictions_history"
        history_df = pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"Error loading prediction history: {e}")
        conn.close()
        return
    conn.close()

    if history_df.empty:
        print("No prediction history found. Run src/predict_multi_asset_v2.py first.")
        return

    # Convert date and numeric columns
    history_df['prediction_date'] = pd.to_datetime(history_df['prediction_date'])
    history_df['Pred_Z'] = pd.to_numeric(history_df['Pred_Z'])
    
    available_dates = sorted(history_df['prediction_date'].unique())
    
    print(f"History contains predictions for {len(available_dates)} dates.")
    
    for target_date in available_dates:
        print(f"\n--- Validating signals from {target_date.date()} ---")
        
        day_preds = history_df[history_df['prediction_date'] == target_date]
        tickers = day_preds['Ticker'].tolist()
        
        # Download data for these tickers around that date
        start_fetch = target_date - timedelta(days=5)
        end_fetch = target_date + timedelta(days=5)
        
        data = yf.download(tickers, start=start_fetch, end=end_fetch, group_by='ticker', progress=False)
        
        results = []
        for _, row in day_preds.iterrows():
            ticker = row['Ticker']
            signal = row['Signal']
            
            try:
                # Get specific ticker data
                if len(tickers) > 1:
                    t_data = data[ticker].dropna()
                else:
                    t_data = data.dropna()
                
                if target_date not in t_data.index:
                    valid_indices = t_data.index[t_data.index >= target_date]
                    if len(valid_indices) == 0: continue
                    p_date = valid_indices[0]
                else:
                    p_date = target_date
                
                idx = t_data.index.get_loc(p_date)
                if idx + 1 >= len(t_data):
                    print(f"  {ticker}: No follow-up data yet (Market might still be open or tomorrow is a holiday)")
                    continue
                
                price_now = t_data.iloc[idx]['Close']
                price_next = t_data.iloc[idx+1]['Close']
                actual_return = (price_next - price_now) / price_now
                
                is_correct = (signal == 'BUY' and actual_return > 0) or \
                             (signal == 'SELL' and actual_return < 0) or \
                             (signal == 'NEUTRAL')
                
                results.append({
                    'Ticker': ticker,
                    'Signal': signal,
                    'Return': f"{actual_return:+.2%}",
                    'Result': "WIN" if is_correct else "LOSS"
                })
            except Exception:
                continue
        
        if results:
            res_df = pd.DataFrame(results)
            win_rate = (res_df['Result'] == 'WIN').mean()
            print(res_df.to_string(index=False))
            print(f"Win Rate for this batch: {win_rate:.1%}")
        else:
            print("  No validation data available for this date yet.")

if __name__ == "__main__":
    validate_history()
