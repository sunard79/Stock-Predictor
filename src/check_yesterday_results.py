import sqlite3
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

DB_PATH = "database/stocks.db"

def check_results():
    conn = sqlite3.connect(DB_PATH)
    
    # 1. Get predictions from March 4th
    print("--- Predictions from March 4, 2026 ---")
    try:
        preds = pd.read_sql_query("SELECT * FROM daily_predictions_history WHERE prediction_date = '2026-03-04'", conn)
        if preds.empty:
            print("No predictions found for March 4th. Checking March 3rd...")
            preds = pd.read_sql_query("SELECT * FROM daily_predictions_history WHERE prediction_date = '2026-03-03'", conn)
            if preds.empty:
                print("No recent predictions found in daily_predictions_history.")
                conn.close()
                return
            else:
                target_date = "2026-03-03"
        else:
            target_date = "2026-03-04"
            
        print(preds[['Ticker', 'Market', 'Signal', 'Pred_Z']].to_string(index=False))
        
        # 2. Get today's results (March 5th)
        today_date = "2026-03-05"
        tickers = preds['Ticker'].unique().tolist()
        
        print(f"\n--- Results for {today_date} (compared to {target_date} Close) ---")
        
        results = []
        for ticker in tickers:
            # Fetch data from Yahoo Finance for the period to ensure we have today's price
            start_fetch = (datetime.strptime(target_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
            # We fetch up to tomorrow to ensure we get today's data if available
            end_fetch = (datetime.strptime(today_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            
            df = yf.download(ticker, start=start_fetch, end=end_fetch, progress=False)
            if df.empty:
                continue
            
            df = df.sort_index()
            
            # Get target date close and today's close
            try:
                # Handle potential MultiIndex columns from newer yfinance
                close_col = ('Close', ticker) if isinstance(df.columns, pd.MultiIndex) else 'Close'
                
                # Check if target date and today exist in the index
                # Sometimes yfinance doesn't return exactly what we ask if markets are closed
                available_dates = df.index.strftime('%Y-%m-%d').tolist()
                
                if target_date not in available_dates:
                    # Find the last available date before or on target_date
                    target_dt = datetime.strptime(target_date, "%Y-%m-%d")
                    prev_dates = [d for d in df.index if d <= target_dt]
                    if not prev_dates: continue
                    target_idx_dt = prev_dates[-1]
                else:
                    target_idx_dt = pd.Timestamp(target_date)
                    
                if today_date not in available_dates:
                    # Use the latest available price
                    today_idx_dt = df.index[-1]
                else:
                    today_idx_dt = pd.Timestamp(today_date)
                
                if target_idx_dt == today_idx_dt:
                    continue # Same day, can't calculate return
                
                price_prev = float(df.loc[target_idx_dt, close_col])
                price_today = float(df.loc[today_idx_dt, close_col])
                
                ret = (price_today - price_prev) / price_prev
                
                pred_row = preds[preds['Ticker'] == ticker].iloc[0]
                signal = pred_row['Signal']
                
                is_correct = False
                if signal == 'BUY' and ret > 0: is_correct = True
                elif signal == 'SELL' and ret < 0: is_correct = True
                elif signal == 'NEUTRAL': is_correct = True
                
                results.append({
                    'Ticker': ticker,
                    'Signal': signal,
                    'Return': ret,
                    'Status': 'CORRECT' if is_correct else 'WRONG'
                })
            except Exception as e:
                # print(f"Error processing {ticker}: {e}")
                continue
                
        if results:
            res_df = pd.DataFrame(results)
            print(res_df.to_string(index=False))
            
            accuracy = (res_df['Status'] == 'CORRECT').mean()
            print(f"\nOverall Accuracy: {accuracy:.1%}")
        else:
            print("Could not calculate returns for any tickers. Markets might not be open yet or data not available.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    check_results()
