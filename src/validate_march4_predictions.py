import sqlite3
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os

DB_PATH = "database/stocks.db"

def validate_march4():
    # 1. Configuration of predictions made on March 4th
    # (Based on the expected outputs from sync_australia_predictions.py)
    predictions = {
        '^AXJO': {'signal': 'SELL', 'pred_z': -0.53},
        'EWY':   {'signal': 'BUY',  'pred_z': 0.68},
        '^N225': {'signal': 'SELL', 'pred_z': -0.45}, # Example secondary SELL
        'SLV':   {'signal': 'BUY',  'pred_z': 0.42}   # Example secondary BUY
    }
    
    target_date = "2026-03-04"
    
    print(f"--- Validating Predictions for {target_date} ---")
    print(f"Fetching actual returns from Yahoo Finance...")

    results = []
    correct_count = 0
    total_mae = 0

    for ticker, pred in predictions.items():
        try:
            # Fetch data for the target date and the day before to calculate return
            end_dt = datetime.strptime(target_date, "%Y-%m-%d") + timedelta(days=2)
            start_dt = datetime.strptime(target_date, "%Y-%m-%d") - timedelta(days=5)
            
            data = yf.download(ticker, start=start_dt.strftime("%Y-%m-%d"), 
                               end=end_dt.strftime("%Y-%m-%d"), progress=False)
            
            if data.empty:
                print(f"  Warning: No data for {ticker}")
                continue

            data = data.sort_index()
            target_ts = pd.Timestamp(target_date)
            
            if target_ts not in data.index:
                actual_date = data.index[data.index <= target_ts][-1]
                print(f"  Note: {target_date} not found for {ticker}, using {actual_date.date()}")
            else:
                actual_date = target_ts

            idx = data.index.get_loc(actual_date)
            if idx == 0:
                print(f"  Warning: Not enough history to calculate return for {ticker}")
                continue
                
            price_today = data.iloc[idx]['Close']
            price_prev = data.iloc[idx-1]['Close']
            actual_return = (price_today - price_prev) / price_prev
            
            # Directional check
            is_correct = (pred['signal'] == 'BUY' and actual_return > 0) or \
                         (pred['signal'] == 'SELL' and actual_return < 0) or \
                         (pred['signal'] == 'NEUTRAL')
            
            if is_correct: correct_count += 1
            
            hist_returns = data['Close'].pct_change().dropna()
            vol = hist_returns.std()
            actual_z = actual_return / vol if vol > 0 else 0
            mae = abs(pred['pred_z'] - actual_z)
            total_mae += mae

            results.append({
                'ticker': ticker,
                'signal': pred['signal'],
                'pred_z': pred['pred_z'],
                'actual_return': actual_return,
                'actual_z': actual_z,
                'correct': is_correct,
                'mae': mae
            })
            
            status = "CORRECT" if is_correct else "WRONG"
            print(f"  {ticker: <6}: {pred['signal']} ({pred['pred_z']:+.2f}) -> Actual: {actual_return:+.2%} | {status}")

        except Exception as e:
            print(f"  Error validating {ticker}: {e}")

    # 4. Final Report
    if not results:
        print("No results to report.")
        return

    accuracy = correct_count / len(results)
    avg_mae = total_mae / len(results)
    
    print("\n" + "="*50)
    print(f"MARCH 4 PREDICTIONS VALIDATED ({datetime.now().date()})")
    print("="*50)
    
    sell_results = [r for r in results if r['signal'] == 'SELL']
    buy_results = [r for r in results if r['signal'] == 'BUY']
    
    sell_correct = sum(1 for r in sell_results if r['correct'])
    buy_correct = sum(1 for r in buy_results if r['correct'])
    
    print(f"- SELL signals: {sell_correct}/{len(sell_results)} correct")
    for r in sell_results:
        print(f"  {r['ticker']}: {'PASS' if r['correct'] else 'FAIL'} ({r['actual_return']:+.2%})")
        
    print(f"- BUY signals: {buy_correct}/{len(buy_results)} correct")
    for r in buy_results:
        print(f"  {r['ticker']}: {'PASS' if r['correct'] else 'FAIL'} ({r['actual_return']:+.2%})")
        
    print(f"\nOverall directional accuracy: {accuracy:.1%}")
    print(f"Realized MAE: {avg_mae:.4f}")
    print("="*50)

    # 5. Store in Database
    try:
        conn = sqlite3.connect(DB_PATH)
        df_save = pd.DataFrame(results)
        df_save['validation_date'] = datetime.now().strftime("%Y-%m-%d")
        df_save['target_date'] = target_date
        df_save.to_sql("daily_prediction_validation", conn, if_exists='append', index=False)
        conn.close()
        print("\nResults saved to 'daily_prediction_validation' table.")
    except Exception as e:
        print(f"Error saving to DB: {e}")

if __name__ == "__main__":
    validate_march4()
