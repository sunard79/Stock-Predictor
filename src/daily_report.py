import os
import subprocess
import pandas as pd
from datetime import datetime

def run_step(script_command, description):
    print()
    print(f">>> Step: {description} ({script_command})...")
    try:
        import sys
        # Split command to handle arguments
        cmd_parts = script_command.split()
        script_name = cmd_parts[0]
        args = cmd_parts[1:]
        
        full_cmd = [sys.executable, f"src/{script_name}"] + args
        result = subprocess.run(full_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error in {script_name}: {result.stderr}")
            return False
        print(result.stdout.strip())
        return True
    except Exception as e:
        print(f"Failed to execute {script_name}: {e}")
        return False

def generate_report():
    print("====================================================")
    print(f"STOCK PREDICTOR: DAILY ACTION REPORT - {datetime.now().strftime('%Y-%m-%d')}")
    print("====================================================")

    steps = [
        ("data_collection_multi_asset.py", "Updating Price & Macro Data"),
        ("news_collector.py", "Fetching Live News Headlines"),
        ("local_nlp_analyzer.py", "Running FinBERT Analysis"),
        ("map_news_to_tickers.py", "Mapping News to Tickers"),
        ("sector_sentiment.py", "Calculating Sector Momentum"),
        ("feature_engineering_robust.py", "Building Robust Dataset"),
        ("predict_multi_asset_v2.py --blend-mode agreement", "Generating AI Signals (v2)")
    ]

    for script, desc in steps:
        if not run_step(script, desc):
            print()
            print("!!! Pipeline halted due to error. !!!")
            return

    print()
    print("="*52)
    print("REPORT GENERATION COMPLETE")
    print("Review the 'Next Day Predictions' table above for actions.")
    print("="*52)

if __name__ == "__main__":
    generate_report()
