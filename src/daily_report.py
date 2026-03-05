import os
import subprocess
import pandas as pd
from datetime import datetime

def run_step(script_name, description):
    print()
    print(f">>> Step: {description} ({script_name})...")
    try:
        import sys
        result = subprocess.run([sys.executable, f"src/{script_name}"], capture_output=True, text=True)
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
        ("news_collector.py", "Fetching Live RSS News"),
        ("local_nlp_analyzer.py", "Running FinBERT Sentiment Analysis"),
        ("map_news_to_tickers.py", "Mapping News to Assets"),
        ("create_multi_asset_dashboard_data.py", "Updating Dashboard Dataset"),
        ("predict_multi_asset.py", "Generating AI Predictions")
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
