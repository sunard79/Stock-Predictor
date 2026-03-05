import os
import glob

# Files to keep (The Core System)
KEEP_LIST = [
    "predict_multi_asset_v2.py",
    "feature_engineering_robust.py",
    "sector_sentiment.py",
    "data_collection_multi_asset.py",
    "news_collector.py",
    "local_nlp_analyzer.py",
    "map_news_to_tickers.py",
    "import_historical_csv.py",
    "daily_report.py",
    "systematic_backtest.py",
    "walk_forward_validation.py",
    "analyze_2023_response.py",
    "shap_interaction_analysis.py",
    "create_multi_asset_dashboard_data.py"
]

def clean_src():
    src_path = "src"
    if not os.path.exists(src_path):
        print("src folder not found.")
        return

    print("--- Cleaning src/ directory ---")
    
    # Get all .py files in src
    all_files = glob.glob(os.path.join(src_path, "*.py"))
    
    deleted_count = 0
    for file_path in all_files:
        filename = os.path.basename(file_path)
        
        # If it's not in the keep list and not this script itself
        if filename not in KEEP_LIST and filename != "clean_workspace.py":
            try:
                os.remove(file_path)
                print(f"Deleted: {filename}")
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {filename}: {e}")

    print(f"\nCleanup complete. Total files deleted: {deleted_count}")
    print(f"Remaining files: {len(KEEP_LIST)}")

if __name__ == "__main__":
    clean_src()
