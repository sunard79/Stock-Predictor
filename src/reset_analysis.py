import sqlite3
import os

DB_PATH = "database/stocks.db"

def reset_analysis_for_translated():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Reset analysis for articles that were recently translated or had Chinese source names
    print("Resetting old analysis to allow re-analysis of English translations...")
    
    query = """
        DELETE FROM news_analysis 
        WHERE article_id IN (
            SELECT id FROM news_articles 
            WHERE source IN ('Caixin', 'Wall Street CN', 'Sina Finance', 'East Money', '中国经济网')
            OR title NOT GLOB '*[一-龥]*' -- Articles that are now English
        )
    """
    cursor.execute(query)
    count = cursor.rowcount
    conn.commit()
    conn.close()
    print(f"Cleared {count} analysis entries. You can now run the local analyzer.")

if __name__ == "__main__":
    reset_analysis_for_translated()
