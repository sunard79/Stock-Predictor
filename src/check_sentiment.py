import sqlite3

DB_PATH = "database/stocks.db"

def check_sentiment_results():
    """Connects to the database and displays a few analyzed articles."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("PRAGMA table_info(news_articles)")
            columns = [col[1] for col in cursor.fetchall()]
            if 'sentiment_label' not in columns:
                print("Error: 'sentiment_label' column not found.")
                print("Please ensure src/sentiment_analyzer.py ran successfully.")
                return

            print("--- Sample of Analyzed News Articles ---")
            cursor.execute("""
                SELECT title, source, sentiment_label, sentiment_confidence, sentiment_reasoning
                FROM news_articles
                WHERE sentiment_label IS NOT NULL
                LIMIT 10;
            """)
            articles = cursor.fetchall()

            if not articles:
                print("No articles with sentiment analysis found yet.")
                print("Make sure to run news_collector.py and sentiment_analyzer.py first.")
                return

            for i, article in enumerate(articles):
                print("\n--- Article", i + 1, "---")
                print("Title:", article['title'])
                print("Source:", article['source'])
                print("Sentiment:", article['sentiment_label'])
                print(f"Confidence: {article['sentiment_confidence']:.2f}")
                print("Reasoning:", article['sentiment_reasoning'])

            print("\n--- End of Sample ---")

            cursor.execute("SELECT COUNT(*) FROM news_articles WHERE sentiment_label IS NULL")
            remaining = cursor.fetchone()[0]
            print("\n", remaining, "articles still need sentiment analysis.")

    except sqlite3.OperationalError as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    check_sentiment_results()
