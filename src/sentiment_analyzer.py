import sqlite3
import os
from google import genai
from dotenv import load_dotenv
from tqdm import tqdm
import json
import time

DB_PATH = "database/stocks.db"
ENV_PATH = "config/.env"

def load_api_key():
    """Loads the Gemini API key from the .env file and returns a genai.Client instance."""
    if not os.path.exists(ENV_PATH):
        print(f"Error: The environment file was not found at {ENV_PATH}")
        return None
    
    load_dotenv(dotenv_path=ENV_PATH)
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key or api_key == "YOUR_GEMINI_API_KEY":
        print("Error: GEMINI_API_KEY not found or not set in config/.env.")
        return None
    
    return genai.Client(api_key=api_key)

def prepare_database():
    """
    Ensures the 'news_articles' table has the correct schema for sentiment analysis.
    """
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(news_articles)")
        columns = [column[1] for column in cursor.fetchall()]

        if 'sentiment_label' not in columns:
            print("Updating database schema for sentiment analysis.")
            # This part handles migration from an older schema if 'sentiment' column exists.
            if 'sentiment' in columns:
                cursor.execute("ALTER TABLE news_articles RENAME TO news_articles_old;")
                cursor.execute("""
                    CREATE TABLE news_articles (
                        id INTEGER PRIMARY KEY,
                        title TEXT NOT NULL,
                        description TEXT,
                        url TEXT UNIQUE NOT NULL,
                        published_date DATETIME NOT NULL,
                        source TEXT NOT NULL,
                        sentiment_label TEXT,
                        sentiment_confidence REAL,
                        sentiment_reasoning TEXT
                    );
                """)
                cursor.execute("""
                    INSERT INTO news_articles (id, title, description, url, published_date, source)
                    SELECT id, title, description, url, published_date, source FROM news_articles_old;
                """)
                cursor.execute("DROP TABLE news_articles_old;")
                print("Database schema updated from old 'sentiment' column.")
            else:
                # If no sentiment column at all, just add the new ones.
                cursor.execute("ALTER TABLE news_articles ADD COLUMN sentiment_label TEXT;")
                cursor.execute("ALTER TABLE news_articles ADD COLUMN sentiment_confidence REAL;")
                cursor.execute("ALTER TABLE news_articles ADD COLUMN sentiment_reasoning TEXT;")
                print("Database schema updated with new sentiment columns.")
        else:
            print("Database schema is already up to date.")

def get_articles_for_analysis():
    """Fetches articles from the database that have not been analyzed yet."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM news_articles WHERE sentiment_label IS NULL")
        articles = cursor.fetchall()
        print(f"Found {len(articles)} articles to analyze.")
        return articles

def analyze_sentiment(client, article_text):
    """
    Analyzes the sentiment of a given text using the Gemini API.
    """
    prompt = f"""
    Analyze the sentiment of the following financial news article regarding the stock market, specifically its potential impact on the S&P 500 (SPY). The article content is:

    ---
    {article_text}
    ---

    Provide your analysis in a structured JSON format. The JSON object must contain these three keys:
    1. "sentiment": A string, which must be one of the following: "bullish", "bearish", or "neutral".
    2. "confidence": A float between 0.0 and 1.0, representing your confidence in the sentiment analysis.
    3. "reasoning": A brief, one-sentence explanation for your sentiment classification.

    Do not include any text outside of the JSON object.
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        time.sleep(12)  # Respect rate limit of 5 requests/minute
        # Assuming the response text is directly JSON or wrapped in ```json ... ```
        json_str = response.text.strip()
        if json_str.startswith("```json") and json_str.endswith("```"):
            json_str = json_str[len("```json"): -len("```")].strip()
        
        result = json.loads(json_str)
        if all(k in result for k in ['sentiment', 'confidence', 'reasoning']):
            return result
        else:
            print(f"Warning: API response missing required keys. Full response: {response.text}")
            return None
    except Exception as e:
        print(f"An error occurred during API call or JSON parsing: {e}")
        time.sleep(5) 
        return None

def update_article_sentiment(article_id, sentiment_data):
    """Updates the sentiment for a specific article in the database."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE news_articles
            SET sentiment_label = ?, sentiment_confidence = ?, sentiment_reasoning = ?
            WHERE id = ?
        """, (sentiment_data['sentiment'], sentiment_data['confidence'], sentiment_data['reasoning'], article_id))
        conn.commit()

if __name__ == "__main__":
    client = load_api_key()
    if client is None:
        exit()

    prepare_database()
    articles = get_articles_for_analysis()

    if articles:
        print(f"\nStarting sentiment analysis for {len(articles)} articles...")
        for article in tqdm(articles, desc="Analyzing sentiments"):
            article_content = f"Title: {article['title']}\n\n{article['description']}"
            
            print(f"\n--- Analyzing Article ID: {article['id']} - '{article['title'][:70]}...' ---")
            sentiment_result = analyze_sentiment(client, article_content)

            if sentiment_result:
                update_article_sentiment(article['id'], sentiment_result)
                print(f"  Sentiment: {sentiment_result['sentiment']}")
                print(f"  Confidence: {sentiment_result['confidence']:.2f}")
                print(f"  Reasoning: {sentiment_result['reasoning']}")
            else:
                print("  Sentiment analysis failed for this article.")
        
        print("\nSentiment analysis complete.")
    else:
        print("No new articles to analyze.")
