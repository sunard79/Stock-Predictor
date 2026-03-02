import sqlite3
import os
from google import genai
from dotenv import load_dotenv
from tqdm import tqdm
import json
import time

DB_PATH = "database/stocks.db"
ENV_PATH = "config/.env"

# All supported tickers for asset impact analysis
ALL_ASSETS = ["SPY", "QQQ", "DIA", "IWM", "^AXJO", "EWA", "EWJ", "EWU", "EWG", "MCHI", "EWZ", "GLD", "SLV", "USO", "^VIX", "TLT"]

def load_api_key():
    """Loads the Gemini API key from the .env file and returns a genai.Client instance."""
    if not os.path.exists(ENV_PATH):
        print(f"Error: The environment file was not found at {ENV_PATH}")
        return None
    
    load_dotenv(dotenv_path=ENV_PATH)
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("Error: GEMINI_API_KEY not found in config/.env.")
        return None
    
    return genai.Client(api_key=api_key)

def prepare_database():
    """
    Ensures the 'news_analysis' table exists with the correct schema for enhanced analysis.
    """
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS news_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_id INTEGER,
                sentiment_label TEXT,
                sentiment_confidence REAL,
                sentiment_reasoning TEXT,
                affected_assets TEXT,
                asset_impact_reasoning TEXT,
                geographic_focus TEXT,
                analyzed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (article_id) REFERENCES news_articles(id)
            );
        """)
        conn.commit()
    print("Table 'news_analysis' is ready.")

def get_articles_for_analysis():
    """Fetches articles from 'news_articles' that haven't been analyzed in 'news_analysis'."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        # Find articles that don't have a corresponding entry in news_analysis
        cursor.execute("""
            SELECT na.* FROM news_articles na
            LEFT JOIN news_analysis res ON na.id = res.article_id
            WHERE res.id IS NULL
        """)
        articles = cursor.fetchall()
        print(f"Found {len(articles)} articles to analyze.")
        return articles

def detect_geographic_focus_keywords(text):
    """Simple rule-based geographic focus detection as a fallback or pre-check."""
    text = text.lower()
    focus = []
    if any(k in text for k in ['australia', 'rba', 'asx', 'sydney']):
        focus.append("Australia")
    if any(k in text for k in ['fed', 'us economy', 'wall street', 'washington']):
        focus.append("United States")
    if any(k in text for k in ['china', 'japan', 'asia', 'pacific', 'beijing', 'tokyo']):
        focus.append("Asia")
    return ", ".join(focus) if focus else "Global/Other"

def analyze_article_enhanced(client, article_text):
    """
    Analyzes the article using Gemini with enhanced fields for asset impact and geographic focus.
    Includes retry logic for rate limits.
    """
    asset_list_str = ", ".join(ALL_ASSETS)
    
    prompt = f"""
    Analyze the following financial news article for sentiment and market impact.
    The article content is:
    ---
    {article_text}
    ---

    Task:
    1. Determine overall sentiment (bullish, bearish, neutral).
    2. Which of these assets are most affected: {asset_list_str}? 
       Identify top 3-5 assets and provide a brief explanation for each.
    3. Identify the primary geographic focus (e.g., United States, Australia, China, Europe, etc.).
    4. Provide reasoning for your sentiment classification.

    Respond STRICTLY in JSON format with these keys:
    - "sentiment": "bullish" | "bearish" | "neutral"
    - "confidence": float (0.0 to 1.0)
    - "reasoning": "brief explanation for sentiment"
    - "affected_assets": [list of 3-5 tickers]
    - "asset_impact_reasoning": "brief explanation of why each identified asset is affected"
    - "geographic_focus": "primary region/country affected"

    Do not include any text outside of the JSON object.
    """
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt
            )
            
            json_str = response.text.strip()
            if json_str.startswith("```json") and json_str.endswith("```"):
                json_str = json_str[len("```json"): -len("```")].strip()
            
            result = json.loads(json_str)
            time.sleep(12)  # Respect rate limit after success
            return result
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                wait_time = 60 * (attempt + 1)
                print(f"\n[Rate Limit] Attempt {attempt+1} failed. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"\n[Error] API call failed: {e}")
                time.sleep(5)
                return None
    return None

def store_analysis(article_id, analysis):
    """Stores the enhanced analysis result in the news_analysis table."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        # Convert list to JSON string for storage
        affected_assets_json = json.dumps(analysis.get('affected_assets', []))
        
        cursor.execute("""
            INSERT INTO news_analysis (
                article_id, sentiment_label, sentiment_confidence, sentiment_reasoning,
                affected_assets, asset_impact_reasoning, geographic_focus
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            article_id,
            analysis.get('sentiment'),
            analysis.get('confidence'),
            analysis.get('reasoning'),
            affected_assets_json,
            analysis.get('asset_impact_reasoning'),
            analysis.get('geographic_focus')
        ))
        conn.commit()

def main():
    client = load_api_key()
    if not client:
        return

    prepare_database()
    articles = get_articles_for_analysis()

    if not articles:
        print("No new articles to analyze.")
        return

    print(f"Starting enhanced analysis for {len(articles)} articles...")
    
    for article in tqdm(articles, desc="Processing Articles"):
        article_content = f"Title: {article['title']}\n\n{article['description']}"
        
        # We use Gemini for everything, but could combine with local rules if needed.
        analysis_result = analyze_article_enhanced(client, article_content)
        
        if analysis_result:
            store_analysis(article['id'], analysis_result)
            print()
            print(f"+ Analyzed: {article['title'][:50]}...")
            print(f"  Focus: {analysis_result.get('geographic_focus')}")
            print(f"  Impact: {analysis_result.get('affected_assets')}")
        else:
            print()
            print(f"- Failed to analyze article: {article['id']}")

if __name__ == "__main__":
    main()
