import sqlite3
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

DB_PATH = "database/stocks.db"
MODEL_NAME = "ProsusAI/finbert"

def prepare_database():
    """Ensures the 'news_analysis' table exists with the correct columns."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS news_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_id INTEGER,
                sentiment_label TEXT,
                sentiment_score REAL,
                sentiment_confidence REAL,
                weighted_sentiment REAL,
                provider TEXT DEFAULT 'finbert',
                analyzed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (article_id) REFERENCES news_articles(id)
            );
        """)
        
        # Ensure 'weighted_sentiment' and 'provider' exist if table was created differently
        cursor.execute("PRAGMA table_info(news_analysis)")
        cols = [info[1] for info in cursor.fetchall()]
        if 'weighted_sentiment' not in cols:
            cursor.execute("ALTER TABLE news_analysis ADD COLUMN weighted_sentiment REAL")
        if 'provider' not in cols:
            cursor.execute("ALTER TABLE news_analysis ADD COLUMN provider TEXT DEFAULT 'finbert'")
        
        conn.commit()
    print("Database ready for FinBERT analysis.")

def get_unanalyzed_articles():
    """Fetches articles that haven't been analyzed by FinBERT yet."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT na.* FROM news_articles na
            WHERE na.id NOT IN (SELECT article_id FROM news_analysis WHERE provider = 'finbert')
        """)
        return cursor.fetchall()

def run_finbert():
    prepare_database()
    articles = get_unanalyzed_articles()
    
    if not articles:
        print("No new articles to analyze with FinBERT.")
        return

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print(f"Analyzing {len(articles)} articles on {device}...")
    
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        
        for art in tqdm(articles):
            # Ticker articles have 'description' and 'title'
            text = art['description'] or art['title']
            if not text:
                continue
                
            # Truncate text to fit model context window (usually 512 tokens)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            # FinBERT labels: 0 -> positive, 1 -> negative, 2 -> neutral
            pos_prob = probs[0][0].item()
            neg_prob = probs[0][1].item()
            neu_prob = probs[0][2].item()
            
            # Map to labels
            labels = ["bullish", "bearish", "neutral"]
            best_idx = torch.argmax(probs, dim=-1).item()
            sentiment_label = labels[best_idx]
            sentiment_confidence = probs[0][best_idx].item()
            
            # Calculate weighted sentiment score (-1 to 1)
            # Standard FinBERT mapping: positive - negative
            weighted_sentiment = pos_prob - neg_prob
            
            try:
                cursor.execute("""
                    INSERT INTO news_analysis 
                    (article_id, sentiment_label, sentiment_score, sentiment_confidence, weighted_sentiment, provider)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    art['id'],
                    sentiment_label,
                    weighted_sentiment, # redundant but safe
                    sentiment_confidence,
                    weighted_sentiment,
                    'finbert'
                ))
                conn.commit()
            except Exception as e:
                print(f"Error saving analysis for ID {art['id']}: {e}")

    print("\nFinBERT analysis complete.")

if __name__ == "__main__":
    run_finbert()
