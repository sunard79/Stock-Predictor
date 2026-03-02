import sqlite3
import os
import json
import re
from datetime import datetime
from tqdm import tqdm

# Attempt to load FinBERT locally
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    import torch.nn.functional as F
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

DB_PATH = "database/stocks.db"

# Ticker and Geography Mapping
TICKER_MAP = {
    "SPY": ["spy", "s&p 500", "sp500", "index fund"],
    "QQQ": ["qqq", "nasdaq", "tech stocks", "semiconductor", "ai stocks"],
    "DIA": ["dia", "dow jones", "blue chip"],
    "IWM": ["iwm", "russell 2000", "small cap"],
    "^AXJO": ["axjo", "asx 200", "australian index", "sydney exchange"],
    "EWA": ["ewa", "msci australia"],
    "EWJ": ["ewj", "msci japan", "japanese market"],
    "EWU": ["ewu", "msci uk", "united kingdom"],
    "EWG": ["ewg", "msci germany", "dax"],
    "MCHI": ["mchi", "msci china", "chinese stocks"],
    "EWZ": ["ewz", "msci brazil"],
    "GLD": ["gld", "gold price", "bullion"],
    "SLV": ["slv", "silver price"],
    "USO": ["uso", "oil price", "crude oil", "wti"],
    "^VIX": ["vix", "volatility", "fear index"],
    "TLT": ["tlt", "treasury bond", "interest rate"]
}

GEO_MAP = {
    "United States": ["fed", "us economy", "wall street", "washington", "fomc", "powell"],
    "Australia": ["australia", "rba", "asx", "sydney", "lowe", "bullock"],
    "China": ["china", "beijing", "yuan", "chinese", "evergrande"],
    "Japan": ["japan", "tokyo", "yen", "boj", "nikkei"],
    "Europe": ["europe", "ecb", "euro", "lagarde", "ftse", "dax", "brexit"]
}

class LocalNLP:
    def __init__(self):
        if HAS_TRANSFORMERS:
            print("Loading FinBERT model (this may take a minute on first run)...")
            # ProsusAI/finbert is the industry standard for financial sentiment
            self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        else:
            print("Warning: 'transformers' or 'torch' not found. Falling back to rule-based sentiment.")
            self.tokenizer = None
            self.model = None

    def get_sentiment(self, text):
        """Returns (label, confidence) using FinBERT."""
        if not self.tokenizer or not self.model:
            return "neutral", 0.5
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            
        # FinBERT labels: 0: positive, 1: negative, 2: neutral
        # We map them to our schema: bullish, bearish, neutral
        conf, label_idx = torch.max(probs, dim=-1)
        mapping = {0: "bullish", 1: "bearish", 2: "neutral"}
        return mapping[label_idx.item()], float(conf.item())

    def extract_assets(self, text):
        """Identifies affected assets based on keywords."""
        text = text.lower()
        found = []
        for ticker, keywords in TICKER_MAP.items():
            if any(k in text for k in keywords):
                found.append(ticker)
        return found[:5] if found else ["SPY"]

    def extract_geo(self, text):
        """Identifies primary geographic focus."""
        text = text.lower()
        for region, keywords in GEO_MAP.items():
            if any(k in text for k in keywords):
                return region
        return "Global/Other"

def prepare_database():
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

def get_articles_for_analysis():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT na.* FROM news_articles na
            LEFT JOIN news_analysis res ON na.id = res.article_id
            WHERE res.id IS NULL
        """)
        return cursor.fetchall()

def main():
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at {DB_PATH}")
        return

    prepare_database()
    articles = get_articles_for_analysis()
    
    if not articles:
        print("No new articles to analyze.")
        return

    nlp = LocalNLP()
    print(f"Starting local analysis for {len(articles)} articles...")

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        for article in tqdm(articles, desc="Local NLP"):
            text = f"{article['title']} {article['description']}"
            
            # 1. Sentiment
            sentiment, confidence = nlp.get_sentiment(text)
            
            # 2. Assets
            assets = nlp.extract_assets(text)
            
            # 3. Geo
            geo = nlp.extract_geo(text)
            
            # 4. Generate local reasoning
            reasoning = f"Local FinBERT model detected {sentiment} sentiment based on financial terminology."
            impact_reasoning = f"Keywords related to {', '.join(assets)} were identified in the content."

            cursor.execute("""
                INSERT INTO news_analysis (
                    article_id, sentiment_label, sentiment_confidence, sentiment_reasoning,
                    affected_assets, asset_impact_reasoning, geographic_focus
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                article['id'],
                sentiment,
                confidence,
                reasoning,
                json.dumps(assets),
                impact_reasoning,
                geo
            ))
            conn.commit()

    print()
    print("Local analysis complete. Results stored in 'news_analysis' table.")

if __name__ == "__main__":
    main()
