import sqlite3
import os
from tqdm import tqdm

DB_PATH = "database/stocks.db"

# Ticker-specific keyword mapping
MAPPING_RULES = {
    "SPY": ["s&p 500", "s&p", "broad market", "us stocks", "wall street", "berkshire hathaway", "warren buffett", "dividend stocks", "u.s. stocks"],
    "QQQ": ["nasdaq", "tech stocks", "technology sector", "nvidia", "ai", "c3 ai", "semiconductor", "artificial intelligence"],
    "DIA": ["dow jones", "dow", "blue chip", "earnings", "quarterly results"],
    "IWM": ["small cap", "russell 2000", "small business", "small investors"],
    "^AXJO": ["australian market", "asx", "australia stocks", "sydney", "australia stock exchange", "magellan financial", "bhp", "rio tinto", "iron ore", "coal exports", "aud", "mortgage rates", "australian housing"],
    "EWA": ["australian market", "asx", "australia stocks", "sydney", "australia stock exchange", "magellan financial", "bhp", "rio tinto", "iron ore", "coal exports", "aud", "mortgage rates", "australian housing"],
    "EWJ": ["japan", "nikkei", "japanese stocks", "asia stocks", "asia markets", "tokyo"],
    "^N225": ["japan", "nikkei", "japanese stocks", "asia stocks", "asia markets", "tokyo"],
    "EWU": ["uk market", "ftse", "british stocks", "european shares", "europe", "london"],
    "EWG": ["germany", "dax", "german stocks", "european shares", "europe", "frankfurt"],
    "MCHI": ["china", "chinese stocks", "hong kong", "asia stocks", "asia markets", "beijing", "shanghai", "pboc", "property", "real estate", "state council", "economic stimulus", "yuan", "rmb", "a-shares", "csi 300", "shenzhen", "chinese yuan", "cnh", "cny"],
    "FXI": ["china", "chinese stocks", "hong kong", "shanghai", "beijing", "china large-cap", "pboc", "property", "state council", "yuan", "rmb", "a-shares", "shanghai composite"],
    "KWEB": ["china tech", "chinese internet", "tencent", "alibaba", "meituan", "jd.com", "technology crackdown", "hang seng tech", "pinduoduo", "xiaomi"],
    "^HSI": ["hong kong", "hang seng", "hsi", "hkex", "victoria harbour"],
    "EWY": ["south korea", "korean stocks", "kospi", "samsung", "seoul"],
    "EPI": ["india", "indian stocks", "nifty", "sensex", "mumbai"],
    "EWZ": ["brazil", "emerging markets", "latin america", "sao paulo"],
    "GLD": ["gold", "gold price", "safe haven", "iran", "strikes", "geopolitical"],
    "SLV": ["silver", "silver price"],
    "USO": ["oil", "crude oil", "oil price", "venezuela", "iran", "strikes"],
    "^VIX": ["volatility", "fear", "market fear", "vix", "risk appetite", "iran", "strikes", "geopolitical"],
    "TLT": ["bonds", "treasuries", "treasury", "risk appetite", "iran", "strikes", "geopolitical"]
}

# Rules for events affecting multiple assets
THEMATIC_RULES = {
    "US_MACRO": {
        "keywords": ["fed", "inflation", "interest rates"],
        "assets": ["SPY", "QQQ", "DIA", "IWM"],
        "reason": "US Macro news (Fed/Inflation) affects all major US indices."
    },
    "AUS_MACRO": {
        "keywords": ["rba", "australian economy"],
        "assets": ["^AXJO", "EWA"],
        "reason": "Australian macro news (RBA) affects domestic tickers."
    },
    "CHINA_MACRO": {
        "keywords": ["pboc", "chinese economy", "stimulus", "property crisis", "yuan", "rmb"],
        "assets": ["MCHI", "FXI", "KWEB", "^HSI"],
        "reason": "Chinese macro/policy news affects all regional trackers."
    },
    "COMMODITIES_SECTOR": {
        "keywords": ["commodities"],
        "assets": ["GLD", "SLV", "USO"],
        "reason": "General commodities news affects gold, silver, and oil."
    },
    "RISK_OFF": {
        "keywords": ["general risk", "market panic", "fear"],
        "assets": ["^VIX", "TLT", "GLD"],
        "reason": "General fear/risk-off sentiment affects safe havens and volatility."
    }
}

def setup_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS news_to_tickers (
            news_id INTEGER,
            ticker TEXT,
            relevance_score REAL,
            mapping_reason TEXT,
            PRIMARY KEY (news_id, ticker),
            FOREIGN KEY (news_id) REFERENCES news_articles(id)
        )
    """)
    conn.commit()
    return conn

def map_article(title, description):
    full_text = f"{title} {description}".lower()
    matches = {}

    for ticker, keywords in MAPPING_RULES.items():
        matched_keywords = [k for k in keywords if k in full_text]
        if matched_keywords:
            matches[ticker] = {
                "score": 0.8 if any(k in title.lower() for k in matched_keywords) else 0.5,
                "reason": f"Matched keywords: {', '.join(matched_keywords)}"
            }

    for theme, config in THEMATIC_RULES.items():
        matched_keywords = [k for k in config['keywords'] if k in full_text]
        if matched_keywords:
            for ticker in config['assets']:
                current_score = matches.get(ticker, {}).get("score", 0)
                new_score = 0.7 if any(k in title.lower() for k in matched_keywords) else 0.4
                if new_score > current_score:
                    matches[ticker] = {
                        "score": new_score,
                        "reason": config['reason']
                    }

    return matches

def main():
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at {DB_PATH}")
        return

    conn = setup_database()
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, description FROM news_articles")
    articles = cursor.fetchall()
    
    print(f"--- Mapping {len(articles)} articles to tickers ---")
    cursor.execute("DELETE FROM news_to_tickers")
    
    ticker_stats = {}
    no_match_count = 0
    
    for news_id, title, desc in tqdm(articles, desc="Mapping"):
        matches = map_article(title, desc if desc else "")
        if not matches:
            no_match_count += 1
            continue
            
        for ticker, info in matches.items():
            cursor.execute("""
                INSERT OR REPLACE INTO news_to_tickers (news_id, ticker, relevance_score, mapping_reason)
                VALUES (?, ?, ?, ?)
            """, (news_id, ticker, info['score'], info['reason']))
            ticker_stats[ticker] = ticker_stats.get(ticker, 0) + 1
            
    conn.commit()
    
    print()
    print("--- Mapping Summary ---")
    print(f"Articles with no match: {no_match_count}")
    print()
    print("Articles per Ticker:")
    for ticker, count in sorted(ticker_stats.items(), key=lambda x: x[1], reverse=True):
        print(f" - {ticker: <6}: {count}")
        
    conn.close()
    print()
    print("Mapping completed and saved to database.")

if __name__ == "__main__":
    main()
