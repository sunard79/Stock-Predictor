from eventregistry import *
import sqlite3
import os
from dotenv import load_dotenv

ENV_PATH = "config/.env"
DB_PATH = "database/stocks.db"

def run_targeted_fetch(er, name, keywords, lang, max_items=200):
    print(f"--- Pass: {name} ({len(keywords)} keywords, {lang}) ---")
    q = QueryArticlesIter(
        keywords=QueryItems.OR(keywords),
        dateStart="2026-01-01",
        dateEnd="2026-01-31"
    )
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    added = 0

    for art in q.execQuery(er, maxItems=max_items, lang=lang):
        title = art.get("title")
        desc = art.get("body")[:500] if art.get("body") else ""
        url = art.get("url")
        published = art.get("dateTime")
        source = art.get("source", {}).get("title")

        try:
            cursor.execute("""
                INSERT INTO news_articles (title, description, url, published_date, source)
                VALUES (?, ?, ?, ?, ?)
            """, (title, desc, url, published, source))
            added += 1
        except sqlite3.IntegrityError:
            continue

    conn.commit()
    conn.close()
    print(f"Added {added} articles for {name}.\n")

def main():
    load_dotenv(ENV_PATH)
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key: return
    
    er = EventRegistry(apiKey=api_key)

    # PASS 1: CHINA & HONG KONG (Deep Context)
    china_kw = [
        "China", "PBOC", "Yuan", "Alibaba", "Tencent", 
        "Evergrande", "Shanghai", "HangSeng", "HKEX", "Beijing"
    ]
    run_targeted_fetch(er, "China/HK", china_kw, "eng", max_items=300)

    # PASS 2: JAPAN, KOREA & INDIA
    asia_kw = [
        "Nikkei", "Yen", "Samsung", "KOSPI", "Nifty", "Sensex", 
        "Mumbai", "RBI", "Tokyo", "Asia"
    ]
    run_targeted_fetch(er, "Asian Powerhouses", asia_kw, "eng", max_items=300)

    # PASS 3: MAINLAND CHINESE (Native Language)
    zh_kw = ["股市", "央行", "通胀", "利率", "经济", "半导体", "房地产", "芯片", "华为", "比亚迪"]
    run_targeted_fetch(er, "Mainland China (ZH)", zh_kw, "zho", max_items=200)

if __name__ == "__main__":
    main()
