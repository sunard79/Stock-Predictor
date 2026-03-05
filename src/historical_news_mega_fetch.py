from eventregistry import *
import sqlite3
import os
from dotenv import load_dotenv

ENV_PATH = "config/.env"
DB_PATH = "database/stocks.db"

def run_targeted_fetch(er, name, keywords, lang, start, end, max_items=500):
    print(f"--- Pass: {name} ({start} to {end}) ---")
    q = QueryArticlesIter(
        keywords=QueryItems.OR(keywords),
        dateStart=start,
        dateEnd=end
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
    if not api_key: 
        print("Missing NEWS_API_KEY in .env")
        return
    
    er = EventRegistry(apiKey=api_key)

    # 1. 2022 Rate Hike Cycle (Focus on March-June when hikes began)
    macro_kw = ["Federal Reserve", "Inflation", "Interest Rate", "Powell", "CPI"]
    run_targeted_fetch(er, "2022 Rate Hikes (Spring)", macro_kw, "eng", "2022-03-01", "2022-06-30")

    # 2. 2023 Banking Crisis (SVB Collapse)
    crisis_kw = ["SVB", "Silicon Valley Bank", "Bank Run", "Credit Suisse", "Contagion", "Bailout"]
    run_targeted_fetch(er, "2023 Banking Crisis", crisis_kw, "eng", "2023-03-08", "2023-04-30")

    # 3. 2024 AI Rally (Nvidia Earnings breakout)
    ai_kw = ["Nvidia", "AI", "Artificial Intelligence", "Semiconductor", "Tech Rally"]
    run_targeted_fetch(er, "2024 AI Rally", ai_kw, "eng", "2024-02-01", "2024-05-30")

if __name__ == "__main__":
    main()
