from eventregistry import *
import os
from dotenv import load_dotenv

ENV_PATH = "config/.env"

def test_api():
    load_dotenv(ENV_PATH)
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        print("NEWS_API_KEY not found.")
        return

    er = EventRegistry(apiKey=api_key)
    
    # Test a single broad query for 2023 SVB crisis
    print("Testing broad historical query for March 2023...")
    q = QueryArticlesIter(
        keywords=QueryItems.OR(["Silicon Valley Bank", " SVB "]),
        dateStart="2023-03-10",
        dateEnd="2023-03-15"
    )
    
    count = 0
    try:
        for art in q.execQuery(er, maxItems=5):
            print(f"Found: {art.get('title')}")
            count += 1
        
        if count == 0:
            print("❌ No articles found. Your API plan likely limits historical access to the last 30-90 days.")
        else:
            print(f"✅ Success! Found {count} sample articles.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api()
