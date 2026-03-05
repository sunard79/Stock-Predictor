import sqlite3
import os
import time
from google import genai
from dotenv import load_dotenv
from tqdm import tqdm

DB_PATH = "database/stocks.db"
ENV_PATH = "config/.env"

def is_chinese(text):
    if not text: return False
    return any('\u4e00' <= char <= '\u9fff' for char in text)

def translate_article(client, title, description):
    prompt = f"""
    Translate the following Chinese financial news into English. 
    Maintain professional financial terminology.
    Format your response as: TITLE: [translated title] | BODY: [translated body]
    
    TITLE: {title}
    BODY: {description}
    """
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.0-flash', 
                contents=prompt
            )
            text = response.text.strip()
            text = text.replace("```json", "").replace("```", "").strip()
            
            if "TITLE:" in text and "BODY:" in text:
                parts = text.split("|")
                t = parts[0].replace("TITLE:", "").strip()
                b = parts[1].replace("BODY:", "").strip()
                return t, b
            return text, ""
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = (attempt + 1) * 60 # Wait 1, 2, 3... minutes
                print(f"\n[Rate Limit] Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                print(f"Translation error: {e}")
                return None, None
    return None, None

def process_translations():
    load_dotenv(ENV_PATH)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: return
    
    client = genai.Client(api_key=api_key)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, title, description FROM news_articles")
    articles = cursor.fetchall()
    
    to_translate = []
    for art_id, title, desc in articles:
        if is_chinese(title):
            to_translate.append((art_id, title, desc))
            
    if not to_translate:
        print("No Chinese articles found to translate.")
        conn.close()
        return

    print(f"Translating {len(to_translate)} articles using Gemini 1.5 Flash...")
    for art_id, title, desc in tqdm(to_translate):
        new_title, new_desc = translate_article(client, title, desc)
        if new_title:
            cursor.execute("""
                UPDATE news_articles 
                SET title = ?, description = ? 
                WHERE id = ?
            """, (new_title, new_desc, art_id))
            conn.commit()
        time.sleep(4) # Standard safety delay

    conn.close()
    print()
    print("Translation complete.")

if __name__ == "__main__":
    process_translations()
