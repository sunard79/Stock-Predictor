import sqlite3
import os
import deepl
from dotenv import load_dotenv
from tqdm import tqdm

DB_PATH = "database/stocks.db"
ENV_PATH = "config/.env"

def is_chinese(text):
    if not text: return False
    return any('\u4e00' <= char <= '\u9fff' for char in text)

def process_deepl_translations():
    if os.path.exists(ENV_PATH):
        load_dotenv(dotenv_path=ENV_PATH)
    else:
        load_dotenv()

    api_key = os.getenv("DEEPL_API_KEY")
    if not api_key:
        print("Error: DEEPL_API_KEY not found in config/.env")
        return

    # Strip characters that might cause auth issues
    api_key = api_key.strip().replace('"', '').replace("'", "")
    
    # Using the newer DeepLClient
    client = deepl.DeepLClient(api_key)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT id, title, description FROM news_articles")
    articles = cursor.fetchall()

    to_translate = [a for a in articles if is_chinese(a[1])]

    if not to_translate:
        print("No Chinese articles found to translate.")
        conn.close()
        return

    print(f"Translating {len(to_translate)} articles using DeepLClient...")
    
    for art_id, title, desc in tqdm(to_translate):
        try:
            translated_title = client.translate_text(title, target_lang="EN-US").text
            
            translated_desc = ""
            if desc:
                translated_desc = client.translate_text(desc, target_lang="EN-US").text

            cursor.execute("""
                UPDATE news_articles 
                SET title = ?, description = ? 
                WHERE id = ?
            """, (translated_title, translated_desc, art_id))
            conn.commit()
            
        except Exception as e:
            print()
            print(f"Error translating article {art_id}: {e}")
            continue

    conn.close()
    print()
    print("DeepL Translation complete.")

if __name__ == "__main__":
    process_deepl_translations()
