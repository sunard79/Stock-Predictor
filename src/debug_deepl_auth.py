import os
import deepl
from dotenv import load_dotenv

ENV_PATH = "config/.env"

def debug_key():
    if not os.path.exists(ENV_PATH):
        print("Error: .env not found")
        return

    load_dotenv(ENV_PATH)
    key = os.getenv("DEEPL_API_KEY")
    
    if not key:
        print("Error: Key not found in environment")
        return

    print(f"Loaded Key: '{key}'")
    print(f"Length: {len(key)}")
    
    # Try a simple usage check to verify auth
    try:
        translator = deepl.Translator(key)
        usage = translator.get_usage()
        if usage.any_limit_reached:
            print("Status: Key is VALID but limit reached.")
        else:
            print("Status: Key is VALID and working!")
            print(f"Character usage: {usage.character.count} of {usage.character.limit}")
    except Exception as e:
        print(f"Status: FAILED - {e}")

if __name__ == "__main__":
    debug_key()
