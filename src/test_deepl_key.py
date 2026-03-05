import os
from dotenv import load_dotenv

ENV_PATH = "config/.env"

def test_deepl_key():
    load_dotenv(ENV_PATH)
    key = os.getenv("DEEPL_API_KEY")
    
    if not key:
        print("Error: DEEPL_API_KEY not found.")
        return

    # Diagnostic output
    print(f"Key Length: {len(key)}")
    print(f"Key Ends with :fx? {key.endswith(':fx')}")
    
    if " " in key:
        print("WARNING: Found spaces in your API key! Please remove them from .env")
    if '"' in key or "'" in key:
        print("WARNING: Found quotes in your API key! Please remove them from .env")

if __name__ == "__main__":
    test_deepl_key()
