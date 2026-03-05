import os
from dotenv import load_dotenv

ENV_PATH = "config/.env"

def test_key_loading():
    if not os.path.exists(ENV_PATH):
        print(f"Error: .env file not found at {ENV_PATH}")
        return

    load_dotenv(dotenv_path=ENV_PATH)
    key = os.getenv("NEWS_API_KEY")

    if not key:
        print("Error: NEWS_API_KEY not found in .env")
    else:
        # Mask the middle of the key for safety
        masked_key = key[:4] + "*" * (len(key) - 8) + key[-4:] if len(key) > 8 else "****"
        print(f"Key loaded successfully!")
        print(f"Key length: {len(key)}")
        print(f"Masked key: {masked_key}")
        
        if "-" in key:
            print("Warning: Your key contains dashes (-). NewsAPI.org keys usually do not have dashes.")
        if " " in key:
            print("Warning: Your key contains spaces. Please check for leading/trailing spaces in .env")

if __name__ == "__main__":
    test_key_loading()
