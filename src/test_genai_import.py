import google.generativeai as genai
import os
from dotenv import load_dotenv

DB_PATH = "database/stocks.db" # Not strictly needed for this test, but kept for context
ENV_PATH = "config/.env"

def test_genai_import():
    load_dotenv(dotenv_path=ENV_PATH)
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("API Key not found in config/.env. Please ensure it's set.")
        return

    print("--- Testing google.generativeai import ---")
    print("Type of 'genai' object:", type(genai))
    print("Attributes of 'genai' object:", sorted(dir(genai)))
    print("-" * 40)

    # Test genai.configure
    try:
        print("Attempting genai.configure(api_key=api_key)...")
        genai.configure(api_key=api_key)
        print("genai.configure worked successfully.")
    except AttributeError:
        print("genai.configure failed with AttributeError (not found).")
        print("Attempting to set os.environ['GOOGLE_API_KEY']...")
        os.environ['GOOGLE_API_KEY'] = api_key
        print("os.environ['GOOGLE_API_KEY'] set.")
    except Exception as e:
        print("genai.configure failed with unexpected error:", e)
    print("-" * 40)

    # Test GenerativeModel
    print("Attempting to instantiate GenerativeModel...")
    model_found = False
    try:
        if hasattr(genai, 'GenerativeModel'):
            print("genai has 'GenerativeModel' attribute.")
            model = genai.GenerativeModel('gemini-pro')
            print("genai.GenerativeModel('gemini-pro') instantiated successfully.")
            model_found = True
        else:
            print("genai does NOT have 'GenerativeModel' attribute.")
            try:
                import google.ai.generativelanguage as glm
                if hasattr(glm, 'GenerativeModel'):
                    print("google.ai.generativelanguage has 'GenerativeModel' attribute.")
                    model = glm.GenerativeModel('gemini-pro')
                    print("glm.GenerativeModel('gemini-pro') instantiated successfully.")
                    model_found = True
                else:
                    print("Neither genai nor google.ai.generativelanguage has 'GenerativeModel'.")
            except ImportError:
                print("Could not import google.ai.generativelanguage.")
    except Exception as e:
        print("Error instantiating GenerativeModel:", e)
    
    if not model_found:
        print("\nCould not find a working way to instantiate GenerativeModel.")
        print("Please ensure you have the correct Google AI library installed and consult its documentation.")
    else:
        print("\nSuccessfully identified a way to use GenerativeModel.")

if __name__ == "__main__":
    test_genai_import()
