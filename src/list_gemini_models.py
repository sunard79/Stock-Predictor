from google import genai
import os
from dotenv import load_dotenv

ENV_PATH = "config/.env"

def list_gemini_models():
    """Lists the Gemini models available to the user's API key."""
    load_dotenv(dotenv_path=ENV_PATH)
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key or api_key == "YOUR_GEMINI_API_KEY":
        print("Error: GEMINI_API_KEY not found or not set in config/.env.")
        print("Please add your key to the config/.env file.")
        return
    
    # Instantiate the client
    try:
        client = genai.Client(api_key=api_key)
        print("GenAI Client instantiated successfully.")
    except Exception as e:
        print(f"Error instantiating GenAI Client: {e}")
        print("Please ensure your API key is valid and the library is correctly installed.")
        return

    print() # Newline
    print("--- Available Gemini Models ---")
    try:
        # Assuming list_models is directly on the client object or in client.models
        # This is an educated guess based on typical client library patterns.
        # If this fails, we will need to explore `dir(client)` or official docs.
        for m in client.list_models(): 
            if 'generateContent' in m.supported_generation_methods:
                print(f"  {m.name}")
    except AttributeError:
        print("Client object does not have `list_models` attribute directly. Attempting client.models.list_models()...")
        try:
            for m in client.models.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    print(f"  {m.name}")
        except Exception as e:
            print(f"An error occurred while listing models: {e}")
            print("Could not find a way to list models from the client object. Please consult the official google-genai documentation for `list_models` usage.")
    except Exception as e:
        print(f"An error occurred while listing models: {e}")

if __name__ == "__main__":
    list_gemini_models()
