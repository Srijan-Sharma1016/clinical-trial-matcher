import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load your API key securely from the .env file we made earlier
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Error: Could not find GEMINI_API_KEY in your .env file.")
    exit()

genai.configure(api_key=api_key)

print("🔍 Querying Google AI Studio for supported models...\n")

# Loop through all models available to your specific API key
for m in genai.list_models():
    # We only care about models that support 'generateContent' (text generation)
    if 'generateContent' in m.supported_generation_methods:
        print(f"✅ Model String: {m.name}")
        print(f"   Display Name: {m.display_name}")
        print("-" * 50)