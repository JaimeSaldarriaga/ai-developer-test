# main.py - Step 2

import os
import requests
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List

# --- SETUP FUNCTIONS ---
def setup_apis():
    """Load .env and configure APIs."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found.")
    genai.configure(api_key=api_key)
    print("APIs configured.")

# --- CORE FUNCTIONS ---
def fetch_headlines() -> List[str]:
    # (This function remains the same as Step 1)
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        print("Error: NEWS_API_KEY not found.")
        return []
    url = f"https://newsapi.org/v2/top-headlines?country=us&category=technology&apiKey={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        headlines = [article['title'] for article in data.get('articles', []) if article.get('title')]
        return headlines[:5]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news: {e}")
        return []

def generate_narration(headline: str, llm: genai.GenerativeModel) -> str:
    """Generates a brief narration for a headline."""
    prompt = f"Narrate this headline in one engaging sentence: '{headline}'"
    try:
        response = llm.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Could not generate narration: {e}"

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    setup_apis()
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')

    print("Fetching latest headlines...")
    headlines = fetch_headlines()

    if headlines:
        print("\n--- Narrating Headlines ---")
        for headline in headlines:
            print(f"\nHEADLINE: {headline}")
            narration = generate_narration(headline, gemini_model)
            print(f"NARRATION: {narration}")
    else:
        print("Could not retrieve headlines.")