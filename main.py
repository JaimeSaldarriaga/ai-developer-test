import os
import requests
from dotenv import load_dotenv
from typing import List

def fetch_headlines() -> List[str]:
    """Fetches top tech headlines from the NewsAPI."""
    # Load environment variables from .env file
    load_dotenv()
    api_key = os.getenv("NEWS_API_KEY")

    if not api_key:
        print("Error: NEWS_API_KEY not found in your .env file.")
        return []

    url = f"https://newsapi.org/v2/top-headlines?country=us&category=technology&apiKey={api_key}"

    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for bad status codes
        data = response.json()
        headlines = [article['title'] for article in data.get('articles', []) if article.get('title')]
        return headlines[:5] # Get top 5 headlines
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news: {e}")
        return []

if __name__ == "__main__":
    print("Fetching latest headlines...")
    headlines = fetch_headlines()

    if headlines:
        print("\n--- Headlines Fetched Successfully ---")
        for i, headline in enumerate(headlines, 1):
            print(f"{i}: {headline}")
    else:
        print("Could not retrieve headlines.")