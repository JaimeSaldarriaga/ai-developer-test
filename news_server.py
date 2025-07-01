# news_server.py

import os
import httpx
from typing import Any, List
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

# Load environment variables for the server
load_dotenv()

# Initialize FastMCP server.
mcp = FastMCP("news_provider")

# Constants for the News API
NEWS_API_BASE = "https://newsapi.org/v2/top-headlines"
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
USER_AGENT = "MCP-News-Server/1.0"

async def make_news_request(url: str) -> dict[str, Any] | None:
    """Helper function to make a request to the News API."""
    if not NEWS_API_KEY:
        # Errors in the server should print to its own console/log
        print("SERVER ERROR: NEWS_API_KEY not found in .env file.")
        return None

    headers = {"User-Agent": USER_AGENT, "X-Api-Key": NEWS_API_KEY}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=10.0)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"SERVER ERROR: Could not fetch from News API: {e}")
            return None

@mcp.tool()
async def get_latest_headlines(country: str = "us", category: str = "technology") -> List[str]:
    """
    Get the latest news headlines for a given country and category.
    This is the tool our main application will call.
    """
    url = f"{NEWS_API_BASE}?country={country}&category={category}"
    data = await make_news_request(url)

    if not data or "articles" not in data or not data["articles"]:
        return ["No headlines found."]

   
    return [article['title'] for article in data["articles"][:5]]

if __name__ == "__main__":
 
    mcp.run(transport='stdio')