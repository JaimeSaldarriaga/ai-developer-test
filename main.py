import os
import sys # Import the sys module
import asyncio
import json
from typing import List

# --- AI and API Libraries ---
import torch
import google.generativeai as genai
from diffusers import DiffusionPipeline
from dotenv import load_dotenv
import requests # Import requests for the fallback


from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

# --- CLI Interaction Library ---
from prompt_toolkit import PromptSession



def setup_apis():
    """Loads environment variables and configures APIs."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file.")
    genai.configure(api_key=api_key)
    print("APIs configured.")

def initialize_image_pipeline():
    """Initializes the local text-to-image pipeline."""
    print("Initializing local image generation pipeline...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")
    pipeline = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)
    print("Image generation pipeline ready.")
    return pipeline



def _fetch_headlines_directly() -> List[str]:
    """
    Fallback function to get headlines directly from NewsAPI.
    This is used if the MCP server communication fails.
    """
    print("--- MCP connection failed. Using direct API fallback. ---")
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        return []
    url = f"https://newsapi.org/v2/top-headlines?country=us&category=technology&apiKey={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return [a['title'] for a in data.get('articles', []) if a.get('title')][:5]
    except Exception as e:
        print(f"Direct API fallback failed: {e}")
        return []

async def fetch_headlines_from_mcp() -> List[str] | None:
    """
    Connects to the news_server.py subprocess using the correct Python
    executable to ensure a stable environment.
    """
    print("Attempting to fetch headlines from MCP server using official client...")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["news_server.py"],
        env=os.environ
    )
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tool_result = await session.call_tool(
                    "get_latest_headlines",
                    arguments={"country": "us", "category": "technology"}
                )

                if isinstance(tool_result, types.CallToolResult) and tool_result.structured:
                    final_result = tool_result.structured.get("result")
                    if isinstance(final_result, list):
                        print("--- MCP connection successful! ---")
                        return final_result

                print(f"MCP client returned unexpected data structure: {tool_result}")

    except Exception as e:
        print(f"Failed to execute MCP with official client: {e}")
    
    return None

async def simulate_narration(text: str):
    """Prints text word-by-word to simulate a narrator."""
    for word in text.split():
        print(word, end=' ', flush=True)
        await asyncio.sleep(0.2)
    print()

async def narrate_and_handle_questions(headline: str, llm: genai.GenerativeModel):
    """
    Uses a robust asyncio.wait pattern to handle narration and user
    interruption concurrently.
    """
    print("\n" + "="*80)
    print(f"H E A D L I N E: {headline}")
    print("="*80)
    print("(You can start typing a question at any time and press ENTER to interrupt)")

    narration_prompt = f"Narrate this headline in one or two engaging sentences: '{headline}'"
    narration_response = await llm.generate_content_async(narration_prompt)
    full_narration = narration_response.text.strip()

    session = PromptSession('> ')
    
    narration_task = asyncio.create_task(simulate_narration(full_narration))
    input_task = asyncio.create_task(session.prompt_async())

    done, pending = await asyncio.wait(
        [narration_task, input_task],
        return_when=asyncio.FIRST_COMPLETED
    )

    if input_task in done:
        user_input = input_task.result()
        if user_input:
            print("\n--- Pausing narration ---")
            question_prompt = f"In the context of the headline '{headline}', answer this question: '{user_input}'"
            answer_response = await llm.generate_content_async(question_prompt)
            print(f"ANSWER: {answer_response.text.strip()}")
            print("--- Resuming ---")

    for task in pending:
        task.cancel()

    await asyncio.sleep(1)

def generate_image_locally(prompt: str, pipeline: DiffusionPipeline, index: int):
    """Generates and saves an image for a headline."""
    print(f"\nGenerating local image for: '{prompt[:50]}...'")
    try:
        safe_filename = "".join([c for c in prompt if c.isalnum() or c.isspace()]).strip()
        safe_filename = f"headline_{index+1}_{safe_filename.replace(' ', '_').lower()[:50]}.png"
        image = pipeline(prompt, num_inference_steps=20).images[0]
        image.save(safe_filename)
        print(f"Image saved to: {safe_filename}")
    except Exception as e:
        print(f"Error generating image: {e}")



async def main():
    """The main entry point for the CLI application."""
    try:
        setup_apis()
        image_pipe = initialize_image_pipeline()
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        
        headlines = await fetch_headlines_from_mcp()
        if headlines is None:
            headlines = _fetch_headlines_directly()
        
        if not headlines:
            print("Could not retrieve headlines. Exiting.")
            return

        print("\n*** Welcome to the AI News Narrator ***")
        for i, headline in enumerate(headlines):
            await narrate_and_handle_questions(headline, gemini_model)
            generate_image_locally(headline, image_pipe, i)
            
        print("\n" + "="*80 + "\nAll headlines processed. Application finished.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
