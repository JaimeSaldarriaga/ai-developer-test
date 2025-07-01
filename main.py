# main.py 

import os
import asyncio
import subprocess
import json
from typing import List

# --- AI and API Libraries ---
import torch
import google.generativeai as genai
from diffusers import DiffusionPipeline
from dotenv import load_dotenv

# --- CLI Interaction Library ---
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout


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
    # Determine device for PyTorch (GPU or CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")
    pipeline = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)
    print("Image generation pipeline ready.")
    return pipeline



async def fetch_headlines_from_mcp() -> List[str]:
    """
    Launches news_server.py as a subprocess and calls its tool
    to get headlines using a manual JSON-RPC protocol over stdio.
    """
    print("Launching MCP news server subprocess...")
    command = ["python", "news_server.py"]
    
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE  # Capture stderr to see server errors
    )

    mcp_request = {
        "jsonrpc": "2.0",
        "method": "news_provider/get_latest_headlines",
        "params": {"country": "us", "category": "technology"},
        "id": 1,
    }

    request_json = json.dumps(mcp_request) + '\n'
    process.stdin.write(request_json.encode('utf-8'))
    await process.stdin.drain()

    # Read response from server
    response_json = await process.stdout.readline()
    
    # Read any errors from the server
    server_error = await process.stderr.read()
    if server_error:
        print(f"MCP SERVER LOG: {server_error.decode()}")
    
    await process.terminate()

    if not response_json:
        print("CLIENT ERROR: No response from MCP server.")
        return []

    try:
        response_data = json.loads(response_json)
        if "result" in response_data:
            return response_data["result"]
        elif "error" in response_data:
            print(f"MCP SERVER ERROR: {response_data['error']}")
            return []
    except json.JSONDecodeError:
        print("CLIENT ERROR: Could not decode server response.")
    return []

async def simulate_narration(text: str):
    """Prints text word-by-word to simulate a narrator."""
    for word in text.split():
        print(word, end=' ', flush=True)
        await asyncio.sleep(0.2)
    print()

async def narrate_and_handle_questions(headline: str, llm: genai.GenerativeModel):
    """
    Narrates a headline, allows interruption for questions, and resumes.
    """
    print("\n" + "="*80)
    print(f"H E A D L I N E: {headline}")
    print("="*80)
    print("(Press ENTER at any time to ask a question about this headline)")

    narration_prompt = f"Narrate this headline in one or two engaging sentences: '{headline}'"
    narration_response = await llm.generate_content_async(narration_prompt)
    full_narration = narration_response.text.strip()

    session = PromptSession()
    narration_task = asyncio.create_task(simulate_narration(full_narration))

    while not narration_task.done():
        try:
            user_input = await session.prompt_async('> ', refresh_interval=0.5)
            if user_input:
                narration_task.cancel()
                print("\n--- Pausing narration ---")
                question_prompt = f"In the context of the headline '{headline}', answer this question: '{user_input}'"
                answer_response = await llm.generate_content_async(question_prompt)
                print(f"ANSWER: {answer_response.text.strip()}")
                print("--- Resuming ---")
                break
        except (asyncio.CancelledError, EOFError):
            break
    
    if not narration_task.done():
        narration_task.cancel()
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
        gemini_model = genai.GenerativeModel('gemini-1.0-pro')
        
        headlines = await fetch_headlines_from_mcp()
        
        if not headlines or "No headlines found." in headlines[0]:
            print("Could not retrieve headlines via MCP. Exiting.")
            return

        print("\n*** Welcome to the AI News Narrator ***")
        for i, headline in enumerate(headlines):
            with patch_stdout():
                await narrate_and_handle_questions(headline, gemini_model)
            generate_image_locally(headline, image_pipe, i)
            
        print("\n" + "="*80 + "\nAll headlines processed. Application finished.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())



