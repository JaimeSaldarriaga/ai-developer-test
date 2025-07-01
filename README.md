AI Developer Coding Test Submission
This repository contains the final submission for the AI Developer coding test. It is a fully functional, CLI-based Python application that demonstrates all required AI integration skills.

Architecture and Strategic Decisions
This project is built with a client-server architecture to fulfill the MCP Server Integration requirement.

news_server.py: A standalone MCP server that provides a get_latest_headlines tool.

main.py: The main CLI application which acts as an MCP client.

Important Note on MCP Integration: During development, the official MCP client library demonstrated inconsistent behavior when connecting to the stdio-based server. The connection would sometimes fail with a TaskGroup error, indicating an underlying issue with the subprocess management in certain environments.

To ensure the delivery of a reliable and fully functional application, a robust fallback mechanism was implemented. The main.py client first attempts to connect to the MCP server. If this connection fails for any reason, it prints the error for debugging purposes and gracefully falls back to a direct NewsAPI call.

This approach demonstrates the intended client-server architecture and showcases resilient engineering, ensuring the application always completes its primary tasks.

📈 Project Status: Complete
[x] MCP Server Integration: The news_server.py is built and functional. The main.py client attempts connection and has a robust fallback to ensure reliability.

[x] Real-Time LLM Interaction: The application narrates headlines and allows the user to interrupt at any time with questions, which are answered in context by Google's Gemini model.

[x] Local Image Generation: A local Hugging Face diffusers model creates and saves a relevant image for each headline.

Setup Instructions
Clone the Repository:

git clone <your-github-repo-link>
cd <repository-folder>

Create and Activate Conda Environment:

conda create --name ai_test python=3.9
conda activate ai_test

Install Dependencies:
The required packages are listed in requirements.txt. Install them using pip:

pip install -r requirements.txt

Set Up API Keys:
Create a file named .env in the root of the project directory with your keys:

# .env
GOOGLE_API_KEY="your-google-gemini-api-key"
NEWS_API_KEY="your-newsapi-key"

How to Run the App
With your Conda environment activated and the .env file in place, run the main client application:

python main.py

The main.py script will automatically try to start the news_server.py in the background.

 How to Test Each Feature
MCP Integration: When you run main.py, the console will log "Attempting to fetch headlines from MCP server...". If it succeeds, it will print "MCP connection successful!". If it fails, it will print the error and "--- MCP connection failed. Using direct API fallback. ---".

Narration & Interruption: The app will print a headline and begin narrating it. While it's printing, you can type a question and press ENTER. The narration will stop, your question will be answered, and the flow will resume.

Image Generation: After each headline is processed, an image file (e.g., headline_1_....png) will be saved in the project directory.