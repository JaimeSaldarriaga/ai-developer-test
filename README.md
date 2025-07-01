# AI Developer Coding Test

This repository contains the submission for the AI Developer coding test. It's a CLI-based Python application that demonstrates practical AI integration skills by fetching news, narrating it with an LLM, and generating relevant images locally.

---

## Project Status

This project is being built in stages. Here is the current progress:

-   [x] **Step 1: Headline Fetching**: Connect to NewsAPI and retrieve headlines.
-   [x] **Step 2: LLM Narration**: Integrate Google Gemini to narrate each headline.
-   [X] **Step 3: Local Image Generation and MCP Server Creation**: Use a local Hugging Face model to create an image for each headline adding the MCP server for the News API logic.
-   [X] **Step 4: User Interruption**: Implement asynchronous logic to allow real-time user questions.

---

## Setup Instructions

1.  **Clone the Repository**:
    ```bash
    git clone <your-github-repo-link>
    cd <repository-folder>
    ```

2.  **Create and Activate Conda Environment**:
    This project uses Python 3.10+.
    ```bash
    conda create --name ai_test python=3.10
    conda activate ai_test
    ```

3.  **Install Dependencies**:
    ```bash
    pip install google-generativeai requests python-dotenv prompt_toolkit torch diffusers transformers accelerate
    ```

4.  **Set Up API Keys**:
    Create a file named `.env` in the root of the project directory. Add your API keys as follows:
    ```
    # .env
    GOOGLE_API_KEY="your-google-gemini-api-key"
    NEWS_API_KEY="your-newsapi-key"
    ```

## How to Run the App

With your Conda environment activated and the `.env` file in place, run the following command in your terminal:

```bash
python main.py

