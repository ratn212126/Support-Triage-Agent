# Support Triage Agent

This is a terminal-based AI support triage agent built for the HackerRank Orchestrate hackathon. It uses a **Local Retrieval-Augmented Generation (RAG)** architecture to search through the provided offline support corpus and answer support tickets using the Gemini API.

## How it works

1. **Local Search (TF-IDF)**: When the script starts, it loads all markdown files from the `../data/` directory and indexes them using `scikit-learn`.
2. **Context Retrieval**: For each ticket, it filters documents by the specified `company`, uses cosine similarity to find the Top 5 most relevant articles, and combines them into a single context block.
3. **AI Generation**: It sends the ticket and the retrieved context block to the **Google Gemini API** (`gemini-2.0-flash`). The AI classifies the ticket and generates a grounded response strictly based on the context. If the issue is risky or ambiguous, it is safely escalated.

## Prerequisites

- Python 3.8+
- A Google Gemini API Key

## Setup & Installation

1. Navigate to the `code/` directory:
   ```bash
   cd code
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set your Gemini API key as an environment variable:
   - **Windows (Command Prompt):** `set GEMINI_API_KEY=your_api_key_here`
   - **Windows (PowerShell):** `$env:GEMINI_API_KEY="your_api_key_here"`
   - **macOS/Linux:** `export GEMINI_API_KEY=your_api_key_here`

## Usage

Run the agent. By default, it will read `../support_tickets/support_tickets.csv` and write its predictions to `../support_tickets/output.csv`:

```bash
python main.py
```

### Optional Arguments
You can customize the input and output paths if needed:
```bash
python main.py --input path/to/input.csv --output path/to/output.csv --api-key YOUR_API_KEY
```
