# Chitrapur GPT

A conversational AI assistant powered by LangChain and Chainlit that can answer questions about the Chitrapur Saraswat religious community using a knowledge base of books and magazines.

## Features

- Interactive chat interface powered by Chainlit
- Intelligent document retrieval using hybrid search (exact match, BM25, and fuzzy matching)
- Support for multiple LLM providers (OpenAI, Anthropic, Groq)
- Source citations for answers
- Automatic query expansion for better search results

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/chitrapurgpt.git
cd chitrapurgpt
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory with your API keys:
```
OPENAI_API_KEY = your-openai-api-key
ANTHROPIC_API_KEY = your-anthropic-api-key
GROQ_API_KEY = your-groq-api-key
LANGCHAIN_API_KEY = your-langchain-api-key
LANGCHAIN_TRACING = "true"
LANGCHAIN_PROJECT = your-langchain-project
```

## Running Locally

Start the application:
```bash
chainlit run app.py
```

The chat interface will be available at `http://localhost:8000`

## Project Structure

- `app.py` - Main Chainlit application entry point
- `steps.py` - Core logic for processing user queries and generating responses
- `retrievers.py` - Document retrieval implementations (Exact Match, BM25, Fuzzy Match)
- `ingest.py` - Knowledge base ingestion utilities
- `knowledge_base.jsonl` - Processed document store
- `documents/` - Raw document storage

## Deployment

This application can be easily deployed on Render:

1. Push your code to GitHub
2. Create a new Web Service on Render
3. Connect your GitHub repository
4. Configure the service:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `chainlit run app.py`
   - Add your API keys as environment variables