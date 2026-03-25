# Automerge Server

Python API server with RAG for AI-powered merge conflict resolution using Ollama and Qwen 3.5.

## Prerequisites

1. **Ollama** installed and running
2. **Qwen 3.5** model pulled: `ollama pull qwen3.5`
3. **Python 3.10+**

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment config
cp .env.example .env

# Start Ollama (if not running)
ollama serve

# Start the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

- `GET /predictor/health/` - Health check
- `POST /predictor/resolve/` - Resolve single conflict
- `POST /predictor/resolve/batch/` - Resolve multiple conflicts

## RAG Indexing

Build the vector index from git history and codebase:

```bash
python scripts/build_index.py
```

## VS Code Extension

The server powers the `vscodeext/` extension. Set `MOCK_MODE = false` in `apiService.ts` to use the real API.
