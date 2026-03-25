# Automerge Server

Python API server with RAG for AI-powered merge conflict resolution using Ollama and Qwen 3.x (default: qwen3-vl:235b-cloud).

## Prerequisites

1. **Ollama** installed and running
2. **Qwen 3.x** model pulled: `ollama pull qwen3-vl:235b-cloud`
3. **Python 3.10+**

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment config
cp .env.example .env

# Start Ollama (if not running)
ollama serve

# Start the server (default port 8080)
uvicorn main:app --reload --host 0.0.0.0 --port 8080
## Configuration

Default settings (see `config.py`):

- **Ollama Base URL:** `http://localhost:11434`
- **Ollama Model:** `qwen3-vl:235b-cloud`
- **ChromaDB Directory:** `./chroma_db`
- **Embedding Model:** `all-MiniLM-L6-v2`
- **Host:** `0.0.0.0`
- **Port:** `8080`
- **Git History Depth:** `500`

You can override these by editing `.env`.
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
