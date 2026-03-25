# Automerge Server API - cURL Examples

API documentation and example cURL commands for the Automerge Server.

## Base URL

```
http://localhost:8080
```

## Configuration

Default settings (from `config.py`):
- **Host:** `0.0.0.0`
- **Port:** `8080`
- **Model:** `qwen2.5-coder:32b`

---

## Endpoints

### 1. Root Endpoint

Get API information and available endpoints.

```bash
curl -X GET http://localhost:8080/
```

**Response:**
```json
{
  "name": "Automerge Server",
  "version": "0.1.0",
  "description": "AI-powered merge conflict resolution with RAG",
  "endpoints": {
    "health": "/predictor/health/",
    "resolve": "/predictor/resolve/",
    "batch_resolve": "/predictor/resolve/batch/"
  }
}
```

---

### 2. Health Check

Check if the service is healthy and Ollama is available.

```bash
curl -X GET http://localhost:8080/predictor/health/
```

**Response:**
```json
{
  "status": "healthy",
  "ollama_available": true,
  "model_loaded": "qwen2.5-coder:32b"
}
```

---

### 3. Resolve Single Conflict

Resolve a single merge conflict.

**Endpoint:** `POST /predictor/resolve/`

**Request Body:**
```json
{
  "conflict_text": "<<<<<<< HEAD\nconsole.log('hello');\n=======\nconsole.log('world');\n>>>>>>> feature-branch",
  "language": "typescript",
  "file_path": "src/index.ts"
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8080/predictor/resolve/ \
  -H "Content-Type: application/json" \
  -d '{
    "conflict_text": "<<<<<<< HEAD\nconsole.log('\''hello'\'');\n=======\nconsole.log('\''world'\'');\n>>>>>>> feature-branch",
    "language": "typescript",
    "file_path": "src/index.ts"
  }'
```

**Using a file:**
```bash
curl -X POST http://localhost:8080/predictor/resolve/ \
  -H "Content-Type: application/json" \
  -d @conflict.json
```

**Response:**
```json
{
  "result": "console.log('hello');",
  "summary": "Merged changes - kept HEAD version as it represents the stable main branch",
  "confidence": 0.85
}
```

---

### 4. Resolve Batch Conflicts

Resolve multiple merge conflicts in a single request.

**Endpoint:** `POST /predictor/resolve/batch/`

**Request Body:**
```json
{
  "conflicts": [
    {
      "conflict_text": "<<<<<<< HEAD\nconst x = 1;\n=======\nconst x = 2;\n>>>>>>> feature",
      "language": "javascript",
      "file_path": "app.js"
    },
    {
      "conflict_text": "<<<<<<< HEAD\ndef foo():\n    pass\n=======\ndef foo():\n    return True\n>>>>>>> feature",
      "language": "python",
      "file_path": "utils.py"
    }
  ]
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8080/predictor/resolve/batch/ \
  -H "Content-Type: application/json" \
  -d '{
    "conflicts": [
      {
        "conflict_text": "<<<<<<< HEAD\nconst x = 1;\n=======\nconst x = 2;\n>>>>>>> feature",
        "language": "javascript",
        "file_path": "app.js"
      },
      {
        "conflict_text": "<<<<<<< HEAD\ndef foo():\n    pass\n=======\ndef foo():\n    return True\n>>>>>>> feature",
        "language": "python",
        "file_path": "utils.py"
      }
    ]
  }'
```

**Response:**
```json
{
  "results": [
    {
      "result": "const x = 1;",
      "summary": "Kept HEAD version",
      "confidence": 0.80
    },
    {
      "result": "def foo():\n    return True",
      "summary": "Merged - feature branch adds useful return value",
      "confidence": 0.92
    }
  ]
}
```

---

## Complete Examples

### Example 1: Python Conflict

```bash
curl -X POST http://localhost:8080/predictor/resolve/ \
  -H "Content-Type: application/json" \
  -d '{
    "conflict_text": "<<<<<<< HEAD\ndef calculate(a, b):\n    return a + b\n=======\ndef calculate(a, b):\n    return a * b\n>>>>>>> feature",
    "language": "python",
    "file_path": "calculator.py"
  }'
```

### Example 2: JavaScript/TypeScript Conflict

```bash
curl -X POST http://localhost:8080/predictor/resolve/ \
  -H "Content-Type: application/json" \
  -d '{
    "conflict_text": "<<<<<<< HEAD\nasync function fetchData() {\n  return await api.get('\''/users'\'');\n}\n=======\nasync function fetchData() {\n  return await api.get('\''/posts'\'');\n}\n>>>>>>> feature",
    "language": "typescript",
    "file_path": "api.ts"
  }'
```

### Example 3: Go Conflict

```bash
curl -X POST http://localhost:8080/predictor/resolve/ \
  -H "Content-Type: application/json" \
  -d '{
    "conflict_text": "<<<<<<< HEAD\nfunc (s *Server) Start() error {\n    return s.listen()\n}\n=======\nfunc (s *Server) Start() error {\n    return s.listenTLS()\n}\n>>>>>>> feature",
    "language": "go",
    "file_path": "server.go"
  }'
```

---

## Error Handling

When resolution fails, the API returns the original conflict text with an error summary:

```json
{
  "result": "<<<<<<< HEAD\n...\n=======\n...\n>>>>>>> feature",
  "summary": "Resolution failed: <error message>",
  "confidence": 0.0
}
```

---

## Testing with jq

For better output formatting, pipe through `jq`:

```bash
curl -s -X POST http://localhost:8080/predictor/resolve/ \
  -H "Content-Type: application/json" \
  -d '{"conflict_text": "...", "language": "python"}' | jq .
```

Check health status:
```bash
curl -s http://localhost:8080/predictor/health/ | jq .
```

---

## Starting the Server

```bash
# From the project directory
python main.py

# Or with uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8080
```

---

## Notes

- The server uses **FastAPI** with automatic OpenAPI documentation available at `http://localhost:8080/docs`
- CORS is enabled for all origins (suitable for local development with VS Code extension)
- The RAG service uses ChromaDB for vector storage and Ollama for LLM inference
- Default model: `qwen2.5-coder:32b`
