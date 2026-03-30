#!/bin/zsh

echo "=== Health Check ==="
curl -s http://localhost:8080/predictor/health/ | python3 -m json.tool

# echo -e "\n=== Resolve with HuggingFace ==="
# curl -s -X POST http://localhost:8080/predictor/resolve/ \
#   -H "Content-Type: application/json" \
#   -d '{
#     "conflict_text": "<<<<<<< HEAD\nconst x = 1;\n=======\nconst x = 2;\n>>>>>>> feat",
#     "language": "javascript",
#     "provider": "huggingface"
#   }' | python3 -m json.tool

# echo -e "\n=== Resolve with Ollama ==="
# curl -s -X POST http://localhost:8080/predictor/resolve/ \
#   -H "Content-Type: application/json" \
#   -d '{
#     "conflict_text": "<<<<<<< HEAD\nconst x = 1;\n=======\nconst x = 2;\n>>>>>>> feat",
#     "language": "javascript",
#     "provider": "ollama"
#   }' | python3 -m json.tool

echo -e "\n=== Batch Resolve ==="
curl -s -X POST http://localhost:8080/predictor/resolve/batch/ \
  -H "Content-Type: application/json" \
  -d '{
    "conflicts": [
      {
        "conflict_text": "<<<<<<< HEAD\nconst x = 1;\n=======\nconst x = 2;\n>>>>>>> feat",
        "language": "javascript",
        "provider": "huggingface"
      },
      {
        "conflict_text": "<<<<<<< HEAD\ndef add(x, y):\n    return x + y\n=======\ndef add(a, b):\n    return a * b\n>>>>>>> feature",
        "language": "python",
        "provider": "huggingface"
      }
    ]
  }' | python3 -m json.tool
