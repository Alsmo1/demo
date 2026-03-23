#!/bin/bash
set -e

# Start the RAG service in the background from its directory
cd /code/scr
echo "Starting RAG Service on port 8001..."
uvicorn co:app --host 127.0.0.1 --port 8001 &
RAG_PID=$!

# Wait for RAG service to be healthy using Curl
MAX_WAIT=300
ELAPSED=0
until curl -sf http://127.0.0.1:8001/health > /dev/null 2>&1; do
    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "❌ RAG service failed to start"
        kill $RAG_PID
        exit 1
    fi
    sleep 3
    ELAPSED=$((ELAPSED + 3))
done

echo "✅ RAG Service is up!"

# Start the main FastAPI app in the foreground mapped to Hugging Face's exposed port
cd /code/backend
echo "Starting Main Service on port 7860..."
uvicorn main:app --host 0.0.0.0 --port 7860