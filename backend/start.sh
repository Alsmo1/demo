#!/bin/bash
set -e

# Start RAG service from the scr directory
cd ../scr
uvicorn co:app --host 127.0.0.1 --port 8001 &
RAG_PID=$!

MAX_WAIT=300
ELAPSED=0
until curl -sf http://127.0.0.1:8001/health > /dev/null 2>&1; do
    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "RAG service failed to start"
        kill $RAG_PID
        exit 1
    fi
    sleep 3
    ELAPSED=$((ELAPSED + 3))
done

# Start the main FastAPI app from the backend directory
cd ../backend
uvicorn main:app --host 0.0.0.0 --port 7860