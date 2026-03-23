from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import json
import sys
import os

# Add current directory to path to find rag.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag import rag_answer_stream

app = FastAPI()

@app.get("/health")
async def health():
    """Health check endpoint required by start.sh startup script."""
    return {"status": "ok", "service": "rag"}

class Question(BaseModel):
    question: str
    history: List[Dict] = []
    conversation_id: Optional[int] = None

@app.post("/ask_stream")
async def ask_stream(data: Question):
    async def event_generator():
        content_started = False
        try:
            async for chunk in rag_answer_stream(data.question, data.history):
                # 3. Filter [STATUS] markers -> send as status event
                if chunk.startswith("[STATUS]"):
                    status_msg = chunk.replace("[STATUS]", "").strip()
                    yield f"data: {json.dumps({'status': status_msg})}\n\n"
                    continue
                
                # 4. Filter [CONTENT] marker -> start sending tokens
                if chunk.startswith("[CONTENT]"):
                    content_started = True
                    # Handle rare case where content follows immediately
                    remainder = chunk.replace("[CONTENT]", "")
                    if remainder:
                        yield f"data: {json.dumps({'token': remainder})}\n\n"
                    continue
                
                # 2. Wrap token in proper SSE format
                if content_started or not chunk.startswith("["):
                    yield f"data: {json.dumps({'token': chunk})}\n\n"
            
            # 5. Send done signal
            yield f"data: {json.dumps({'status': 'done'})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    # 1. Return media_type="text/event-stream"
    return StreamingResponse(event_generator(), media_type="text/event-stream")