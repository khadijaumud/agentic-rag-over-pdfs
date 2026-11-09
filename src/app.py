"""FastAPI Application for Agentic RAG.
Endpoints:
- POST /chat (compatible with eval.py and UI)
- GET /chat/stream (SSE streaming)
- GET /documents (list indexed reports)
- POST /documents/upload (background ingestion)
- DELETE /documents/{doc_id} (cascade delete)
- POST /documents/reindex (rebuild vector index without reparsing)
- GET /conversations & GET /conversations/{id}
- GET /health (service health & stats)
- POST /feedback (user satisfaction rating)
"""

import uuid
import json
import asyncio
import logging
logger = logging.getLogger(__name__)
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from urllib.parse import quote
from pydantic import BaseModel

from src.config import PDF_DIR, DATA_DIR, MAX_USER_BUDGET_USD, MAX_HISTORY_TOKENS
from src.db import (
    init_db,
    list_documents as db_list_docs,
    delete_document as db_delete_doc,
    get_document,
    get_page_chunks,
    save_conversation,
    save_message,
    get_conversation_history,
    record_usage,
    get_user_total_cost,
    get_connection
)
from src.ingestion import ingest_pdf_file, rebuild_vector_index_from_db
from src.agent import run_agent
from src.pipeline import run_single_pass_pipeline
from src import vector_store

app = FastAPI(
    title="Agentic RAG over Annual Reports",
    description="Agentic multi-step tool-calling and single-pass RAG over financial PDFs",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure database schema is initialized on import
init_db()

@app.on_event("startup")
def startup_event():
    """Ensure database schema is initialized on start."""
    init_db()

class ChatRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    mode: Optional[str] = "agent"  # "agent" or "pipeline"
    document_filter: Optional[str] = None

@app.get("/health")
async def health():
    """Health check endpoint reporting indexed document count."""
    docs = db_list_docs()
    ready_count = sum(1 for d in docs if d["status"] == "ready")
    return {
        "status": "ok",
        "total_documents": len(docs),
        "ready_documents": ready_count,
        "vector_indexed_chunks": vector_store.count() if vector_store.is_available() else None,
    }

@app.post("/chat")
async def chat(req: ChatRequest, x_user_id: Optional[str] = Header(default="default_user", alias="X-User-Id")):
    """Core chat endpoint used by eval.py and frontend."""
    conv_id = req.conversation_id or f"conv_{uuid.uuid4().hex[:8]}"
    save_conversation(conv_id, title=req.question[:50], user_id=x_user_id)
    
    # Check user budget ceiling (e.g. $5.00 limit per user)
    spent = get_user_total_cost(x_user_id)
    if spent >= MAX_USER_BUDGET_USD:
        raise HTTPException(status_code=429, detail=f"User budget limit reached (${spent:.2f} spent).")
        
    history = get_conversation_history(conv_id, limit=6, max_tokens=MAX_HISTORY_TOKENS)
    
    # Infer mode: explicit mode in request body, or header/user-id indicating pipeline
    mode = req.mode or "agent"
    if "pipeline" in x_user_id.lower():
        mode = "pipeline"
        
    # Dispatch between multi-step agent and single-pass baseline
    try:
        if mode == "pipeline":
            result = run_single_pass_pipeline(
        req.question,
        document_filter=req.document_filter)
        else:
            result = run_agent(
        req.question,
        conversation_history=history,
        document_filter=req.document_filter
    )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
        
    usage = result.get("usage", {})
    cost = usage.get("usd", 0.0)
    p_tokens = usage.get("prompt_tokens", 0)
    c_tokens = usage.get("completion_tokens", 0)
    
    # Save user message and assistant reply to DB
    user_msg_id = f"msg_{uuid.uuid4().hex[:8]}"
    save_message(user_msg_id, conv_id, role="user", content=req.question)
    
    asst_msg_id = f"msg_{uuid.uuid4().hex[:8]}"
    save_message(
        asst_msg_id,
        conv_id,
        role="assistant",
        content=result["answer"],
        citations=result.get("citations", []),
        agent_steps=result.get("agent_steps"),
        prompt_tokens=p_tokens,
        completion_tokens=c_tokens,
        cost_usd=cost
    )
    
    # Record usage
    models = usage.get("models", [])
    model = usage.get("model") or (models[0] if models else req.mode or "agent")
    record_usage(x_user_id, conv_id, p_tokens, c_tokens, cost, model=model)
    
    return {
        "conversation_id": conv_id,
        "message_id": asst_msg_id,
        "answer": result["answer"],
        "citations": result.get("citations", []),
        "agent_steps": result.get("agent_steps", []),
        "usage": {
            "usd": cost,
            "prompt_tokens": p_tokens,
            "completion_tokens": c_tokens,
            "provider": usage.get("provider"),
            "model": usage.get("model"),
            "providers": usage.get("providers", []),
            "models": usage.get("models", []),
        }
    }

@app.get("/chat/stream")
async def chat_stream(
    question: str,
    conversation_id: Optional[str] = None,
    mode: str = "agent",
    document_filter: Optional[str] = None,
    x_user_id: Optional[str] = Header(
        default="default_user",
        alias="X-User-Id"
    )
):
    """Server-Sent Events (SSE) streaming chat endpoint."""
    async def event_generator():
        yield f"data: {json.dumps({'event': 'start', 'question': question})}\n\n"
        await asyncio.sleep(0.05)
        
        # Run the synchronous work behind the async endpoint in a worker thread.
        loop = asyncio.get_event_loop()
        req_obj = ChatRequest(
    question=question,
    conversation_id=conversation_id,
    mode=mode,
    document_filter=document_filter
)
        try:
            resp_data = await loop.run_in_executor(
                None, lambda: asyncio.run(chat(req_obj, x_user_id))
            )
        except HTTPException as exc:
            yield f"data: {json.dumps({'event': 'error', 'status': exc.status_code, 'detail': exc.detail})}\n\n"
            return
        
        except Exception:
            error_payload = {
                "event": "error",
                "status": 500,
                "detail": "Chat service failed. Check backend logs."
            }
            yield f"data: {json.dumps(error_payload)}\n\n"
            return
        # Stream agent steps
        for step in resp_data.get("agent_steps", []):
            yield f"data: {json.dumps({'event': 'step', 'step': step})}\n\n"
            await asyncio.sleep(0.02)
            
        # Stream tokens / words
        words = resp_data["answer"].split(" ")
        for w in words:
            yield f"data: {json.dumps({'event': 'token', 'token': w + ' '})}\n\n"
            await asyncio.sleep(0.01)
            
        # Stream final payload with citations and usage
        yield f"data: {json.dumps({'event': 'done', 'citations': resp_data['citations'], 'usage': resp_data['usage']})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/documents")
async def list_documents():
    """List all indexed documents with page counts and status."""
    return db_list_docs()

@app.get("/documents/{doc_id}/pages/{page_number}")
async def document_page(doc_id: str, page_number: int):
    """Return the extracted passage behind a citation."""
    if page_number < 1:
        raise HTTPException(status_code=400, detail="page_number must be 1 or greater.")
    if not get_document(doc_id):
        raise HTTPException(status_code=404, detail="Document not found.")
    chunks = get_page_chunks(doc_id, page_number)
    if not chunks:
        raise HTTPException(status_code=404, detail="Page content not found.")
    return {
        "document": doc_id,
        "page": page_number,
        "passages": [chunk["content"] for chunk in chunks],
    }

@app.get("/documents/{doc_id}/file")
async def document_file(doc_id: str):
    """Serve the original PDF so a citation can open the real report page."""
    document = get_document(doc_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found.")
    path = PDF_DIR / document["filename"]
    if not path.is_file():
        raise HTTPException(status_code=404, detail="PDF file is not available.")
    filename = str(document["filename"])
    return FileResponse(
        path,
        media_type="application/pdf",
        headers={"Content-Disposition": f"inline; filename*=UTF-8''{quote(filename)}"},
    )

@app.post("/documents/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload a new PDF document and process in the background."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    safe_filename = Path(file.filename).name
    if safe_filename != file.filename or safe_filename in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid PDF filename.")

    save_path = PDF_DIR / safe_filename
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)
        
    # Launch background ingestion
    background_tasks.add_task(ingest_pdf_file, save_path, force=True)
    
    return {
        "filename": safe_filename,
        "status": "pending",
        "message": f"Document {safe_filename} uploaded. Processing in background."
    }

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Cascade delete a document and its chunks from DB and FTS5."""
    doc = get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    if vector_store.is_available():
        try:
            vector_store.delete_document(doc_id)
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Vector index deletion failed.") from exc
    pdf_path = PDF_DIR / doc["filename"]
    if pdf_path.exists():
        try:
            pdf_path.unlink()
        except OSError as exc:
            raise HTTPException(status_code=503, detail="PDF file deletion failed.") from exc
    db_delete_doc(doc_id)
    return {"status": "deleted", "document": doc_id}

@app.post("/documents/reindex")
async def reindex_all(background_tasks: BackgroundTasks):
    """Rebuild vector index from SQLite chunks in background without reparsing PDFs."""
    background_tasks.add_task(rebuild_vector_index_from_db)
    return {"status": "started", "message": "Re-indexing initiated in background."}

@app.get("/conversations")
async def list_conversations(user_id: str = "default_user"):
    """List all saved conversations for a user."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM conversations WHERE user_id = ? ORDER BY updated_at DESC", (user_id,))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.get("/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    """Get full message history for a conversation."""
    history = get_conversation_history(conv_id, limit=50)
    return {"conversation_id": conv_id, "messages": history}

class FeedbackRequest(BaseModel):
    message_id: str
    score: int  # +1 or -1
    notes: Optional[str] = None

@app.post("/feedback")
async def submit_feedback(fb: FeedbackRequest):
    """Record user thumbs up/down feedback."""
    if fb.score not in (-1, 1):
        raise HTTPException(status_code=422, detail="score must be -1 or 1.")
    conn = get_connection()
    with conn:
        conn.execute(
            "INSERT INTO feedback (message_id, score, notes) VALUES (?, ?, ?)",
            (fb.message_id, fb.score, fb.notes)
        )
    conn.close()
    return {"status": "recorded"}
