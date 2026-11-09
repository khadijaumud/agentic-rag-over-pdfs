import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from src.config import DB_PATH

def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    db_path = db_path or DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db(db_path: Path = DB_PATH) -> None:
    conn = get_connection(db_path)
    cur = conn.cursor()
    
    # 1. Documents table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        filename TEXT NOT NULL,
        page_count INTEGER NOT NULL,
        file_size_bytes INTEGER NOT NULL,
        status TEXT NOT NULL DEFAULT 'ready',
        error_message TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # 2. Chunks table (Relational source of truth)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id TEXT PRIMARY KEY,
        document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        page_number INTEGER NOT NULL,
        chunk_index INTEGER NOT NULL,
        content TEXT NOT NULL,
        token_count INTEGER NOT NULL DEFAULT 0,
        embedding BLOB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # Index for fast lookups
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_page ON chunks(document_id, page_number);")
    
    # 3. FTS5 table for BM25 lexical search
    cur.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
        id UNINDEXED,
        document_id UNINDEXED,
        page_number UNINDEXED,
        content,
        tokenize = 'porter unicode61'
    );
    """)
    
    # 4. Conversations table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        title TEXT,
        user_id TEXT DEFAULT 'default_user',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # 5. Messages table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        citations_json TEXT,
        agent_steps_json TEXT,
        prompt_tokens INTEGER DEFAULT 0,
        completion_tokens INTEGER DEFAULT 0,
        cost_usd REAL DEFAULT 0.0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);")

    # 6. Usage records table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS usage_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        conversation_id TEXT,
        prompt_tokens INTEGER DEFAULT 0,
        completion_tokens INTEGER DEFAULT 0,
        cost_usd REAL DEFAULT 0.0,
        model TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 7. Feedback table (Bonus +2 pts)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        message_id TEXT NOT NULL,
        score INTEGER NOT NULL,
        notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    conn.commit()
    conn.close()

# Helper DB functions

def save_document(doc_id: str, filename: str, page_count: int, file_size: int, status: str = "ready", error: Optional[str] = None) -> None:
    conn = get_connection()
    with conn:
        conn.execute("""
            INSERT INTO documents (id, filename, page_count, file_size_bytes, status, error_message)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                page_count=excluded.page_count,
                file_size_bytes=excluded.file_size_bytes,
                status=excluded.status,
                error_message=excluded.error_message
        """, (doc_id, filename, page_count, file_size, status, error))
    conn.close()

def save_chunks(chunks_data: List[Dict[str, Any]]) -> None:
    """Insert chunks into relational chunks table and chunks_fts table."""
    conn = get_connection()
    with conn:
        for c in chunks_data:
            emb_blob = None
            if "embedding" in c and c["embedding"] is not None:
                emb = np.array(c["embedding"], dtype=np.float32)
                emb_blob = emb.tobytes()
            
            conn.execute("""
                INSERT INTO chunks (id, document_id, page_number, chunk_index, content, token_count, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    content=excluded.content,
                    token_count=excluded.token_count,
                    embedding=coalesce(excluded.embedding, chunks.embedding)
            """, (c["id"], c["document_id"], c["page_number"], c["chunk_index"], c["content"], c.get("token_count", 0), emb_blob))

            # Sync with FTS5
            conn.execute("DELETE FROM chunks_fts WHERE id = ?", (c["id"],))
            conn.execute("""
                INSERT INTO chunks_fts (id, document_id, page_number, content)
                VALUES (?, ?, ?, ?)
            """, (c["id"], c["document_id"], c["page_number"], c["content"]))
    conn.close()

def delete_document(doc_id: str) -> None:
    """Cascade delete document and chunks and fts entries."""
    conn = get_connection()
    with conn:
        conn.execute("DELETE FROM chunks_fts WHERE document_id = ?", (doc_id,))
        conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.close()

def get_document(doc_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None

def list_documents() -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM documents ORDER BY filename ASC")
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_page_chunks(document_id: str, page_number: int) -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM chunks WHERE document_id = ? AND page_number = ? ORDER BY chunk_index ASC", (document_id, page_number))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_all_chunks(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    if limit:
        cur.execute("SELECT id, document_id, page_number, chunk_index, content, token_count, embedding FROM chunks LIMIT ?", (limit,))
    else:
        cur.execute("SELECT id, document_id, page_number, chunk_index, content, token_count, embedding FROM chunks")
    rows = cur.fetchall()
    res = []
    for r in rows:
        d = dict(r)
        if d["embedding"]:
            d["embedding"] = np.frombuffer(d["embedding"], dtype=np.float32)
        res.append(d)
    conn.close()
    return res

def save_conversation(conv_id: str, title: str, user_id: str = "default_user") -> None:
    conn = get_connection()
    with conn:
        conn.execute("""
            INSERT INTO conversations (id, title, user_id)
            VALUES (?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title=COALESCE(NULLIF(conversations.title, ''), excluded.title),
                updated_at=CURRENT_TIMESTAMP
        """, (conv_id, title, user_id))
    conn.close()

def save_message(msg_id: str, conv_id: str, role: str, content: str, citations: Optional[List[Dict[str, Any]]] = None, agent_steps: Optional[List[Dict[str, Any]]] = None, prompt_tokens: int = 0, completion_tokens: int = 0, cost_usd: float = 0.0) -> None:
    conn = get_connection()
    with conn:
        conn.execute("""
            INSERT INTO messages (id, conversation_id, role, content, citations_json, agent_steps_json, prompt_tokens, completion_tokens, cost_usd)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            msg_id,
            conv_id,
            role,
            content,
            json.dumps(citations) if citations else None,
            json.dumps(agent_steps) if agent_steps else None,
            prompt_tokens,
            completion_tokens,
            cost_usd
        ))
    conn.close()

def get_conversation_history(
    conv_id: str,
    limit: int = 10,
    max_tokens: Optional[int] = None,
) -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, role, content, citations_json, agent_steps_json,
               prompt_tokens, completion_tokens, cost_usd
        FROM messages
        WHERE conversation_id = ?
        ORDER BY created_at ASC, rowid ASC
    """, (conv_id,))
    rows = cur.fetchall()
    conn.close()
    res = []
    for r in rows:
        d = dict(r)
        if d.get("citations_json"):
            d["citations"] = json.loads(d["citations_json"])
        if d.get("agent_steps_json"):
            d["agent_steps"] = json.loads(d["agent_steps_json"])
        if d["role"] == "assistant":
            d["usage"] = {
                "prompt_tokens": d.pop("prompt_tokens", 0) or 0,
                "completion_tokens": d.pop("completion_tokens", 0) or 0,
                "usd": d.pop("cost_usd", 0.0) or 0.0,
            }
        else:
            d.pop("prompt_tokens", None)
            d.pop("completion_tokens", None)
            d.pop("cost_usd", None)
        res.append(d)
    recent = res[-limit:]
    if max_tokens is None:
        return recent
    selected = []
    token_total = 0
    for message in reversed(recent):
        message_tokens = max(1, len(message["content"]) // 4)
        if selected and token_total + message_tokens > max_tokens:
            break
        selected.append(message)
        token_total += message_tokens
    return list(reversed(selected))

def record_usage(user_id: str, conv_id: str, prompt_tokens: int, completion_tokens: int, cost_usd: float, model: str) -> None:
    conn = get_connection()
    with conn:
        conn.execute("""
            INSERT INTO usage_records (user_id, conversation_id, prompt_tokens, completion_tokens, cost_usd, model)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (user_id, conv_id, prompt_tokens, completion_tokens, cost_usd, model))
    conn.close()

def get_user_total_cost(user_id: str) -> float:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT SUM(cost_usd) FROM usage_records WHERE user_id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()
    return float(row[0] or 0.0)
