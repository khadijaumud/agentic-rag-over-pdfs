"""Ingestion Service for PDFs:
- Background ingestion with status tracking (pending / processing / ready / failed)
- Layout-aware parsing and page-preserving chunking
- Dense embedding generation and SQLite relational + FTS5 indexing
- Rebuilding vector index directly from SQLite database without reparsing PDFs
"""

import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.parser import parse_pdf
from src.chunker import chunk_document_pages
from src.embeddings import embed_texts
from src.db import save_document, save_chunks, get_connection, get_all_chunks
from src.config import PDF_DIR

def ingest_pdf_file(pdf_path: Path, force: bool = False) -> Dict[str, Any]:
    """Ingest a single PDF file into SQLite and FTS5."""
    doc_id = pdf_path.name
    file_size = pdf_path.stat().st_size
    
    # Check if already processed
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT status FROM documents WHERE id = ?", (doc_id,))
    row = cur.fetchone()
    conn.close()
    
    if row and row["status"] == "ready" and not force:
        return {"document": doc_id, "status": "already_indexed"}
        
    save_document(doc_id=doc_id, filename=doc_id, page_count=0, file_size=file_size, status="processing")
    
    try:
        t0 = time.time()
        pages = parse_pdf(pdf_path, preferred_parser="auto")
        page_count = len(pages)
        
        chunks = chunk_document_pages(document_id=doc_id, pages=pages, target_chunk_size=600, chunk_overlap=100)
        
        # Compute embeddings for chunks
        chunk_texts = [c["content"] for c in chunks]
        embeddings = embed_texts(chunk_texts, batch_size=32)
        
        for c, emb in zip(chunks, embeddings):
            c["embedding"] = emb
            
        # Save to SQLite
        save_chunks(chunks)
        save_document(doc_id=doc_id, filename=doc_id, page_count=page_count, file_size=file_size, status="ready")
        
        duration = round(time.time() - t0, 2)
        return {
            "document": doc_id,
            "page_count": page_count,
            "chunk_count": len(chunks),
            "status": "ready",
            "seconds": duration
        }
    except Exception as e:
        save_document(doc_id=doc_id, filename=doc_id, page_count=0, file_size=file_size, status="failed", error=str(e))
        return {"document": doc_id, "status": "failed", "error": str(e)}

def ingest_all_starter_pdfs(pdf_dir: Path = PDF_DIR, force: bool = False) -> List[Dict[str, Any]]:
    """Batch ingest all 14 corporate annual reports."""
    pdf_files = sorted(list(pdf_dir.glob("*.pdf")))
    results = []
    print(f"[Ingestion] Found {len(pdf_files)} PDFs in {pdf_dir}...")
    for pdf in pdf_files:
        res = ingest_pdf_file(pdf, force=force)
        results.append(res)
        print(f"  Processed {pdf.name}: {res.get('status')} ({res.get('page_count', 0)} pages, {res.get('chunk_count', 0)} chunks)")
    return results

def rebuild_vector_index_from_db(batch_size: int = 32) -> Dict[str, Any]:
    """Rebuild vector embeddings strictly from the SQLite database without parsing PDFs again."""
    t0 = time.time()
    chunks = get_all_chunks()
    if not chunks:
        return {"status": "empty", "reindexed_count": 0}
        
    texts = [c["content"] for c in chunks]
    embeddings = embed_texts(texts, batch_size=batch_size)
    
    conn = get_connection()
    with conn:
        for c, emb in zip(chunks, embeddings):
            conn.execute(
                "UPDATE chunks SET embedding = ? WHERE id = ?",
                (emb.tobytes(), c["id"])
            )
    conn.close()
    
    duration = round(time.time() - t0, 2)
    return {
        "status": "success",
        "reindexed_count": len(chunks),
        "seconds": duration
    }
