"""Initial ingestion script for all 14 corporate annual reports.
Parses PDFs, generates chunks with 1-based page metadata, computes embeddings,
and stores in SQLite database with FTS5 lexical index.
"""

import time
from src.db import init_db, list_documents, get_connection
from src.ingestion import ingest_all_starter_pdfs
from src.config import PDF_DIR

def main():
    print("=" * 60)
    print("Agentic RAG: Corporate Annual Reports Ingestion")
    print("=" * 60)
    
    init_db()
    t0 = time.time()
    
    results = ingest_all_starter_pdfs(PDF_DIR, force=False)
    
    duration = time.time() - t0
    docs = list_documents()
    
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM chunks")
    chunk_count = cur.fetchone()[0]
    conn.close()
    
    print("\n" + "=" * 60)
    print(f"Ingestion Complete in {duration:.1f} seconds!")
    print(f"Total Reports Indexed: {len(docs)}")
    print(f"Total Chunks Created: {chunk_count}")
    print("=" * 60)
    for d in docs:
        print(f"  - {d['filename']:<35} | {d['page_count']:>3} pages | Status: {d['status']}")

if __name__ == "__main__":
    main()
