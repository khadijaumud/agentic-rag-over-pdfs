"""Chunking strategy for financial annual reports.
Keeps 1-based page metadata, handles long pages with overlap,
and prefixes document and page context headers to chunks.
"""

from typing import List, Dict, Any

def estimate_tokens(text: str) -> int:
    """Fast estimation of tokens (roughly 4 chars per token in English)."""
    return max(1, len(text) // 4)

def chunk_page_text(
    document_id: str,
    page_number: int,
    text: str,
    target_chunk_size: int = 600,  # ~600 tokens (approx 2400 chars)
    chunk_overlap: int = 100       # ~100 tokens (approx 400 chars)
) -> List[Dict[str, Any]]:
    """Chunk text of a single page into overlapping segments.
    If page is small (<= target_chunk_size), returns 1 chunk.
    Always preserves exact 1-based page number.
    """
    if not text.strip():
        return []

    tokens = text.split()
    if len(tokens) <= target_chunk_size:
        chunk_content = f"[{document_id} - Page {page_number}]\n{text.strip()}"
        return [{
            "id": f"{document_id}_p{page_number}_c0",
            "document_id": document_id,
            "page_number": page_number,
            "chunk_index": 0,
            "content": chunk_content,
            "token_count": len(tokens)
        }]

    chunks = []
    chunk_idx = 0
    start = 0
    step = target_chunk_size - chunk_overlap
    
    while start < len(tokens):
        end = min(start + target_chunk_size, len(tokens))
        chunk_words = tokens[start:end]
        chunk_body = " ".join(chunk_words)
        chunk_content = f"[{document_id} - Page {page_number}]\n{chunk_body}"
        
        chunks.append({
            "id": f"{document_id}_p{page_number}_c{chunk_idx}",
            "document_id": document_id,
            "page_number": page_number,
            "chunk_index": chunk_idx,
            "content": chunk_content,
            "token_count": len(chunk_words)
        })
        
        if end >= len(tokens):
            break
        start += step
        chunk_idx += 1

    return chunks

def chunk_document_pages(document_id: str, pages: List[Dict[str, Any]], target_chunk_size: int = 600, chunk_overlap: int = 100) -> List[Dict[str, Any]]:
    """Chunk all pages of a parsed PDF."""
    all_chunks = []
    for p in pages:
        page_chunks = chunk_page_text(
            document_id=document_id,
            page_number=p["page_number"],
            text=p["text"],
            target_chunk_size=target_chunk_size,
            chunk_overlap=chunk_overlap
        )
        all_chunks.extend(page_chunks)
    return all_chunks
