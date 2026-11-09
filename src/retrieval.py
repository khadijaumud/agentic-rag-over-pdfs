"""Hybrid Retrieval Engine:
- Lexical Search (SQLite FTS5 BM25)
- Dense Search (Local MiniLM Cosine Similarity)
- Reciprocal Rank Fusion (RRF)
- FlashRank Cross-Encoder Reranking
- Relevance Threshold Cutoff
"""

import re
from typing import List, Dict, Any, Optional
import numpy as np

from src.db import get_connection, get_all_chunks
from src.embeddings import embed_query, compute_cosine_similarity
from src.config import (
    DEFAULT_TOP_K_DENSE,
    DEFAULT_TOP_K_LEXICAL,
    DEFAULT_RRF_K,
    DEFAULT_RERANK_TOP_N,
    DEFAULT_RELEVANCE_THRESHOLD
)
from src import vector_store

_RANKER = None

_LEXICAL_SYNONYMS = {
    "staff": ("employees", "personnel", "workforce", "people"),
    "employees": ("staff", "personnel", "workforce", "people"),
    "professionals": ("employees", "personnel", "workforce", "people"),
    "leadership": ("directors", "officers", "management", "executives"),
    "positions": ("roles", "offices", "appointments"),
    "changed": ("appointed", "resigned", "retired", "elected"),
    "launched": ("introduced", "developed", "products", "enhancements"),
    "products": ("product", "platforms", "enhancements"),
    "acquisitions": ("acquisition", "acquired", "merger", "mergers"),
    "mergers": ("merger", "acquisition", "acquired"),
    "operations": ("operating",),
}

def get_reranker():
    """Lazy load FlashRank Ranker."""
    global _RANKER
    if _RANKER is None:
        try:
            from flashrank import Ranker
            _RANKER = Ranker()
        except Exception as e:
            print(f"[Reranker Warning] FlashRank not initialized: {e}")
            _RANKER = None
    return _RANKER

def sanitize_fts5_query(query: str) -> str:
    """Sanitize user query for SQLite FTS5 MATCH syntax."""
    # Extract alphanumeric words and numbers
    tokens = re.findall(r"[A-Za-z0-9_]+", query)
    if not tokens:
        return '""'
    expanded = list(tokens)
    for token in tokens:
        expanded.extend(_LEXICAL_SYNONYMS.get(token.lower(), ()))
    # Use OR or AND for tokens, quoting each token safely
    return " OR ".join(f'"{t}"' for t in dict.fromkeys(expanded))

def lexical_search(
    query: str,
    document_filter: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K_LEXICAL
) -> List[Dict[str, Any]]:
    """BM25 search via SQLite FTS5."""
    sanitized = sanitize_fts5_query(query)
    if not sanitized or sanitized == '""':
        return []

    conn = get_connection()
    cur = conn.cursor()
    
    if document_filter:
        sql = """
            SELECT c.id, c.document_id, c.page_number, c.content, bm25(chunks_fts) as score
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.id
            WHERE chunks_fts MATCH ? AND c.document_id = ?
            ORDER BY score ASC
            LIMIT ?
        """
        cur.execute(sql, (sanitized, document_filter, top_k))
    else:
        sql = """
            SELECT c.id, c.document_id, c.page_number, c.content, bm25(chunks_fts) as score
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.id
            WHERE chunks_fts MATCH ?
            ORDER BY score ASC
            LIMIT ?
        """
        cur.execute(sql, (sanitized, top_k))
    
    rows = cur.fetchall()
    conn.close()
    
    results = []
    for r in rows:
        results.append({
            "id": r["id"],
            "document_id": r["document_id"],
            "page_number": r["page_number"],
            "content": r["content"],
            # FTS5 bm25 produces negative/lower values for better matches, invert for positive score
            "lexical_score": -float(r["score"])
        })
    return results

def dense_search(
    query: str,
    document_filter: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K_DENSE
) -> List[Dict[str, Any]]:
    """Dense vector search using normalized cosine similarity."""
    query_emb = embed_query(query)
    if vector_store.is_available():
        try:
            return vector_store.search(query_emb, top_k, document_filter)
        except Exception as exc:
            print(f"[Vector DB Warning] Qdrant search failed; using SQLite embeddings: {exc}")
    
    conn = get_connection()
    cur = conn.cursor()
    if document_filter:
        cur.execute("SELECT id, document_id, page_number, content, embedding FROM chunks WHERE document_id = ? AND embedding IS NOT NULL", (document_filter,))
    else:
        cur.execute("SELECT id, document_id, page_number, content, embedding FROM chunks WHERE embedding IS NOT NULL")
    
    rows = cur.fetchall()
    conn.close()
    
    if not rows:
        return []
    
    doc_ids = []
    doc_metas = []
    emb_list = []
    
    for r in rows:
        emb_arr = np.frombuffer(r["embedding"], dtype=np.float32)
        emb_list.append(emb_arr)
        doc_ids.append(r["id"])
        doc_metas.append({
            "id": r["id"],
            "document_id": r["document_id"],
            "page_number": r["page_number"],
            "content": r["content"]
        })
    
    emb_matrix = np.stack(emb_list)
    scores = compute_cosine_similarity(query_emb, emb_matrix)
    
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        item = dict(doc_metas[idx])
        item["dense_score"] = float(scores[idx])
        results.append(item)
    
    return results

def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    lexical_results: List[Dict[str, Any]],
    k: int = DEFAULT_RRF_K
) -> List[Dict[str, Any]]:
    """Fuse dense and lexical results using Reciprocal Rank Fusion."""
    rrf_scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict[str, Any]] = {}
    
    for rank, item in enumerate(dense_results):
        cid = item["id"]
        doc_map[cid] = item
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank + 1))
        
    for rank, item in enumerate(lexical_results):
        cid = item["id"]
        if cid not in doc_map:
            doc_map[cid] = item
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank + 1))
        
    sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
    fused = []
    for cid in sorted_cids:
        item = dict(doc_map[cid])
        item["rrf_score"] = rrf_scores[cid]
        fused.append(item)
        
    return fused

def rerank_passages(
    query: str,
    candidates: List[Dict[str, Any]],
    top_n: int = DEFAULT_RERANK_TOP_N
) -> List[Dict[str, Any]]:
    """Rerank candidates using FlashRank (or preserve RRF order if unavailable)."""
    if not candidates:
        return []
    
    ranker = get_reranker()
    if ranker is None:
        # Keep the threshold meaningful when the optional reranker is unavailable.
        # RRF scores are normalized relative to this shortlist's best result.
        best_rrf = max((float(c.get("rrf_score", 0.0)) for c in candidates), default=1.0)
        results = []
        for candidate in candidates[:top_n]:
            item = dict(candidate)
            item["rerank_score"] = float(item.get("rrf_score", 0.0)) / max(best_rrf, 1e-12)
            results.append(item)
        return results
    
    try:
        from flashrank import RerankRequest
        passages = [{"id": c["id"], "text": c["content"], "meta": c} for c in candidates]
        rerank_req = RerankRequest(query=query, passages=passages)
        ranked = ranker.rerank(rerank_req)
        
        results = []
        for r in ranked[:top_n]:
            item = dict(r["meta"])
            item["rerank_score"] = float(r.get("score", 0.0))
            results.append(item)
        return results
    except Exception as e:
        print(f"[Reranker Error] {e}. Falling back to RRF order.")

        best_rrf = max(
            (
                float(candidate.get("rrf_score", 0.0))
                for candidate in candidates
            ),
            default=1.0
        )

        results = []

        for candidate in candidates[:top_n]:
            item = dict(candidate)

            item["rerank_score"] = (
                float(item.get("rrf_score", 0.0))
                / max(best_rrf, 1e-12)
            )

        results.append(item)

    return results

def hybrid_search(
    query: str,
    document_filter: Optional[str] = None,
    top_k: int = DEFAULT_RERANK_TOP_N,
    threshold: float = DEFAULT_RELEVANCE_THRESHOLD
) -> List[Dict[str, Any]]:
    """End-to-end hybrid search pipeline:
    1. Dense Search
    2. Lexical Search
    3. RRF Fusion
    4. FlashRank Rerank
    5. Relevance Threshold Filter
    """
    dense_res = dense_search(query, document_filter=document_filter, top_k=DEFAULT_TOP_K_DENSE)
    lexical_res = lexical_search(query, document_filter=document_filter, top_k=DEFAULT_TOP_K_LEXICAL)
    
    if not dense_res and not lexical_res:
        return []
    
    fused = reciprocal_rank_fusion(dense_res, lexical_res, k=DEFAULT_RRF_K)
    
    # Rerank top 15 fused candidates
    candidates_to_rerank = fused[:15]
    reranked = rerank_passages(query, candidates_to_rerank, top_n=top_k)
    
    # Filter by threshold if rerank_score exists
    filtered = []
    for item in reranked:
        score = item.get("rerank_score", item.get("rrf_score", 0.0))
        if score >= threshold:
            filtered.append(item)
            
    return filtered
