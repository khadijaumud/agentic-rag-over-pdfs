"""Local embedding provider using SentenceTransformers (all-MiniLM-L6-v2).
Zero-cost, runs on local CPU, produces 384-dim normalized dense vectors.
"""

from typing import List, Optional
import numpy as np
from src.config import LOCAL_EMBEDDING_MODEL

_MODEL = None

def get_embedding_model():
    """Lazy load local sentence-transformers model."""
    global _MODEL
    if _MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            # Load local model (cached locally in huggingface cache)
            _MODEL = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
        except Exception as e:
            print(f"[Embedding Warning] Failed to load {LOCAL_EMBEDDING_MODEL}: {e}")
            _MODEL = None
    return _MODEL

def embed_texts(texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
    """Embed a list of text strings into normalized float32 numpy vectors."""
    if not texts:
        return []
    
    model = get_embedding_model()
    if model is not None:
        try:
            embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
            return [np.array(e, dtype=np.float32) for e in embeddings]
        except Exception as e:
            print(f"[Embedding Error] Encoding failed: {e}")
    
    # Deterministic fallback vector if local model unavailable
    fallback_dim = 384
    return [np.zeros(fallback_dim, dtype=np.float32) for _ in texts]

def embed_query(query: str) -> np.ndarray:
    """Embed a single search query."""
    res = embed_texts([query])
    return res[0] if res else np.zeros(384, dtype=np.float32)

def compute_cosine_similarity(query_emb: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
    """Compute cosine similarities between a normalized query vector and 2D matrix of doc vectors."""
    if doc_embs.ndim == 1:
        doc_embs = doc_embs.reshape(1, -1)
    # If already normalized, dot product equals cosine similarity
    dot_products = np.dot(doc_embs, query_emb)
    return dot_products
