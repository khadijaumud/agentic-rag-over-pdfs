"""Qdrant-backed dense vector index with explicit SQLite fallback for local tests."""

import logging
import uuid
from typing import Any, Dict, Iterable, List, Optional

import requests

from src.config import VECTOR_DB_TIMEOUT_SECONDS, VECTOR_DB_URL, VECTOR_COLLECTION

LOGGER = logging.getLogger(__name__)


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"agentic-rag:{chunk_id}"))


def _url(path: str) -> str:
    return f"{VECTOR_DB_URL.rstrip('/')}{path}"


def is_available() -> bool:
    try:
        response = requests.get(_url("/collections"), timeout=VECTOR_DB_TIMEOUT_SECONDS)
        return response.ok
    except requests.RequestException:
        return False


def _ensure_collection(size: int) -> None:
    response = requests.get(
        _url(f"/collections/{VECTOR_COLLECTION}"),
        timeout=VECTOR_DB_TIMEOUT_SECONDS,
    )
    if response.ok:
        return
    if response.status_code != 404:
        response.raise_for_status()
    created = requests.put(
        _url(f"/collections/{VECTOR_COLLECTION}"),
        json={"vectors": {"size": size, "distance": "Cosine"}},
        timeout=VECTOR_DB_TIMEOUT_SECONDS,
    )
    created.raise_for_status()


def upsert_chunks(chunks: Iterable[Dict[str, Any]], batch_size: int = 64) -> int:
    batch: List[Dict[str, Any]] = []
    total = 0
    for chunk in chunks:
        embedding = chunk.get("embedding")
        if embedding is None:
            continue
        vector = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
        batch.append(
            {
                "id": _point_id(chunk["id"]),
                "vector": vector,
                "payload": {
                    "chunk_id": chunk["id"],
                    "document_id": chunk["document_id"],
                    "page_number": chunk["page_number"],
                    "content": chunk["content"],
                },
            }
        )
        if len(batch) >= batch_size:
            total += _upsert_batch(batch)
            batch = []
    if batch:
        total += _upsert_batch(batch)
    return total


def _upsert_batch(points: List[Dict[str, Any]]) -> int:
    _ensure_collection(len(points[0]["vector"]))
    response = requests.put(
        _url(f"/collections/{VECTOR_COLLECTION}/points?wait=true"),
        json={"points": points},
        timeout=VECTOR_DB_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return len(points)


def search(
    query_vector: Any,
    top_k: int,
    document_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    vector = query_vector.tolist() if hasattr(query_vector, "tolist") else list(query_vector)
    body: Dict[str, Any] = {
        "vector": vector,
        "limit": top_k,
        "with_payload": True,
    }
    if document_filter:
        body["filter"] = {
            "must": [
                {"key": "document_id", "match": {"value": document_filter}},
            ]
        }
    response = requests.post(
        _url(f"/collections/{VECTOR_COLLECTION}/points/search"),
        json=body,
        timeout=VECTOR_DB_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return [
        {
            "id": item["payload"]["chunk_id"],
            "document_id": item["payload"]["document_id"],
            "page_number": item["payload"]["page_number"],
            "content": item["payload"]["content"],
            "dense_score": float(item.get("score", 0.0)),
        }
        for item in response.json().get("result", [])
    ]


def delete_document(document_id: str) -> None:
    response = requests.post(
        _url(f"/collections/{VECTOR_COLLECTION}/points/delete?wait=true"),
        json={
            "filter": {
                "must": [
                    {"key": "document_id", "match": {"value": document_id}},
                ]
            }
        },
        timeout=VECTOR_DB_TIMEOUT_SECONDS,
    )
    if response.status_code != 404:
        response.raise_for_status()


def count() -> int:
    response = requests.post(
        _url(f"/collections/{VECTOR_COLLECTION}/points/count"),
        json={"exact": True},
        timeout=VECTOR_DB_TIMEOUT_SECONDS,
    )
    if response.status_code == 404:
        return 0
    response.raise_for_status()
    return int(response.json()["result"]["count"])
