"""Integration tests for FastAPI endpoints using TestClient with mocked LLM.
Validates HTTP contracts, status codes, and DB operations without spending API tokens.
"""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from src.app import app
from src.db import init_db, save_document, save_chunks

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_test_db():
    init_db()
    # Seed a sample document and chunk for testing
    save_document("RWE AG.pdf", "RWE AG.pdf", page_count=285, file_size=1000, status="ready")
    save_chunks([{
        "id": "RWE AG.pdf_p47_c0",
        "document_id": "RWE AG.pdf",
        "page_number": 47,
        "chunk_index": 0,
        "content": "[RWE AG.pdf - Page 47]\nTotal power generation capacity of RWE AG was 39.3 GW at year end.",
        "token_count": 20,
        "embedding": None
    }])

def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["total_documents"] >= 1

def test_list_documents():
    resp = client.get("/documents")
    assert resp.status_code == 200
    docs = resp.json()
    assert isinstance(docs, list)
    assert any(d["filename"] == "RWE AG.pdf" for d in docs)

def test_chat_pipeline_mode():
    mock_llm_response = {
        "text": "The total power capacity of RWE AG was 39.3 GW [RWE AG.pdf, p.47].",
        "prompt_tokens": 120,
        "completion_tokens": 25,
        "cost_usd": 0.0001,
        "model": "gemini-2.5-flash",
        "provider": "gemini"
    }
    
    with patch("src.pipeline.call_llm_with_fallback", return_value=mock_llm_response):
        payload = {
            "question": "What was the total power generation capacity of RWE AG?",
            "mode": "pipeline"
        }
        resp = client.post("/chat", json=payload, headers={"X-User-Id": "test_user"})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "citations" in data
        assert "usage" in data
        assert data["usage"]["usd"] >= 0.0001

def test_chat_agent_mode():
    mock_action = (
        "Thought: I will search for generation capacity.\n"
        "Action: search_documents\n"
        'Action Input: {"query": "power generation capacity", "document_filter": "RWE AG.pdf"}'
    )
    mock_final = (
        "Thought: Found the information.\n"
        "Final Answer: Total power generation capacity was 39.3 GW [RWE AG.pdf, p.47]."
    )
    
    side_effects = [
        {"text": mock_action, "prompt_tokens": 100, "completion_tokens": 30, "cost_usd": 0.0001, "model": "mock", "provider": "mock"},
        {"text": mock_final, "prompt_tokens": 250, "completion_tokens": 30, "cost_usd": 0.0002, "model": "mock", "provider": "mock"}
    ]
    
    with patch("src.agent.call_llm_with_fallback", side_effect=side_effects):
        payload = {
            "question": "What was the total power generation capacity of RWE AG?",
            "mode": "agent"
        }
        resp = client.post("/chat", json=payload, headers={"X-User-Id": "test_user"})
        assert resp.status_code == 200
        data = resp.json()
        assert "39.3 GW" in data["answer"]
        assert len(data["citations"]) >= 1
        assert data["citations"][0]["document"] == "RWE AG.pdf"
        assert data["citations"][0]["page"] == 47
        assert len(data.get("agent_steps", [])) >= 1

def test_feedback_submission():
    payload = {
        "message_id": "test_msg_001",
        "score": 1,
        "notes": "Accurate page reference."
    }
    resp = client.post("/feedback", json=payload)
    assert resp.status_code == 200
    assert resp.json()["status"] == "recorded"
