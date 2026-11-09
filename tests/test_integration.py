"""Integration tests for FastAPI endpoints using TestClient with mocked LLM.
Validates HTTP contracts, status codes, and DB operations without spending API tokens.
"""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from src.app import app
from src.db import init_db, save_document, save_chunks
from src.llm import call_llm_with_fallback

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

def test_citation_passage_endpoint():
    resp = client.get("/documents/RWE AG.pdf/pages/47")
    assert resp.status_code == 200
    data = resp.json()
    assert data["document"] == "RWE AG.pdf"
    assert data["page"] == 47
    assert "39.3 GW" in "\n".join(data["passages"])

def test_pdf_file_endpoint(tmp_path, monkeypatch):
    pdf_path = tmp_path / "RWE AG.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n% test")
    monkeypatch.setattr("src.app.PDF_DIR", tmp_path)
    resp = client.get("/documents/RWE AG.pdf/file")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"

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
        assert any("usage" in step for step in data["agent_steps"])

def test_feedback_submission():
    payload = {
        "message_id": "test_msg_001",
        "score": 1,
        "notes": "Accurate page reference."
    }
    resp = client.post("/feedback", json=payload)
    assert resp.status_code == 200
    assert resp.json()["status"] == "recorded"

def test_missing_llm_configuration_is_explicit(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("FALLBACK_PROVIDER", "groq")
    with pytest.raises(RuntimeError, match="No LLM provider configured"):
        call_llm_with_fallback([{"role": "user", "content": "test"}])

def test_primary_provider_success(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "configured")
    monkeypatch.setenv("GROQ_API_KEY", "")
    response = {
        "text": "primary answer",
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "cost_usd": 0.001,
        "model": "primary-test",
        "provider": "gemini",
    }
    with patch("src.llm.call_gemini", return_value=response) as primary:
        result = call_llm_with_fallback([{"role": "user", "content": "test"}])
    primary.assert_called_once()
    assert result == response

def test_primary_failure_fallback_success(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "configured")
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("FALLBACK_PROVIDER", "ollama")
    fallback = {
        "text": "fallback answer",
        "prompt_tokens": 21,
        "completion_tokens": 9,
        "cost_usd": 0.002,
        "model": "llama3.2",
        "provider": "ollama",
    }
    with patch("src.llm.call_gemini", side_effect=RuntimeError("primary down")):
        with patch("src.llm.call_ollama", return_value=fallback) as fallback_call:
            result = call_llm_with_fallback([{"role": "user", "content": "test"}])
    fallback_call.assert_called_once()
    assert result["provider"] == "ollama"
    assert result["prompt_tokens"] == 21
    assert result["cost_usd"] == 0.002

def test_both_providers_fail_gracefully(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "configured")
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("FALLBACK_PROVIDER", "ollama")
    with patch("src.llm.call_gemini", side_effect=RuntimeError("primary down")):
        with patch("src.llm.call_ollama", side_effect=RuntimeError("fallback down")):
            with pytest.raises(RuntimeError, match="All LLM providers failed"):
                call_llm_with_fallback([{"role": "user", "content": "test"}])
