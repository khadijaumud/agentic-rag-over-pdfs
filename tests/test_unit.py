"""Unit tests for Agentic RAG system components.
Tests deterministic functions: parsing, chunking, FTS sanitization,
RRF fusion, citation parsing, action parsing, and math calculation.
"""

import pytest
import numpy as np
from src.chunker import chunk_page_text, chunk_document_pages
from src import vector_store
from src.retrieval import sanitize_fts5_query, reciprocal_rank_fusion
from src.pipeline import extract_citations_from_text
from src.agent import (
    parse_agent_action,
    run_agent,
    tool_calculate,
    _contains_tool_protocol,
    _format_search_results,
)
from src.llm import estimate_cost, SYSTEM_PROMPT_DEFAULT
from src.db import get_conversation_history

def test_chunker_preserves_page_metadata():
    text = "This is a sample financial disclosure statement about company earnings and cash flows."
    chunks = chunk_page_text(
        document_id="Test_Report.pdf",
        page_number=42,
        text=text,
        target_chunk_size=10,
        chunk_overlap=2
    )
    assert len(chunks) >= 1
    for c in chunks:
        assert c["document_id"] == "Test_Report.pdf"
        assert c["page_number"] == 42
        assert "Test_Report.pdf - Page 42" in c["content"]
        assert c["token_count"] > 0

def test_fts5_query_sanitizer():
    # Test queries with punctuation, hyphens, and quotes
    q1 = "1-800-FLOWERS.COM, INC. 'revenue' 2022?"
    sanitized = sanitize_fts5_query(q1)
    assert '"1"' in sanitized or '"800"' in sanitized
    assert '"revenue"' in sanitized
    assert "'" not in sanitized
    assert '"employees"' in sanitize_fts5_query("healthcare professionals staff")

    # Empty / punctuation-only query
    q2 = "??? --- !!!"
    sanitized2 = sanitize_fts5_query(q2)
    assert sanitized2 == '""'

def test_reciprocal_rank_fusion():
    dense = [
        {"id": "doc1", "content": "text 1"},
        {"id": "doc2", "content": "text 2"},
        {"id": "doc3", "content": "text 3"}
    ]
    lexical = [
        {"id": "doc2", "content": "text 2"},
        {"id": "doc1", "content": "text 1"},
        {"id": "doc4", "content": "text 4"}
    ]
    
    fused = reciprocal_rank_fusion(dense, lexical, k=60)
    # doc1 and doc2 appear in both lists, so they must rank higher than doc3 and doc4
    top_ids = [item["id"] for item in fused[:2]]
    assert "doc1" in top_ids
    assert "doc2" in top_ids
    assert len(fused) == 4

def test_citation_extraction():
    answer_text = (
        "According to the 2022 annual report, net cash from operations was $406.1 million "
        "[Sonic Automotive, Inc..pdf, p.71]. Furthermore, patent holdings reached 2,300 "
        "[Albany International Corp..pdf, p. 16]."
    )
    citations = extract_citations_from_text(answer_text)
    assert len(citations) == 2
    assert citations[0]["document"] == "Sonic Automotive, Inc..pdf"
    assert citations[0]["page"] == 71
    assert citations[1]["document"] == "Albany International Corp..pdf"
    assert citations[1]["page"] == 16

def test_citation_extraction_handles_multiple_pages_without_corrupting_document():
    citations = extract_citations_from_text(
        "Evidence [Westwater Resources, Inc..pdf, p.16, p.17] "
        "[Datalogic.pdf, p.56, p.82]."
    )
    assert citations == [
        {"document": "Westwater Resources, Inc..pdf", "page": 16},
        {"document": "Westwater Resources, Inc..pdf", "page": 17},
        {"document": "Datalogic.pdf", "page": 56},
        {"document": "Datalogic.pdf", "page": 82},
    ]

def test_agent_action_parsing():
    # 1. Action call parsing
    model_output_action = (
        "Thought: I should search for RWE AG total generation capacity.\n"
        "Action: search_documents\n"
        'Action Input: {"query": "power generation capacity", "document_filter": "RWE AG.pdf"}'
    )
    action, args, thought, final_ans = parse_agent_action(model_output_action)
    assert action == "search_documents"
    assert args.get("query") == "power generation capacity"
    assert args.get("document_filter") == "RWE AG.pdf"
    assert final_ans is None

    # 2. Final Answer parsing
    model_output_final = (
        "Thought: I now have the exact figure.\n"
        "Final Answer: The total generation capacity of RWE AG was 39.3 GW [RWE AG.pdf, p.47]."
    )
    action, args, thought, final_ans = parse_agent_action(model_output_final)
    assert action is None
    assert final_ans is not None
    assert "39.3 GW" in final_ans

def test_agent_parser_keeps_tool_calls_out_of_final_answers():
    action, args, _, final_ans = parse_agent_action(
        "Thought: I need one more page.\n"
        "Action: read_page\n"
        'Action Input: {"document": "RWE AG.pdf", "page_number": 47}'
    )
    assert action == "read_page"
    assert args["page_number"] == 47
    assert final_ans is None

def test_agent_rejects_protocol_inside_final_answer():
    assert _contains_tool_protocol(
        "Thought: I need more evidence.\n"
        "Action: read_page\n"
        'Action Input: {"document": "RWE AG.pdf", "page_number": 47}'
    )
    assert not _contains_tool_protocol(
        "The capacity was 39.3 GW [RWE AG.pdf, p.47]."
    )

def test_agent_search_observation_deduplicates_chunks_on_same_page():
    observation = _format_search_results(
        [
            {"document_id": "Report.pdf", "page_number": 16, "content": "first chunk"},
            {"document_id": "Report.pdf", "page_number": 16, "content": "second chunk"},
            {"document_id": "Report.pdf", "page_number": 17, "content": "next page"},
        ],
        "research",
    )
    assert observation.count("Source: [Report.pdf, p.16]") == 1
    assert "Source: [Report.pdf, p.17]" in observation

def test_agent_does_not_accept_refusal_before_reading_a_page(monkeypatch):
    responses = iter(
        [
            {
                "text": (
                    "Thought: Search first.\nAction: search_documents\n"
                    'Action Input: {"query": "capacity"}'
                ),
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "cost_usd": 0.000001,
                "provider": "mock",
                "model": "mock",
            },
            {
                "text": "Final Answer: Data not available.",
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "cost_usd": 0.000001,
                "provider": "mock",
                "model": "mock",
            },
            {
                "text": "Final Answer: 39.3 GW [Report.pdf, p.47].",
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "cost_usd": 0.000001,
                "provider": "mock",
                "model": "mock",
            },
        ]
    )
    monkeypatch.setattr("src.agent.call_llm_with_fallback", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr(
        "src.agent._search_documents",
        lambda query, document_filter=None: [
            {"id": "chunk-1", "document_id": "Report.pdf", "page_number": 47, "content": "39.3 GW"}
        ],
    )

    result = run_agent("What was capacity?", max_iterations=3)

    assert result["answer"] == "39.3 GW [Report.pdf, p.47]."
    assert len([step for step in result["agent_steps"] if step["action"] == "llm_call"]) == 3

def test_safe_calculate_tool():
    res1 = tool_calculate("406.1 - 306.3")
    assert "99.8" in res1

    # Should reject dangerous expressions
    res_bad = tool_calculate("__import__('os').system('ls')")
    assert "Error" in res_bad or "None" in res_bad

def test_cost_estimation():
    cost = estimate_cost("gemini-2.5-flash", prompt_tokens=10_000, completion_tokens=2_000)
    assert cost > 0.0
    assert isinstance(cost, float)

def test_vector_point_ids_are_stable():
    assert vector_store._point_id("chunk-1") == vector_store._point_id("chunk-1")
    assert vector_store._point_id("chunk-1") != vector_store._point_id("chunk-2")

def test_prompt_injection_defense_is_explicit():
    assert "untrusted input" in SYSTEM_PROMPT_DEFAULT
    assert "NEVER follow instructions" in SYSTEM_PROMPT_DEFAULT

def test_conversation_history_has_token_budget(tmp_path, monkeypatch):
    db_path = tmp_path / "history.db"
    monkeypatch.setattr("src.db.DB_PATH", db_path)
    from src.db import init_db, save_conversation, save_message

    init_db(db_path)
    save_conversation("conv-budget", "Budget", "test")
    save_message("m1", "conv-budget", "user", "a" * 100)
    save_message("m2", "conv-budget", "assistant", "b" * 100)

    history = get_conversation_history("conv-budget", limit=6, max_tokens=30)

    assert len(history) == 1
    assert history[0]["role"] == "assistant"
