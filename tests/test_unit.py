"""Unit tests for Agentic RAG system components.
Tests deterministic functions: parsing, chunking, FTS sanitization,
RRF fusion, citation parsing, action parsing, and math calculation.
"""

import pytest
import numpy as np
from src.chunker import chunk_page_text, chunk_document_pages
from src.retrieval import sanitize_fts5_query, reciprocal_rank_fusion
from src.pipeline import extract_citations_from_text
from src.agent import parse_agent_action, tool_calculate
from src.llm import estimate_cost

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
