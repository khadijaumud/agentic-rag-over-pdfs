"""Regression tests for the sidebar layout and the question-answer flow."""

import ast
import json
import typing
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.app import app
from src.db import init_db, save_document, save_chunks, get_conversation_history
from src.retrieval import rerank_passages

client = TestClient(app)

UI_PATH = Path(__file__).resolve().parents[1] / "src" / "ui.py"

MOCK_PIPELINE = {
    "text": "Capacity was 39.3 GW [RWE AG.pdf, p.47].",
    "prompt_tokens": 50,
    "completion_tokens": 10,
    "cost_usd": 0.0001,
    "model": "mock",
    "provider": "mock",
}


def _seed() -> None:
    init_db()
    save_document("RWE AG.pdf", "RWE AG.pdf", page_count=285, file_size=1000, status="ready")
    save_document("Datalogic.pdf", "Datalogic.pdf", page_count=240, file_size=1000, status="ready")
    save_chunks([
        {"id": "RWE AG.pdf_p47_c0", "document_id": "RWE AG.pdf", "page_number": 47, "chunk_index": 0,
         "content": "Total power generation capacity of RWE AG was 39.3 GW.", "token_count": 12, "embedding": None},
        {"id": "Datalogic.pdf_p56_c0", "document_id": "Datalogic.pdf", "page_number": 56, "chunk_index": 0,
         "content": "Datalogic power generation capacity is not reported.", "token_count": 10, "embedding": None},
    ])


def _ui_tree():
    return ast.parse(UI_PATH.read_text(encoding="utf-8"))


def _sidebar_body(tree=None):
    tree = tree or _ui_tree()
    for node in tree.body:
        if isinstance(node, ast.With):
            for item in node.items:
                expr = item.context_expr
                if isinstance(expr, ast.Attribute) and expr.attr == "sidebar":
                    return node
    raise AssertionError("src/ui.py has no top-level `with st.sidebar:` block")


def _calls_in(node):
    return [
        ast.unparse(call.func)
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
    ]


def test_sidebar_contains_library_filter_mode_and_history():
    sidebar = _sidebar_body()
    calls = _calls_in(sidebar)
    assert "render_library" in calls
    assert "st.selectbox" in calls
    assert "st.radio" in calls
    assert "st.file_uploader" in calls
    assert "load_conversation" in calls


def test_library_selects_ready_pdf_without_rendering_a_link():
    source = UI_PATH.read_text(encoding="utf-8")
    assert "on_click=select_document" in source
    assert 'st.session_state.document_selector = document_id' in source
    assert 'args=(filename,)' in source
    assert "target=\"_blank\"" not in source


def test_document_filter_uses_filename_contract_and_has_clear_control():
    source = UI_PATH.read_text(encoding="utf-8")
    assert 'document_options = [None] + [str(document["filename"]) for document in ready_documents]' in source
    assert 'key="document_selector"' in source
    assert "clear_document_filter" in source
    assert "on_click=clear_document_filter" in source
    assert '"document_filter": st.session_state.document_filter' in source


def test_done_answer_is_rendered_and_preserved_after_stream():
    source = UI_PATH.read_text(encoding="utf-8")
    assert 'final_answer = str(done.get("answer") or "".join(answer_parts)).strip()' in source
    assert "placeholder.markdown(final_answer)" in source
    assert 'if completed_answer:' in source
    assert '"answer": completed_answer' in source


def test_sse_parser_handles_event_boundaries_and_multiline_data():
    tree = _ui_tree()
    parser_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "parse_sse_events")
    namespace = {"json": json, **vars(typing)}
    exec(compile(ast.Module(body=[parser_node], type_ignores=[]), str(UI_PATH), "exec"), namespace)
    parse_sse_events = namespace["parse_sse_events"]
    events = list(parse_sse_events([
        b'data: {"event":"token","token":"hello"}\r\n',
        b"\r\n",
        'data: {"event":"done",\n',
        'data: "answer":"hello"}\n',
        "\n",
    ]))
    assert events == [
        {"event": "token", "token": "hello"},
        {"event": "done", "answer": "hello"},
    ]


def test_sidebar_widgets_bind_session_state_keys():
    sidebar = _sidebar_body()
    keys = {}
    for call in ast.walk(sidebar):
        if isinstance(call, ast.Call) and ast.unparse(call.func) in ("st.selectbox", "st.radio"):
            for kw in call.keywords:
                if kw.arg == "key":
                    keys[ast.unparse(call.func)] = ast.literal_eval(kw.value)
    assert keys == {"st.selectbox": "document_selector", "st.radio": "mode"}


def test_dashboard_does_not_render_sidebar_controls():
    tree = _ui_tree()
    sidebar = _sidebar_body(tree)
    outside = [node for node in tree.body if node is not sidebar and not isinstance(node, ast.FunctionDef)]
    calls = [c for node in outside for c in _calls_in(node)]
    assert "st.selectbox" not in calls
    assert "st.radio" not in calls
    assert "render_library" not in calls


def test_rerank_fallback_keeps_every_candidate():
    candidates = [
        {"id": f"c{i}", "content": f"text {i}", "rrf_score": 1.0 / (i + 1)} for i in range(5)
    ]

    class BrokenRanker:
        def rerank(self, request):
            raise RuntimeError("ranker exploded")

    with patch("src.retrieval.get_reranker", return_value=BrokenRanker()):
        ranked = rerank_passages("query", candidates, top_n=3)
    assert [item["id"] for item in ranked] == ["c0", "c1", "c2"]
    assert ranked[0]["rerank_score"] == 1.0


def test_document_filter_reaches_pipeline_retrieval():
    _seed()
    with patch("src.pipeline.call_llm_with_fallback", return_value=MOCK_PIPELINE):
        with patch("src.pipeline.hybrid_search", wraps=__import__("src.retrieval", fromlist=["hybrid_search"]).hybrid_search) as search:
            resp = client.post(
                "/chat",
                json={"question": "power generation capacity", "mode": "pipeline", "document_filter": "Datalogic.pdf"},
                headers={"X-User-Id": "regression_user"},
            )
    assert resp.status_code == 200
    assert search.call_args.kwargs["document_filter"] == "Datalogic.pdf"
    assert resp.json()["mode"] == "pipeline"


def test_unknown_document_filter_is_rejected():
    _seed()
    resp = client.post(
        "/chat",
        json={"question": "anything", "mode": "pipeline", "document_filter": "Missing.pdf"},
        headers={"X-User-Id": "regression_user"},
    )
    assert resp.status_code == 404
    assert "Missing.pdf" in resp.json()["detail"]


def test_invalid_mode_is_rejected():
    resp = client.post("/chat", json={"question": "anything", "mode": "oracle"})
    assert resp.status_code == 422


def test_stream_emits_steps_tokens_and_done_with_ids():
    _seed()
    with patch("src.pipeline.call_llm_with_fallback", return_value=MOCK_PIPELINE):
        resp = client.get(
            "/chat/stream",
            params={"question": "capacity of RWE", "mode": "pipeline", "document_filter": "RWE AG.pdf",
                    "conversation_id": "conv_stream_regression"},
            headers={"X-User-Id": "regression_user"},
        )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = [json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ")]
    kinds = [event["event"] for event in events]
    assert kinds[0] == "start"
    assert "token" in kinds
    assert kinds[-1] == "done"
    done = events[-1]
    assert done["conversation_id"] == "conv_stream_regression"
    assert done["message_id"].startswith("msg_")
    assert done["answer"] == MOCK_PIPELINE["text"]
    assert done["citations"] == [{"document": "RWE AG.pdf", "page": 47}]
    answer = "".join(event["token"] for event in events if event["event"] == "token").strip()
    assert answer == MOCK_PIPELINE["text"]


def test_stream_reports_llm_failure_as_error_event(monkeypatch):
    _seed()
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("FALLBACK_PROVIDER", "groq")
    resp = client.get("/chat/stream", params={"question": "capacity of RWE", "mode": "pipeline"})
    events = [json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ")]
    assert events[-1]["event"] == "error"
    assert events[-1]["status"] == 503
    assert "No LLM provider configured" in events[-1]["detail"]


def test_follow_up_receives_prior_turns_in_order():
    _seed()
    conv = "conv_followup_regression"
    with patch("src.pipeline.call_llm_with_fallback", return_value=MOCK_PIPELINE) as llm:
        first = client.post("/chat", json={"question": "What is RWE capacity?", "mode": "pipeline", "conversation_id": conv})
        assert first.status_code == 200
        second = client.post("/chat", json={"question": "Which unit is the RWE capacity reported in?", "mode": "pipeline", "conversation_id": conv})
        assert second.status_code == 200
    messages = llm.call_args_list[-1].args[0]
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[0]["content"] == "What is RWE capacity?"
    assert messages[1]["content"] == MOCK_PIPELINE["text"]
    assert "Which unit is the RWE capacity reported in?" in messages[2]["content"]
    assert set(messages[0]) == {"role", "content"}

    history = get_conversation_history(conv, limit=10)
    assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]
    assert history[1]["usage"]["prompt_tokens"] == 50

    conversations = client.get("/conversations", params={"user_id": "default_user"}).json()
    title = next(c["title"] for c in conversations if c["id"] == conv)
    assert title == "What is RWE capacity?"


def test_agent_mode_enforces_document_filter_on_tools():
    _seed()
    responses = iter([
        {"text": 'Thought: search.\nAction: search_documents\nAction Input: {"query": "capacity", "document_filter": "Datalogic.pdf"}',
         "prompt_tokens": 1, "completion_tokens": 1, "cost_usd": 0.0, "provider": "mock", "model": "mock"},
        {"text": 'Thought: read.\nAction: read_page\nAction Input: {"document": "Datalogic.pdf", "page_number": 56}',
         "prompt_tokens": 1, "completion_tokens": 1, "cost_usd": 0.0, "provider": "mock", "model": "mock"},
        {"text": "Final Answer: Capacity was 39.3 GW [RWE AG.pdf, p.47].",
         "prompt_tokens": 1, "completion_tokens": 1, "cost_usd": 0.0, "provider": "mock", "model": "mock"},
    ])
    with patch("src.agent.call_llm_with_fallback", side_effect=lambda *a, **k: next(responses)):
        resp = client.post("/chat", json={"question": "capacity", "mode": "agent", "document_filter": "RWE AG.pdf"})
    assert resp.status_code == 200
    steps = resp.json()["agent_steps"]
    search_step = next(s for s in steps if s["action"] == "search_documents")
    assert "RWE AG.pdf" in search_step["observation"]
    assert "Datalogic" not in search_step["observation"]
    read_step = next(s for s in steps if s["action"] == "read_page")
    assert read_step["observation"].startswith("[Filter]")
