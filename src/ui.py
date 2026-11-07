"""Streamlit interface for the Agentic RAG research assistant.

The UI deliberately stays on top of the existing FastAPI contracts. It owns
presentation and session state only; retrieval, ingestion, citations, and
conversation persistence remain backend responsibilities.
"""

import time
from typing import Any, Dict, List, Optional

import requests
import streamlit as st


API_URL = "http://localhost:8000"
USER_ID = "streamlit_user"

st.set_page_config(
    page_title="Research Desk | Agentic RAG",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --navy: #0b1220;
    --ink: #172033;
    --muted: #6b7890;
    --line: #e7ebf2;
    --soft: #f5f7fb;
    --purple: #6d5dfc;
    --blue: #3478f6;
}

.stApp {
    background: #f8f9fc;
    color: var(--ink);
    font-family: "DM Sans", sans-serif;
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    color: var(--ink);
    font-family: "Space Grotesk", sans-serif;
    letter-spacing: -0.025em;
}
.block-container {
    max-width: 1440px;
    padding: 2rem 3rem 4rem;
}
[data-testid="stSidebar"] {
    background: var(--navy);
    border-right: 0;
}
[data-testid="stSidebar"] * { color: #dce4f4; }
[data-testid="stSidebar"] .stCaption { color: #8d9ab4; }
[data-testid="stSidebar"] .stButton button {
    border: 1px solid #293750;
    background: #151f32;
    color: #dce4f4;
}
[data-testid="stSidebar"] .stButton button:hover {
    border-color: #667cff;
    color: white;
}
[data-testid="stSidebar"] [data-baseweb="radio"] div[role="radiogroup"] {
    gap: 0.4rem;
}
[data-testid="stSidebar"] [data-baseweb="radio"] label {
    background: #111b2d;
    border: 1px solid #26344c;
    border-radius: 10px;
    padding: 0.55rem 0.7rem;
}
[data-testid="stSidebar"] [data-baseweb="radio"] label:has(input:checked) {
    background: #242450;
    border-color: #776dff;
}
.brand {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin-bottom: 2rem;
}
.brand-mark {
    width: 36px;
    height: 36px;
    display: grid;
    place-items: center;
    border-radius: 11px;
    background: linear-gradient(135deg, #766bff, #397cf7);
    color: white;
    font-size: 1.15rem;
    font-weight: 700;
}
.brand-title { color: white; font: 700 1rem "Space Grotesk", sans-serif; }
.brand-subtitle { color: #8d9ab4; font-size: 0.72rem; }
.hero {
    padding: 0.5rem 0 1.8rem;
}
.eyebrow {
    color: var(--purple);
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.hero h1 { margin: 0.3rem 0 0.35rem; font-size: clamp(2rem, 4vw, 3.2rem); }
.hero p { color: var(--muted); font-size: 1.02rem; max-width: 680px; }
.status-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
    margin: 0.9rem 0 1.4rem;
}
.status-pill, .source-pill {
    border: 1px solid var(--line);
    border-radius: 999px;
    background: white;
    color: var(--muted);
    font-size: 0.76rem;
    padding: 0.38rem 0.72rem;
}
.status-pill strong { color: var(--ink); }
.welcome {
    border: 1px solid #e3e7f2;
    border-radius: 18px;
    background: white;
    padding: 1.4rem 1.5rem;
    margin: 0.4rem 0 1.1rem;
    box-shadow: 0 12px 35px rgba(32, 45, 80, 0.05);
}
.welcome h3 { margin: 0 0 0.35rem; }
.welcome p { color: var(--muted); margin: 0; }
.section-label {
    color: #8a96aa;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin: 1.1rem 0 0.55rem;
}
.source-card {
    border: 1px solid #dfe5f0;
    border-left: 3px solid var(--purple);
    border-radius: 12px;
    background: #fbfcff;
    padding: 0.7rem 0.85rem;
    margin: 0.4rem 0;
}
.source-card .source-name { color: var(--ink); font-weight: 600; font-size: 0.84rem; }
.source-card .source-page { color: var(--purple); font-size: 0.76rem; margin-top: 0.15rem; }
.mode-note {
    border-radius: 10px;
    background: #111b2d;
    border: 1px solid #26344c;
    color: #9eacc5;
    font-size: 0.76rem;
    line-height: 1.45;
    padding: 0.7rem 0.75rem;
}
.doc-row {
    border-bottom: 1px solid #253149;
    padding: 0.55rem 0;
}
.doc-name { color: #e7edf8; font-size: 0.78rem; overflow-wrap: anywhere; }
.doc-meta { color: #8090ac; font-size: 0.68rem; margin-top: 0.15rem; }
.ready { color: #6ee7b7; }
.processing { color: #f8c56d; }
.failed { color: #ff8c9b; }
.stChatMessage {
    border-radius: 16px;
    border: 1px solid var(--line);
    box-shadow: 0 8px 22px rgba(32, 45, 80, 0.035);
}
.stChatMessage [data-testid="stMarkdownContainer"] { line-height: 1.65; }
.stChatInputContainer {
    border-top: 1px solid transparent;
}
div[data-testid="stExpander"] {
    border: 1px solid var(--line);
    border-radius: 12px;
    background: white;
}
@media (max-width: 900px) {
    .block-container { padding: 1.25rem 1rem 3rem; }
    .hero h1 { font-size: 2.15rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


def api_get(path: str, timeout: int = 8) -> Optional[requests.Response]:
    try:
        response = requests.get(f"{API_URL}{path}", timeout=timeout)
        response.raise_for_status()
        return response
    except requests.RequestException:
        return None


def render_sources(citations: List[Dict[str, Any]]) -> None:
    if not citations:
        return
    st.markdown('<div class="section-label">Evidence</div>', unsafe_allow_html=True)
    for citation in citations:
        document = citation.get("document") or citation.get("doc") or "Unknown document"
        page = citation.get("page", "?")
        st.markdown(
            f'<div class="source-card"><div class="source-name">◈ {document}</div>'
            f'<div class="source-page">Page {page} · Retrieved evidence</div></div>',
            unsafe_allow_html=True,
        )


def render_steps(steps: List[Dict[str, Any]]) -> None:
    if not steps:
        return
    with st.expander(f"Agent trace · {len(steps)} step{'s' if len(steps) != 1 else ''}", expanded=False):
        for step in steps:
            number = step.get("step", "—")
            thought = step.get("thought") or "Working with retrieved evidence."
            st.markdown(f"**Step {number}**  \n{thought}")
            action = step.get("action")
            if action and action != "Final Answer":
                st.code(
                    f"Action: {action}\nArgs: {step.get('action_args', {})}",
                    language="json",
                )
            observation = step.get("observation")
            if observation:
                st.caption(str(observation)[:700])


def render_usage(usage: Dict[str, Any]) -> None:
    if not usage:
        return
    st.caption(
        f"Usage · {usage.get('prompt_tokens', 0):,} input / "
        f"{usage.get('completion_tokens', 0):,} output tokens · "
        f"${usage.get('usd', 0.0):.5f}"
    )


def render_document_library(documents: List[Dict[str, Any]]) -> None:
    ready = sum(document.get("status") == "ready" for document in documents)
    st.markdown(
        f'<div class="section-label">Document library · {ready}/{len(documents)} ready</div>',
        unsafe_allow_html=True,
    )
    if not documents:
        st.caption("No reports indexed yet.")
        return
    for document in documents:
        status = document.get("status", "unknown")
        status_class = status if status in {"ready", "processing", "failed"} else "processing"
        icon = "●" if status == "ready" else "○"
        st.markdown(
            f'<div class="doc-row"><div class="doc-name"><span class="{status_class}">'
            f'{icon}</span>&nbsp; {document.get("filename", "Untitled")}</div>'
            f'<div class="doc-meta">{document.get("page_count", "—")} pages · {status}</div></div>',
            unsafe_allow_html=True,
        )


def submit_question(question: str) -> None:
    """Run the existing chat contract and append the result to session state."""
    question = question.strip()
    if not question:
        return
    st.session_state.messages.append({"role": "user", "content": question})
    payload = {
        "question": question,
        "conversation_id": st.session_state.conversation_id,
        "mode": st.session_state.mode,
    }
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json=payload,
            headers={"X-User-Id": USER_ID},
            timeout=120,
        )
        if response.status_code == 200:
            data = response.json()
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": data.get("answer", ""),
                    "citations": data.get("citations", []),
                    "agent_steps": data.get("agent_steps", []),
                    "usage": data.get("usage", {}),
                    "message_id": data.get("message_id"),
                }
            )
        else:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"API error ({response.status_code}): {response.text}",
                }
            )
    except requests.RequestException as exc:
        st.session_state.messages.append(
            {"role": "assistant", "content": f"Unable to reach the research service: {exc}"}
        )


if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = f"conv_{int(time.time())}"
if "mode" not in st.session_state:
    st.session_state.mode = "agent"


with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark">◈</div><div>'
        '<div class="brand-title">Research Desk</div>'
        '<div class="brand-subtitle">Annual report intelligence</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-label">Research mode</div>', unsafe_allow_html=True)
    mode = st.radio(
        "Research mode",
        ["agent", "pipeline"],
        format_func=lambda value: (
            "Agent · multi-step research" if value == "agent" else "Pipeline · fast baseline"
        ),
        index=0 if st.session_state.mode == "agent" else 1,
        label_visibility="collapsed",
    )
    st.session_state.mode = mode
    st.markdown(
        (
            '<div class="mode-note"><strong>Agent mode</strong> plans searches and validates '
            "evidence step by step."
            if mode == "agent"
            else '<div class="mode-note"><strong>Pipeline mode</strong> runs one retrieval '
            "pass for a faster baseline."
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">Add to library</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload a PDF report",
        type=["pdf"],
        label_visibility="collapsed",
        help="The report is processed in the background after upload.",
    )
    if uploaded_file is not None and st.button("Upload and index report", use_container_width=True):
        with st.spinner("Sending report to the ingestion queue..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                response = requests.post(f"{API_URL}/documents/upload", files=files, timeout=30)
                if response.status_code == 200:
                    st.success("Upload accepted. Processing has started.")
                    st.rerun()
                else:
                    st.error(f"Upload failed ({response.status_code}).")
            except requests.RequestException as exc:
                st.error(f"Upload could not reach the API: {exc}")

    documents_response = api_get("/documents")
    documents = documents_response.json() if documents_response is not None else []
    if documents_response is None:
        st.error("Research API offline")
        st.caption("Start the FastAPI service on port 8000 to reconnect.")
    else:
        render_document_library(documents)

    st.markdown("---")
    if st.button("Rebuild vector index", use_container_width=True):
        try:
            response = requests.post(f"{API_URL}/documents/reindex", timeout=10)
            if response.status_code == 200:
                st.toast("Vector index rebuild started.")
            else:
                st.error(f"Re-indexing failed ({response.status_code}).")
        except requests.RequestException as exc:
            st.error(f"Re-indexing failed: {exc}")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_id = f"conv_{int(time.time())}"
        st.rerun()


health_response = api_get("/health")
health = health_response.json() if health_response is not None else {}
st.markdown(
    '<div class="hero"><div class="eyebrow">AI research assistant</div>'
    "<h1>Ask better questions of your reports.</h1>"
    "<p>Search 14 annual reports with grounded answers, transparent retrieval steps, "
    "and evidence you can trace back to the original page.</p></div>",
    unsafe_allow_html=True,
)
if health:
    st.markdown(
        f'<div class="status-row"><div class="status-pill">● <strong>Service ready</strong></div>'
        f'<div class="status-pill">Reports <strong>{health.get("ready_documents", 0)}</strong> ready</div>'
        f'<div class="status-pill">Mode <strong>{mode.title()}</strong></div></div>',
        unsafe_allow_html=True,
    )

if not st.session_state.messages:
    st.markdown(
        '<div class="welcome"><h3>Start a research thread</h3>'
        "<p>Ask for a figure, a comparison, or a specific passage. Every answer keeps "
        "its document and page context.</p></div>",
        unsafe_allow_html=True,
    )
    examples = [
        "What was the total power generation capacity of RWE AG?",
        "Compare operating performance across the annual reports.",
        "Find the key risk factors discussed in the reports.",
    ]
    cols = st.columns(3)
    for column, example in zip(cols, examples):
        with column:
            if st.button(example, use_container_width=True):
                with st.spinner("Researching the report corpus..."):
                    submit_question(example)
                st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_steps(message.get("agent_steps", []))
            render_sources(message.get("citations", []))
            render_usage(message.get("usage", {}))

if prompt := st.chat_input("Ask about the annual reports..."):
    with st.spinner(f"Researching with {st.session_state.mode} mode..."):
        submit_question(prompt)
    st.rerun()
