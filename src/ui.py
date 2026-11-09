"""Premium Streamlit workspace for the Agentic RAG research assistant.

This module is presentation-only. It uses the existing FastAPI contracts for
chat, conversations, documents, ingestion, citations, and feedback.
"""

import html
import time
from urllib.parse import quote
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st


API_URL = "http://localhost:8000"
USER_ID = "streamlit_user"

st.set_page_config(
    page_title="Research Desk",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
  --sidebar: #101725;
  --sidebar-2: #172235;
  --ink: #172033;
  --muted: #748198;
  --line: #e7ebf2;
  --canvas: #f7f8fb;
  --panel: #ffffff;
  --purple: #6658d9;
  --purple-soft: #eeecff;
  --blue: #3878ed;
  --green: #159570;
  --amber: #bd7d19;
  --red: #c84b5f;
}

.stApp {
  background: var(--canvas);
  color: var(--ink);
  font-family: "DM Sans", sans-serif;
}
.block-container {
  max-width: 1540px;
  padding: 0.9rem 2rem 3.5rem;
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
  color: var(--ink);
  font-family: "Space Grotesk", sans-serif;
  letter-spacing: -0.025em;
}
button, input, textarea { font-family: "DM Sans", sans-serif; }

/* Refined dark navigation rail */
[data-testid="stSidebar"] {
  background: var(--sidebar);
  border-right: 1px solid #202c41;
}
[data-testid="stSidebar"] * { color: #dce5f5; }
[data-testid="stSidebar"] .stCaption { color: #8998b0; }
[data-testid="stSidebar"] .stButton button {
  background: transparent;
  border: 1px solid #2b3a54;
  color: #dce5f5;
  min-height: 2.25rem;
}
[data-testid="stSidebar"] .stButton button:hover {
  background: var(--sidebar-2);
  border-color: #6877ff;
  color: white;
}
[data-testid="stSidebar"] [data-baseweb="radio"] label {
  border: 1px solid #2b3a54;
  border-radius: 9px;
  background: #141e2e;
  padding: 0.52rem 0.65rem;
}
[data-testid="stSidebar"] [data-baseweb="radio"] label:has(input:checked) {
  border-color: #716cff;
  background: #29284d;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
  border: 1px dashed #3b4b67;
  border-radius: 10px;
  background: #141e2e;
}

.brand {
  display: flex;
  align-items: center;
  gap: 0.65rem;
  padding: 0.15rem 0 1.15rem;
}
.brand-mark {
  display: grid;
  place-items: center;
  width: 34px;
  height: 34px;
  border-radius: 10px;
  background: #6960e7;
  color: white;
  font: 700 1.05rem "Space Grotesk", sans-serif;
}
.brand-title { color: white; font: 700 1rem "Space Grotesk", sans-serif; }
.brand-subtitle { color: #8d9bb2; font-size: 0.68rem; margin-top: 0.12rem; }
.rail-label, .context-label {
  color: #8998b0;
  font-size: 0.66rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  margin: 1rem 0 0.48rem;
}
.conversation-empty {
  color: #8796ad;
  font-size: 0.76rem;
  padding: 0.5rem 0;
}
.conversation-item {
  border-left: 2px solid #5564e7;
  color: #f0f3fb;
  font-size: 0.78rem;
  overflow: hidden;
  padding: 0.38rem 0.55rem;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.conversation-meta { color: #8492a8; font-size: 0.66rem; padding-left: 0.55rem; }
.doc-row {
  border-bottom: 1px solid #27344a;
  padding: 0.48rem 0;
}
.doc-name {
  color: #edf2fb;
  font-size: 0.75rem;
  overflow-wrap: anywhere;
}
.doc-meta { color: #8190a8; font-size: 0.65rem; margin-top: 0.13rem; }
.ready { color: #67ddb2; }
.processing { color: #f1c36d; }
.failed { color: #ff8497; }

/* Workspace header and status */
.workspace-header {
  align-items: center;
  border-bottom: 1px solid var(--line);
  display: flex;
  justify-content: space-between;
  margin-bottom: 1.15rem;
  padding: 0.15rem 0 0.9rem;
}
.workspace-kicker {
  color: var(--muted);
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.09em;
  text-transform: uppercase;
}
.workspace-title { font: 700 1.2rem "Space Grotesk", sans-serif; margin-top: 0.18rem; }
.health {
  align-items: center;
  color: var(--green);
  display: flex;
  font-size: 0.76rem;
  gap: 0.38rem;
}
.health-dot { background: var(--green); border-radius: 50%; height: 7px; width: 7px; }
.thread-intro {
  color: var(--muted);
  font-size: 0.86rem;
  line-height: 1.55;
  margin: 0.2rem 0 1.1rem;
}
.mode-chip {
  background: var(--purple-soft);
  border: 1px solid #dad6ff;
  border-radius: 999px;
  color: #4f45ae;
  display: inline-block;
  font-size: 0.68rem;
  font-weight: 700;
  padding: 0.28rem 0.58rem;
}

/* Message canvas */
.stChatMessage {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 14px;
  box-shadow: 0 5px 16px rgba(28, 41, 74, 0.035);
  margin: 0.65rem 0;
}
.stChatMessage [data-testid="stMarkdownContainer"] { line-height: 1.68; }
.user-message {
  color: #334159;
  font-size: 0.86rem;
  line-height: 1.55;
}
.answer-kicker {
  color: var(--purple);
  font-size: 0.66rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  margin-bottom: 0.38rem;
  text-transform: uppercase;
}
.refusal-kicker { color: var(--amber); }
.evidence-count {
  color: var(--muted);
  font-size: 0.7rem;
  margin-top: 0.7rem;
}
.context-panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 1rem;
}
.context-panel h3 { font-size: 0.96rem; margin: 0 0 0.3rem; }
.context-copy { color: var(--muted); font-size: 0.75rem; line-height: 1.5; }
.source-card {
  background: #fbfcff;
  border: 1px solid #dfe5f0;
  border-left: 3px solid var(--purple);
  border-radius: 10px;
  margin: 0.5rem 0;
  padding: 0.65rem 0.72rem;
}
.source-name { color: var(--ink); font-size: 0.77rem; font-weight: 700; overflow-wrap: anywhere; }
.source-page { color: var(--purple); font-size: 0.69rem; margin-top: 0.18rem; }
.source-selected { background: #f1efff; border-color: #cfcaff; }
.timeline {
  border-left: 1px solid #d9deea;
  margin: 0.6rem 0 0.4rem 0.28rem;
  padding-left: 0.9rem;
}
.timeline-step { margin: 0 0 0.9rem; position: relative; }
.timeline-step:before {
  background: var(--purple);
  border: 3px solid white;
  border-radius: 50%;
  content: "";
  height: 7px;
  left: -1.16rem;
  position: absolute;
  top: 0.28rem;
  width: 7px;
}
.timeline-title { color: var(--ink); font-size: 0.73rem; font-weight: 700; }
.timeline-copy { color: var(--muted); font-size: 0.69rem; line-height: 1.45; margin-top: 0.16rem; }
.metric {
  border-top: 1px solid var(--line);
  display: flex;
  justify-content: space-between;
  padding: 0.55rem 0;
}
.metric-label { color: var(--muted); font-size: 0.7rem; }
.metric-value { color: var(--ink); font-size: 0.73rem; font-weight: 700; }
.empty-workspace {
  border: 1px dashed #cfd6e4;
  border-radius: 14px;
  color: var(--muted);
  padding: 2.2rem 1.3rem;
  text-align: center;
}
.empty-workspace strong { color: var(--ink); display: block; font-size: 1rem; margin-bottom: 0.3rem; }
div[data-testid="stExpander"] {
  background: white;
  border: 1px solid var(--line);
  border-radius: 10px;
}
@media (max-width: 1050px) {
  .block-container { padding: 0.8rem 1rem 3rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


def api_json(
    method: str,
    path: str,
    *,
    timeout: int = 10,
    **kwargs: Any,
) -> Tuple[Optional[Any], Optional[str]]:
    try:
        response = requests.request(method, f"{API_URL}{path}", timeout=timeout, **kwargs)
        if response.status_code >= 400:
            return None, f"{response.status_code}: {response.text}"
        return response.json(), None
    except requests.RequestException as exc:
        return None, str(exc)


def new_conversation() -> None:
    st.session_state.conversation_id = f"conv_{int(time.time())}"
    st.session_state.messages = []
    st.session_state.conversation_title = "New research thread"
    st.session_state.selected_source = None


def load_conversation(conversation_id: str, title: str) -> None:
    data, error = api_json("GET", f"/conversations/{conversation_id}", timeout=10)
    if error:
        st.session_state.connection_error = f"Conversation could not be loaded: {error}"
        return
    st.session_state.conversation_id = conversation_id
    st.session_state.conversation_title = title
    st.session_state.messages = data.get("messages", [])
    st.session_state.selected_source = None
    st.session_state.connection_error = None


def get_conversations() -> Tuple[List[Dict[str, Any]], Optional[str]]:
    data, error = api_json("GET", f"/conversations?user_id={USER_ID}", timeout=8)
    return (data if isinstance(data, list) else []), error


def is_refusal(text: str) -> bool:
    lowered = text.lower()
    markers = (
        "cannot answer",
        "can't answer",
        "not found in the",
        "no relevant",
        "insufficient evidence",
        "does not contain",
    )
    return any(marker in lowered for marker in markers)


def render_sources(citations: List[Dict[str, Any]], prefix: str) -> None:
    if not citations:
        return
    st.markdown(
        f'<div class="evidence-count">{len(citations)} grounded source'
        f'{"s" if len(citations) != 1 else ""}</div>',
        unsafe_allow_html=True,
    )
    source_columns = st.columns(min(len(citations), 3))
    for index, citation in enumerate(citations):
        document = str(citation.get("document") or citation.get("doc") or "Unknown document")
        page = citation.get("page", "?")
        with source_columns[index % len(source_columns)]:
            if st.button(
                f"◈ {document} · p.{page}",
                key=f"{prefix}-source-{index}",
                use_container_width=True,
                help="Inspect this citation in the research context panel.",
            ):
                st.session_state.selected_source = {
                    "document": document,
                    "page": page,
                }
                st.rerun()


def render_timeline(steps: List[Dict[str, Any]], expanded: bool = False) -> None:
    if not steps:
        return
    with st.expander(
        f"Research activity · {len(steps)} step{'s' if len(steps) != 1 else ''}",
        expanded=expanded,
    ):
        timeline_html = ['<div class="timeline">']
        for step in steps:
            action = html.escape(str(step.get("action") or "Synthesis"))
            thought = html.escape(
                str(step.get("thought") or "Working with retrieved evidence.")
            )
            timeline_html.append(
                f'<div class="timeline-step"><div class="timeline-title">{action}</div>'
                f'<div class="timeline-copy">{thought}</div></div>'
            )
        timeline_html.append("</div>")
        st.markdown("".join(timeline_html), unsafe_allow_html=True)
        for index, step in enumerate(steps):
            action = step.get("action")
            observation = step.get("observation")
            if action and action != "Final Answer":
                with st.expander(f"Tool details · step {step.get('step', index + 1)}"):
                    st.code(
                        f"Action: {action}\nArgs: {step.get('action_args', {})}",
                        language="json",
                    )
                    if step.get("usage"):
                        st.json(step["usage"])
                    if observation:
                        st.caption(str(observation)[:900])


def render_usage(usage: Dict[str, Any]) -> None:
    if not usage:
        return
    st.markdown(
        '<div class="metric"><span class="metric-label">Tokens</span>'
        f'<span class="metric-value">{usage.get("prompt_tokens", 0):,} in · '
        f'{usage.get("completion_tokens", 0):,} out</span></div>'
        '<div class="metric"><span class="metric-label">Estimated cost</span>'
        f'<span class="metric-value">${usage.get("usd", 0.0):.5f}</span></div>',
        unsafe_allow_html=True,
    )


def render_feedback(message_id: Optional[str], prefix: str) -> None:
    if not message_id:
        return
    left, right = st.columns([1, 1])
    with left:
        if st.button("Helpful", key=f"{prefix}-up", help="Send positive feedback"):
            api_json(
                "POST",
                "/feedback",
                json={"message_id": message_id, "score": 1},
                timeout=8,
            )
            st.toast("Thanks for the feedback.")
    with right:
        if st.button("Needs work", key=f"{prefix}-down", help="Send negative feedback"):
            api_json(
                "POST",
                "/feedback",
                json={"message_id": message_id, "score": -1},
                timeout=8,
            )
            st.toast("Feedback recorded.")


def render_document_library(documents: List[Dict[str, Any]]) -> None:
    ready = sum(document.get("status") == "ready" for document in documents)
    st.markdown(
        f'<div class="rail-label">Document library · {ready}/{len(documents)} ready</div>',
        unsafe_allow_html=True,
    )
    if not documents:
        st.markdown(
            '<div class="conversation-empty">Your indexed reports will appear here.</div>',
            unsafe_allow_html=True,
        )
        return
    for document in documents:
        status = document.get("status", "unknown")
        status_class = status if status in {"ready", "processing", "failed"} else "processing"
        icon = "●" if status == "ready" else "○"
        filename = html.escape(str(document.get("filename", "Untitled")))
        st.markdown(
            f'<div class="doc-row"><div class="doc-name"><span class="{status_class}">'
            f'{icon}</span>&nbsp; {filename}</div><div class="doc-meta">'
            f'{document.get("page_count", "—")} pages · {html.escape(status)}</div></div>',
            unsafe_allow_html=True,
        )


def submit_question(question: str) -> None:
    question = question.strip()
    if not question:
        return
    st.session_state.messages.append({"role": "user", "content": question})
    payload = {
        "question": question,
        "conversation_id": st.session_state.conversation_id,
        "mode": st.session_state.mode,
    }
    data, error = api_json(
        "POST",
        "/chat",
        timeout=120,
        json=payload,
        headers={"X-User-Id": USER_ID},
    )
    if error:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"The research service could not complete this request. {error}",
                "error": True,
            }
        )
        return
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
    st.session_state.conversation_id = data.get(
        "conversation_id", st.session_state.conversation_id
    )
    st.session_state.conversation_title = question[:52]


if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = f"conv_{int(time.time())}"
if "conversation_title" not in st.session_state:
    st.session_state.conversation_title = "New research thread"
if "mode" not in st.session_state:
    st.session_state.mode = "agent"
if "selected_source" not in st.session_state:
    st.session_state.selected_source = None
if "connection_error" not in st.session_state:
    st.session_state.connection_error = None


documents, documents_error = api_json("GET", "/documents", timeout=8)
documents = documents if isinstance(documents, list) else []
conversations, conversations_error = get_conversations()
health, health_error = api_json("GET", "/health", timeout=8)
health = health if isinstance(health, dict) else {}


with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark">◈</div><div>'
        '<div class="brand-title">Research Desk</div>'
        '<div class="brand-subtitle">Annual report intelligence</div></div></div>',
        unsafe_allow_html=True,
    )
    if st.button("＋  New research thread", use_container_width=True):
        new_conversation()
        st.rerun()
    if st.button("Clear active thread", use_container_width=True):
        st.session_state.messages = []
        st.session_state.selected_source = None
        st.session_state.conversation_title = "New research thread"
        st.rerun()

    st.markdown('<div class="rail-label">Conversations</div>', unsafe_allow_html=True)
    if conversations_error:
        st.markdown(
            '<div class="conversation-empty">Conversation history is unavailable.</div>',
            unsafe_allow_html=True,
        )
    elif not conversations:
        st.markdown(
            '<div class="conversation-empty">No saved threads yet.</div>',
            unsafe_allow_html=True,
        )
    else:
        for conversation in conversations[:12]:
            conversation_id = str(conversation.get("id", ""))
            title = str(conversation.get("title") or "Untitled thread")
            if st.button(
                title[:38],
                key=f"conversation-{conversation_id}",
                use_container_width=True,
                help=title,
            ):
                load_conversation(conversation_id, title)
                st.rerun()

    st.markdown('<div class="rail-label">Research mode</div>', unsafe_allow_html=True)
    selected_mode = st.radio(
        "Research mode",
        ["agent", "pipeline"],
        index=0 if st.session_state.mode == "agent" else 1,
        format_func=lambda value: (
            "Agent · multi-step" if value == "agent" else "Pipeline · single pass"
        ),
        label_visibility="collapsed",
    )
    st.session_state.mode = selected_mode

    st.markdown('<div class="rail-label">Add to library</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload a PDF report",
        type=["pdf"],
        label_visibility="collapsed",
        help="PDFs are accepted immediately and processed in the background.",
    )
    if uploaded_file is not None and st.button(
        "Upload and process", use_container_width=True
    ):
        with st.spinner("Uploading report…"):
            try:
                response = requests.post(
                    f"{API_URL}/documents/upload",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf",
                        )
                    },
                    timeout=30,
                )
                if response.status_code == 200:
                    st.success("Accepted · processing in background")
                    st.rerun()
                else:
                    st.error(f"Upload failed ({response.status_code}).")
            except requests.RequestException as exc:
                st.error(f"Upload could not reach the API: {exc}")

    if documents_error:
        st.error("Document library unavailable.")
    else:
        render_document_library(documents)

    if st.button("Refresh library", use_container_width=True):
        st.rerun()
    if st.button("Rebuild vector index", use_container_width=True):
        data, error = api_json("POST", "/documents/reindex", timeout=10)
        if error:
            st.error(f"Re-indexing failed: {error}")
        else:
            st.toast(data.get("message", "Re-indexing started."))


st.markdown(
    '<div class="workspace-header"><div><div class="workspace-kicker">Research workspace</div>'
    f'<div class="workspace-title">{html.escape(st.session_state.conversation_title)}</div></div>'
    + (
        '<div class="health"><span class="health-dot"></span>Service ready</div>'
        if not health_error
        else '<div class="health" style="color:#c84b5f">Service unavailable</div>'
    )
    + "</div>",
    unsafe_allow_html=True,
)

main_column, context_column = st.columns([1.62, 0.78], gap="large")

with main_column:
    mode_label = "Agent mode" if st.session_state.mode == "agent" else "Pipeline mode"
    st.markdown(
        f'<span class="mode-chip">{mode_label}</span>'
        '<div class="thread-intro">Ask about the indexed annual reports. Answers are '
        "grounded in retrieved PDF passages and retain their page citations.</div>",
        unsafe_allow_html=True,
    )

    if st.session_state.connection_error:
        st.error(st.session_state.connection_error)

    if not st.session_state.messages:
        st.markdown(
            '<div class="empty-workspace"><strong>Start a new research thread</strong>'
            "Ask for a figure, a comparison, or a specific passage. The research context "
            "panel will collect sources and activity as you work.</div>",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="rail-label">Suggested questions</div>', unsafe_allow_html=True)
        suggestions = [
            "What was the total power generation capacity of RWE AG?",
            "Compare operating performance across the annual reports.",
            "Find the key risk factors discussed in the reports.",
        ]
        suggestion_columns = st.columns(3)
        for index, suggestion in enumerate(suggestions):
            with suggestion_columns[index]:
                if st.button(
                    suggestion,
                    key=f"suggestion-{index}",
                    use_container_width=True,
                ):
                    with st.spinner(f"Researching with {mode_label.lower()}…"):
                        submit_question(suggestion)
                    st.rerun()

    for index, message in enumerate(st.session_state.messages):
        role = message.get("role", "assistant")
        with st.chat_message(role):
            if role == "user":
                st.markdown(
                    f'<div class="user-message">{html.escape(str(message.get("content", "")))}</div>',
                    unsafe_allow_html=True,
                )
            else:
                answer = str(message.get("content", ""))
                refusal = is_refusal(answer)
                kicker_class = "refusal-kicker" if refusal else ""
                kicker_text = "No matching evidence" if refusal else "Grounded answer"
                st.markdown(
                    f'<div class="answer-kicker {kicker_class}">{kicker_text}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(answer)
                if message.get("error"):
                    st.warning("This response did not complete successfully.")
                render_sources(message.get("citations", []), f"message-{index}")
                render_timeline(message.get("agent_steps", []))
                render_feedback(message.get("message_id"), f"feedback-{index}")

if prompt := st.chat_input(
    "Ask a question about the reports…",
    disabled=bool(health_error),
):
    with st.spinner(f"Researching with {st.session_state.mode} mode…"):
        submit_question(prompt)
    st.rerun()


with context_column:
    st.markdown(
        '<div class="context-panel"><h3>Research context</h3>'
        '<div class="context-copy">Sources and activity from the active thread. '
        "Nothing is invented here—only backend-returned evidence is shown.</div>",
        unsafe_allow_html=True,
    )
    all_citations: List[Dict[str, Any]] = []
    all_steps: List[Dict[str, Any]] = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "usd": 0.0}
    for message in st.session_state.messages:
        if message.get("role") != "assistant":
            continue
        all_citations.extend(message.get("citations", []))
        all_steps.extend(message.get("agent_steps", []))
        usage = message.get("usage", {})
        total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
        total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
        total_usage["usd"] += usage.get("usd", 0.0)

    selected_source = st.session_state.selected_source
    if selected_source:
        raw_document = str(selected_source.get("document", "Unknown document"))
        raw_page = selected_source.get("page", "?")
        document = html.escape(raw_document)
        page = html.escape(str(raw_page))
        st.markdown(
            f'<div class="context-label">Selected source</div><div class="source-card '
            f'source-selected"><div class="source-name">◈ {document}</div>'
            f'<div class="source-page">Page {page} · Citation detail</div></div>',
            unsafe_allow_html=True,
        )
        if isinstance(raw_page, int):
            st.link_button(
                "Open original PDF at this page",
                f"{API_URL}/documents/{quote(raw_document, safe='')}/file#page={raw_page}",
                use_container_width=True,
            )
        if isinstance(raw_page, int):
            passage, passage_error = api_json(
                "GET",
                f"/documents/{quote(raw_document, safe='')}/pages/{raw_page}",
                timeout=10,
            )
            if passage_error:
                st.error(f"Could not load the cited passage: {passage_error}")
            else:
                st.markdown("**Extracted passage**")
                for text in passage.get("passages", []):
                    st.code(text, language="text")

    st.markdown('<div class="context-label">Sources in thread</div>', unsafe_allow_html=True)
    unique_sources = []
    seen_sources = set()
    for citation in all_citations:
        key = (
            citation.get("document") or citation.get("doc"),
            citation.get("page"),
        )
        if key not in seen_sources:
            seen_sources.add(key)
            unique_sources.append(citation)
    if unique_sources:
        for citation in unique_sources[:10]:
            document = html.escape(
                str(citation.get("document") or citation.get("doc") or "Unknown document")
            )
            page = citation.get("page", "?")
            st.markdown(
                f'<div class="source-card"><div class="source-name">◈ {document}</div>'
                f'<div class="source-page">Page {page}</div></div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="context-copy">Citations will appear after the first answer.</div>',
            unsafe_allow_html=True,
        )

    if all_steps:
        st.markdown('<div class="context-label">Activity summary</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric"><span class="metric-label">Research steps</span>'
            f'<span class="metric-value">{len(all_steps)}</span></div>',
            unsafe_allow_html=True,
        )
    if any(total_usage.values()):
        st.markdown('<div class="context-label">Thread usage</div>', unsafe_allow_html=True)
        render_usage(total_usage)
    st.markdown("</div>", unsafe_allow_html=True)
