"""Streamlit frontend for the Agentic RAG research workspace."""

from __future__ import annotations

import html
import json
import time
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
from urllib.parse import quote

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
:root {
  --ink: #172033;
  --muted: #5f6b7d;
  --subtle: #f4f6fa;
  --line: #dce3ed;
  --panel: #ffffff;
  --accent: #2167d5;
  --accent-dark: #174ca2;
  --accent-soft: #eaf2ff;
  --success: #147a55;
  --warning: #986316;
  --danger: #b3344a;
}
html, body, [class*="css"] { color: var(--ink); }
.stApp { background: #f4f6fa; color: var(--ink); }
.stApp, .stApp * { box-sizing: border-box; }
.block-container { max-width: 1780px; padding: 1.15rem 1.4rem 2.5rem; }
[data-testid="stDecoration"] {
  display: none !important;
}

/* Streamlit's sidebar has both a root and an inner scrolling container. */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child,
[data-testid="stSidebar"] [data-testid="stSidebarContent"],
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
  background: #ffffff !important;
  color: var(--ink) !important;
}
[data-testid="stSidebar"] > div:first-child {
  border-right: 1px solid var(--line) !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
  color: var(--ink);
}
[data-testid="stSidebar"] [data-testid="stButton"] > button,
[data-testid="stSidebar"] [data-testid="stDownloadButton"] > button {
  min-height: 2.35rem;
  height: auto !important;
  overflow: visible !important;
  white-space: normal !important;
  border: 1px solid var(--line) !important;
  border-radius: .55rem !important;
  background: #ffffff !important;
  color: var(--ink) !important;
  font-weight: 650 !important;
  box-shadow: none !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] > button:hover,
[data-testid="stSidebar"] [data-testid="stDownloadButton"] > button:hover {
  border-color: var(--accent) !important;
  background: var(--accent-soft) !important;
  color: var(--accent-dark) !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] > button:focus-visible,
[data-testid="stSidebar"] [data-testid="stDownloadButton"] > button:focus-visible,
[data-testid="stButton"] > button:focus-visible,
[data-testid="stChatInput"] textarea:focus {
  outline: 3px solid #b9d1ff !important;
  outline-offset: 1px;
}
[data-testid="stSidebar"] [data-testid="stButton"] > button:disabled,
[data-testid="stButton"] > button:disabled {
  background: #eef1f5 !important;
  border-color: #d8dee8 !important;
  color: #8a94a5 !important;
  opacity: 1 !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] > button p {
  white-space: normal !important;
  overflow-wrap: anywhere !important;
  text-overflow: clip !important;
  line-height: 1.3 !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
  border: 1px solid #c9d8ee !important;
  border-radius: .7rem !important;
  background: #f7faff !important;
  padding: .55rem !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] section,
[data-testid="stSidebar"] [data-testid="stFileUploader"] section > div {
  background: transparent !important;
  border: 0 !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] small,
[data-testid="stSidebar"] [data-testid="stFileUploader"] span {
  color: var(--muted) !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
  background: var(--accent) !important;
  color: #ffffff !important;
  border: 0 !important;
  border-radius: .45rem !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
  color: var(--ink) !important;
  background: #f7f9fc !important;
  border: 1px solid var(--line) !important;
  border-radius: .5rem !important;
  padding: .55rem .65rem !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
  background: var(--accent-soft) !important;
  border-color: #9dbcf0 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {
  gap: .45rem !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {
  color: var(--ink) !important;
}

[data-testid="stButton"] > button {
  min-height: 2.45rem;
  height: auto !important;
  overflow: visible !important;
  white-space: normal !important;
  border: 1px solid var(--line) !important;
  border-radius: .6rem !important;
  background: #ffffff !important;
  color: var(--ink) !important;
  font-weight: 650 !important;
  box-shadow: 0 1px 2px rgba(20, 34, 58, .04) !important;
}
[data-testid="stButton"] > button p {
  white-space: normal !important;
  overflow-wrap: anywhere !important;
  text-overflow: clip !important;
  line-height: 1.3 !important;
}
[data-testid="stButton"] > button:hover {
  background: var(--accent-soft) !important;
  border-color: #8eb0eb !important;
  color: var(--accent-dark) !important;
}
[data-testid="stChatInput"] {
  border: 1px solid #b8c5d8 !important;
  border-radius: .8rem !important;
  background: #ffffff !important;
  box-shadow: 0 5px 18px rgba(24, 47, 86, .08) !important;
}
[data-testid="stChatInput"] textarea {
  color: var(--ink) !important;
  background: #ffffff !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #738096 !important; }
[data-testid="stChatInput"] button {
  background: var(--accent) !important;
  color: #ffffff !important;
  border-radius: .5rem !important;
}
[data-testid="stChatInput"] button:hover { background: var(--accent-dark) !important; }
[data-testid="stExpander"] {
  border: 1px solid var(--line) !important;
  border-radius: .65rem !important;
  background: #ffffff !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary span { color: var(--ink) !important; }
[data-testid="stAlert"] {
  border: 1px solid var(--line) !important;
  color: var(--ink) !important;
}
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] [role="combobox"] {
  background: #ffffff !important;
  color: var(--ink) !important;
  border-color: #b8c5d8 !important;
}

.brand { display: flex; gap: .7rem; align-items: center; padding: .15rem .1rem 1rem; }
.brand-mark { width: 2.15rem; height: 2.15rem; display: grid; place-items: center; border-radius: .65rem; background: var(--accent); color: #fff !important; font-weight: 800; }
.brand-name { color: var(--ink) !important; font-size: 1.1rem; font-weight: 800; letter-spacing: -.02em; }
.brand-subtitle { color: var(--muted) !important; font-size: .72rem; margin-top: .12rem; }
.section-label { color: #536176 !important; font-size: .66rem; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; margin: 1rem 0 .45rem; }
.upload-card { border: 1px solid #c8d8ef; border-radius: .75rem; background: #f5f9ff; padding: .8rem; margin-bottom: .65rem; }
.upload-card strong { color: #17458f !important; display: block; font-size: .9rem; margin-bottom: .2rem; }
.upload-card span { color: var(--muted) !important; font-size: .74rem; line-height: 1.4; }
.library-summary { display: flex; justify-content: space-between; border: 1px solid var(--line); border-radius: .65rem; padding: .6rem .7rem; background: var(--subtle); font-size: .77rem; }
.library-summary strong { color: var(--success) !important; }
.doc-item { border-bottom: 1px solid #edf0f4; padding: .55rem .05rem; }
.doc-name { color: var(--ink) !important; font-size: .78rem; font-weight: 650; overflow-wrap: anywhere; line-height: 1.35; }
.doc-link, .doc-link:visited { color: var(--accent) !important; text-decoration: none; }
.doc-link:hover, .doc-link:focus-visible { color: var(--accent-dark) !important; text-decoration: underline; }
.doc-meta { color: var(--muted) !important; font-size: .68rem; margin-top: .16rem; }
.status-ready { color: var(--success) !important; } .status-processing, .status-pending { color: var(--warning) !important; } .status-failed { color: var(--danger) !important; }
.workspace-head { display: flex; justify-content: space-between; gap: 1rem; align-items: flex-start; border-bottom: 1px solid var(--line); padding-bottom: .9rem; margin-bottom: 1.05rem; }
.eyebrow { color: var(--muted) !important; font-size: .7rem; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; }
.workspace-title { color: var(--ink) !important; font-size: 1.4rem; font-weight: 800; letter-spacing: -.03em; margin-top: .2rem; overflow-wrap: anywhere; }
.health { color: var(--success) !important; font-size: .78rem; white-space: nowrap; padding-top: .35rem; } .health.offline { color: var(--danger) !important; }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: .9rem; padding: 1rem; box-shadow: 0 5px 18px rgba(31, 45, 75, .04); }
.panel-title { color: var(--ink) !important; font-size: .95rem; font-weight: 800; margin-bottom: .2rem; }
.panel-subtitle { color: var(--muted) !important; font-size: .76rem; line-height: 1.5; }
.mode-badge { display: inline-flex; border: 1px solid #b8d0f7; color: #174c9e !important; background: var(--accent-soft); border-radius: 999px; font-size: .72rem; font-weight: 800; padding: .3rem .6rem; }
.empty-state { border: 1px dashed #b9c8dc; border-radius: .8rem; padding: 2rem 1.2rem; text-align: center; color: var(--muted) !important; background: #fbfcfe; margin-bottom: 1rem; }
.empty-state strong { display: block; color: var(--ink) !important; font-size: 1.08rem; margin-bottom: .4rem; }
.stChatMessage { border: 1px solid var(--line); border-radius: .8rem; background: #ffffff; box-shadow: 0 3px 12px rgba(31, 45, 75, .03); margin: .65rem 0; }
.answer-label { color: var(--accent-dark) !important; font-size: .66rem; font-weight: 850; letter-spacing: .1em; text-transform: uppercase; margin-bottom: .35rem; }
.answer-label.refusal { color: var(--warning) !important; }
.source-chip { display: inline-block; border: 1px solid #c6d6ef; background: #f5f8ff; border-radius: .45rem; color: #174c9e !important; font-size: .72rem; margin: .25rem .25rem 0 0; padding: .28rem .45rem; }
.trace-step { border-left: 2px solid #9dbcf0; padding: 0 0 .8rem .75rem; margin: .2rem 0; }
.trace-dot { color: var(--accent-dark) !important; font-size: .68rem; font-weight: 850; text-transform: uppercase; }
.trace-copy { color: var(--ink) !important; font-size: .74rem; line-height: 1.45; margin-top: .18rem; }
.trace-meta { color: var(--muted) !important; font-size: .66rem; margin-top: .25rem; }
.metric-row { display: flex; justify-content: space-between; border-top: 1px solid #eef1f5; padding: .5rem 0; font-size: .72rem; }
.metric-row span:first-child { color: var(--muted) !important; } .metric-row span:last-child { color: var(--ink) !important; font-weight: 750; }
@media (max-width: 900px) { .block-container { padding: .8rem .7rem 2rem; } .workspace-head { display: block; } .health { padding-top: .6rem; } }
</style>
""",
    unsafe_allow_html=True,
)


def api_json(method: str, path: str, *, timeout: int = 10, **kwargs: Any) -> Tuple[Optional[Any], Optional[str]]:
    try:
        response = requests.request(method, f"{API_URL}{path}", timeout=timeout, **kwargs)
        if response.status_code >= 400:
            return None, f"{response.status_code}: {response.text}"
        return response.json(), None
    except requests.RequestException as exc:
        return None, str(exc)


def reset_thread() -> None:
    st.session_state.conversation_id = f"conv_{int(time.time())}"
    st.session_state.conversation_title = "New research thread"
    st.session_state.messages = []
    st.session_state.selected_source = None


def load_conversation(conversation_id: str, title: str) -> None:
    data, error = api_json("GET", f"/conversations/{quote(conversation_id, safe='')}")
    if error:
        st.session_state.connection_error = f"Conversation could not be loaded: {error}"
        return
    st.session_state.conversation_id = conversation_id
    st.session_state.conversation_title = title
    st.session_state.messages = data.get("messages", [])
    st.session_state.selected_source = None
    st.session_state.connection_error = None


def get_conversations() -> Tuple[List[Dict[str, Any]], Optional[str]]:
    data, error = api_json("GET", f"/conversations?user_id={quote(USER_ID)}", timeout=8)
    return (data if isinstance(data, list) else []), error


def is_refusal(text: str) -> bool:
    markers = ("cannot answer", "can't answer", "not found in the", "no relevant", "insufficient evidence", "does not contain", "not available", "not mentioned")
    return any(marker in text.lower() for marker in markers)


def citation_key(citation: Dict[str, Any]) -> Tuple[str, Any]:
    return str(citation.get("document") or citation.get("doc") or "Unknown document"), citation.get("page", "?")


def render_citations(citations: List[Dict[str, Any]], prefix: str) -> None:
    if not citations:
        return
    st.caption(f"{len(citations)} grounded source{'s' if len(citations) != 1 else ''}")
    columns = st.columns(min(len(citations), 3))
    for index, citation in enumerate(citations):
        document, page = citation_key(citation)
        with columns[index % len(columns)]:
            if st.button(f"◈ {document} · p.{page}", key=f"{prefix}-citation-{index}", use_container_width=True, help="Show the extracted passage and open the original PDF."):
                st.session_state.selected_source = {"document": document, "page": page}
                st.rerun()


def render_trace(steps: List[Dict[str, Any]]) -> None:
    if not steps:
        st.markdown('<div class="panel-subtitle">Agent searches and tool activity will appear here after the first question.</div>', unsafe_allow_html=True)
        return
    st.caption(f"{len(steps)} recorded step{'s' if len(steps) != 1 else ''} · chronological")
    for index, step in enumerate(steps):
        action = str(step.get("action") or "synthesis")
        thought = str(step.get("thought") or step.get("observation") or "Working with report evidence.")
        usage = step.get("usage") or {}
        meta: List[str] = []
        if usage.get("provider") or usage.get("model"):
            meta.append(f"{usage.get('provider', '')} · {usage.get('model', '')}".strip(" ·"))
        if usage.get("latency_seconds") is not None:
            meta.append(f"{usage['latency_seconds']:.2f}s")
        if usage.get("cost_usd") is not None:
            meta.append(f"${usage['cost_usd']:.5f}")
        if usage.get("prompt_tokens") or usage.get("completion_tokens"):
            meta.append(f"{usage.get('prompt_tokens', 0):,} in / {usage.get('completion_tokens', 0):,} out")
        st.markdown(
            f'<div class="trace-step"><div class="trace-dot">{index + 1:02d} · {html.escape(action.replace("_", " "))}</div>'
            f'<div class="trace-copy">{html.escape(thought[:700])}</div><div class="trace-meta">{html.escape(" · ".join(meta))}</div></div>',
            unsafe_allow_html=True,
        )
        if step.get("action_args") or step.get("observation"):
            with st.expander("Inspect tool call and observation", expanded=False):
                if step.get("action_args"):
                    st.json(step["action_args"])
                if step.get("observation"):
                    st.caption(str(step["observation"])[:1500])


def render_usage(messages: Iterable[Dict[str, Any]]) -> None:
    prompt = completion = 0
    cost = 0.0
    has_usage = False
    for message in messages:
        usage = message.get("usage") or {}
        if usage:
            has_usage = True
        prompt += usage.get("prompt_tokens", 0)
        completion += usage.get("completion_tokens", 0)
        cost += usage.get("usd", 0.0)
    if not has_usage:
        return
    st.markdown(
        f'<div class="metric-row"><span>Thread tokens</span><span>{prompt:,} in · {completion:,} out</span></div>'
        f'<div class="metric-row"><span>Estimated cost</span><span>${cost:.5f}</span></div>',
        unsafe_allow_html=True,
    )


def render_selected_source() -> None:
    selected = st.session_state.selected_source
    if not selected:
        st.markdown('<div class="panel-subtitle">Select a citation to inspect its passage and open the source PDF.</div>', unsafe_allow_html=True)
        return
    document = str(selected["document"])
    raw_page = selected["page"]
    try:
        page = int(raw_page)
    except (TypeError, ValueError):
        page = None
    st.markdown(f"**{html.escape(document)} · page {html.escape(str(raw_page))}**")
    if page is not None and page > 0:
        st.link_button("Open original PDF at this page", f"{API_URL}/documents/{quote(document, safe='')}/file#page={page}", use_container_width=True)
        passage, error = api_json("GET", f"/documents/{quote(document, safe='')}/pages/{page}", timeout=10)
        if error:
            st.error(f"Could not load cited passage: {error}")
        else:
            for text in passage.get("passages", []):
                st.code(text, language="text")


def select_document(document_id: str) -> None:
    st.session_state.document_filter = document_id
    st.session_state.document_selector = document_id


def clear_document_filter() -> None:
    st.session_state.document_filter = None
    st.session_state.document_selector = None


def render_library(documents: List[Dict[str, Any]]) -> None:
    ready_documents = [
        document for document in documents
        if document.get("status") == "ready"
    ]

    ready_filenames = [
        str(document.get("filename", "Untitled"))
        for document in ready_documents
    ]

    # Clear selection if the selected document is no longer available.
    if st.session_state.document_filter not in [None] + ready_filenames:
        st.session_state.document_filter = None

    st.markdown(
        f'<div class="library-summary"><span>Document library</span>'
        f'<strong>{len(ready_documents)}/{len(documents)} ready</strong></div>',
        unsafe_allow_html=True,
    )

    if not documents:
        st.info("No reports are indexed yet. Upload a PDF to begin.")
        return

    # Search across all indexed documents.
    all_selected = st.session_state.document_filter is None

    st.button(
        "All indexed reports",
        key="select-all-documents",
        type="primary" if all_selected else "secondary",
        use_container_width=True,
        on_click=clear_document_filter,
        help="Search across all indexed reports.",
    )

    st.caption("SELECT A PDF")

    for document in documents:
        status = str(document.get("status", "unknown"))
        filename = str(document.get("filename", "Untitled"))
        page_count = document.get("page_count", "—")

        if status == "ready":
            is_selected = st.session_state.document_filter == filename

            st.button(
                f"{'✓ ' if is_selected else ''}{filename}",
                key=f"document-{filename}",
                type="primary" if is_selected else "secondary",
                use_container_width=True,
                on_click=select_document,
                args=(filename,),
                help="Select this PDF for your questions.",
            )

            st.caption(f"Ready · {page_count} pages")

        else:
            st.markdown(
                f'<div class="doc-item">'
                f'<div class="doc-name">{html.escape(filename)}</div>'
                f'<div class="doc-meta">'
                f'{html.escape(status.title())} · {page_count} pages'
                f'</div></div>',
                unsafe_allow_html=True,
            )


def parse_sse_events(lines: Iterable[Any]) -> Iterator[Dict[str, Any]]:
    data_lines: List[str] = []

    def emit() -> Iterator[Dict[str, Any]]:
        if not data_lines:
            return
        payload = "\n".join(data_lines)
        try:
            event = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid chat stream event: {exc.msg}") from exc
        if not isinstance(event, dict):
            raise ValueError("Invalid chat stream event: expected an object.")
        yield event

    for raw_line in lines:
        if isinstance(raw_line, bytes):
            line = raw_line.decode("utf-8", errors="replace")
        else:
            line = str(raw_line)
        line = line.rstrip("\r\n")
        if not line:
            yield from emit()
            data_lines.clear()
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip(" "))
    yield from emit()


def stream_question(question: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    params = {
    "question": question,
    "conversation_id": st.session_state.conversation_id,
    "mode": st.session_state.mode,
    "document_filter": st.session_state.document_filter,
}
    answer_parts: List[str] = []
    steps: List[Dict[str, Any]] = []
    done: Dict[str, Any] = {}

    try:
        with requests.get(
            f"{API_URL}/chat/stream",
            params=params,
            headers={"X-User-Id": USER_ID},
            timeout=(10, 600),
            stream=True,
        ) as response:
            if response.status_code >= 400:
                return None, f"{response.status_code}: {response.text}"

            placeholder = st.empty()

            for event in parse_sse_events(response.iter_lines()):

                if event.get("event") == "step":
                    steps.append(event.get("step", {}))

                elif event.get("event") == "token":
                    answer_parts.append(event.get("token", ""))
                    placeholder.markdown("".join(answer_parts) + " ▌")

                elif event.get("event") == "done":
                    done = event

                elif event.get("event") == "error":
                    return None, event.get("detail", "Chat service failed.")

            final_answer = str(done.get("answer") or "".join(answer_parts)).strip()
            if final_answer:
                placeholder.markdown(final_answer)

    except (requests.RequestException, ValueError, UnicodeDecodeError) as exc:
        completed_answer = str(done.get("answer") or "".join(answer_parts)).strip()
        if completed_answer:
            return {
                "answer": completed_answer,
                "citations": done.get("citations", []),
                "agent_steps": steps,
                "usage": done.get("usage", {}),
                "message_id": done.get("message_id"),
                "conversation_id": done.get("conversation_id") or st.session_state.conversation_id,
            }, None
        return None, str(exc)
    except Exception as exc:
        completed_answer = str(done.get("answer") or "".join(answer_parts)).strip()
        if completed_answer:
            return {
                "answer": completed_answer,
                "citations": done.get("citations", []),
                "agent_steps": steps,
                "usage": done.get("usage", {}),
                "message_id": done.get("message_id"),
                "conversation_id": done.get("conversation_id") or st.session_state.conversation_id,
            }, None
        return None, f"Unexpected streaming error: {type(exc).__name__}: {exc}"

    answer = str(done.get("answer") or "".join(answer_parts)).strip()

    if not answer:
        return None, "The chat service returned an empty answer."

    return {
        "answer": answer,
        "citations": done.get("citations", []),
        "agent_steps": steps,
        "usage": done.get("usage", {}),
        "message_id": done.get("message_id"),
        "conversation_id": done.get("conversation_id") or st.session_state.conversation_id,
    }, None


def submit_question(question: str) -> None:
    question = question.strip()

    if not question or st.session_state.get("busy"):
        return

    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    st.session_state.busy = True

    try:
        result, error = stream_question(question)
    except Exception as exc:
        result = None
        error = f"Unexpected error: {type(exc).__name__}: {exc}"
    finally:
        st.session_state.busy = False

    if error:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"The research service could not complete this request. {error}",
            "error": True,
        })
        return

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "citations": result.get("citations", []),
        "agent_steps": result.get("agent_steps", []),
        "usage": result.get("usage", {}),
        "message_id": result.get("message_id"),
    })

    st.session_state.conversation_id = result["conversation_id"]
    if len([m for m in st.session_state.messages if m.get("role") == "user"]) == 1:
        st.session_state.conversation_title = question[:50]

for key, default in {
    "messages": [],
    "conversation_id": f"conv_{int(time.time())}",
    "conversation_title": "New research thread",
    "mode": "agent",
    "selected_source": None,
    "connection_error": None,
    "busy": False,
    "document_filter": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

documents, documents_error = api_json("GET", "/documents", timeout=8)
documents = documents if isinstance(documents, list) else []
conversations, conversations_error = get_conversations()
health, health_error = api_json("GET", "/health", timeout=8)
health = health if isinstance(health, dict) else {}

with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark">◈</div><div><div class="brand-name">Research Desk</div>'
        '<div class="brand-subtitle">Annual-report intelligence workspace</div></div></div>',
        unsafe_allow_html=True,
    )
    if st.button("＋  New conversation", use_container_width=True):
        reset_thread()
        st.rerun()

    st.markdown('<div class="upload-card"><strong>Build your research library</strong><span>Upload a PDF report to make it searchable with page-level citations.</span></div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="visible", help="Choose one annual report in PDF format.")
    if uploaded_file:
        st.caption(f"Selected: {uploaded_file.name} · {uploaded_file.size / 1024 / 1024:.1f} MB")
    if uploaded_file and st.button("Upload and process PDF", type="primary", use_container_width=True):
        with st.spinner("Uploading report…"):
            response = requests.post(
                f"{API_URL}/documents/upload",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                timeout=30,
            )
        if response.status_code == 200:
            st.success("Upload accepted. Processing is running in the background.")
            st.rerun()
        else:
            st.error(f"Upload failed ({response.status_code}): {response.text}")

    st.markdown('<div class="section-label">Document library</div>', unsafe_allow_html=True)
    if documents_error:
        st.error(f"Document library unavailable: {documents_error}")
    else:
        render_library(documents)



    st.markdown('<div class="section-label">Research mode</div>', unsafe_allow_html=True)
    st.radio(
        "Choose how answers are researched",
        ["agent", "pipeline"],
        key="mode",
        format_func=lambda value: (
            "Agent · Multi-step — Searches iteratively and gathers evidence before answering."
            if value == "agent"
            else "Pipeline · Single pass — Retrieves relevant passages and answers in one pass."
        ),
    )

    st.markdown('<div class="section-label">Conversation history</div>', unsafe_allow_html=True)
    if conversations_error:
        st.warning("Conversation history is unavailable.")
    elif conversations:
        for conversation in conversations[:12]:
            conversation_id = str(conversation.get("id", ""))
            title = str(conversation.get("title") or "Untitled thread")
            if st.button(title, key=f"conversation-{conversation_id}", use_container_width=True, help=title):
                load_conversation(conversation_id, title)
                st.rerun()
    else:
        st.caption("No saved conversations yet.")

    with st.expander("Maintenance", expanded=False):
        left, right = st.columns(2)
        with left:
            if st.button("Refresh", use_container_width=True):
                st.rerun()
        with right:
            if st.button("Reindex", use_container_width=True):
                _, error = api_json("POST", "/documents/reindex", timeout=10)
                if error:
                    st.error(f"Reindex failed: {error}")
                else:
                    st.success("Reindex started.")

st.markdown(
    f'<div class="workspace-head"><div><div class="eyebrow">Research workspace</div>'
    f'<div class="workspace-title">{html.escape(st.session_state.conversation_title)}</div></div>'
    f'<div class="health {"offline" if health_error else ""}">{"Service unavailable" if health_error else "Service ready"}</div></div>',
    unsafe_allow_html=True,
)

chat_column, trace_column = st.columns([1.62, .78], gap="large")
with chat_column:
    mode_label = "Agent · multi-step" if st.session_state.mode == "agent" else "Pipeline · single pass"
    st.markdown(f'<span class="mode-badge">◈ {mode_label}</span><div class="panel-subtitle" style="margin:.55rem 0 1rem">Ask about indexed reports. Answers retain page-level citations and the research activity panel shows how evidence was gathered.</div>', unsafe_allow_html=True)
    if st.session_state.connection_error:
        st.error(st.session_state.connection_error)
    if not st.session_state.messages:
        st.markdown('<div class="empty-state"><strong>Start a grounded research session</strong>Upload a report from the sidebar, then ask a focused question about its financial performance, risks, or operations.</div>', unsafe_allow_html=True)
        st.caption("Suggested questions")
        suggestions = [
            "What was the total power generation capacity of RWE AG?",
            "Compare operating performance across the annual reports.",
            "Find the key risk factors discussed in the reports.",
        ]
        suggestion_columns = st.columns(3)
        for index, suggestion in enumerate(suggestions):
            with suggestion_columns[index]:
                if st.button(suggestion, key=f"suggestion-{index}", use_container_width=True):
                    submit_question(suggestion)
                    st.rerun()
    for index, message in enumerate(st.session_state.messages):
        role = message.get("role", "assistant")
        with st.chat_message(role):
            if role == "user":
                st.markdown(message.get("content", ""))
            else:
                answer = str(message.get("content", ""))
                refused = is_refusal(answer)
                st.markdown(f'<div class="answer-label {"refusal" if refused else ""}">{"No matching evidence" if refused else "Grounded answer"}</div>', unsafe_allow_html=True)
                st.markdown(answer)
                if message.get("error"):
                    st.warning("This response did not complete successfully.")
                render_citations(message.get("citations", []), f"message-{index}")

    prompt = st.chat_input(
    "Ask a question about the reports...",
    disabled=st.session_state.busy,
)
    if prompt:
        submit_question(prompt)
        st.rerun()

with trace_column:
    st.markdown('<div class="panel"><div class="panel-title">Agent Activity</div><div class="panel-subtitle">Chronological searches, tool calls, observations, and synthesis from the backend trace.</div></div>', unsafe_allow_html=True)
    all_steps: List[Dict[str, Any]] = []
    all_citations: List[Dict[str, Any]] = []
    for message in st.session_state.messages:
        if message.get("role") == "assistant":
            all_steps.extend(message.get("agent_steps", []))
            all_citations.extend(message.get("citations", []))
    render_trace(all_steps[-24:])
    st.markdown('<div class="section-label">Sources and selected passage</div>', unsafe_allow_html=True)
    render_selected_source()
    st.markdown('<div class="section-label">Returned usage</div>', unsafe_allow_html=True)
    render_usage(st.session_state.messages)
    if all_citations:
        st.caption(f"{len({citation_key(c) for c in all_citations})} unique cited source(s) in this thread")
