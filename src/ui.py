"""Streamlit Web Interface for Agentic RAG.
Features:
- Live chat with multi-step Agent vs Single-pass Pipeline selector
- Agent thought & tool execution inspection pane
- Document manager with upload progress and status
- Grounded citations display with passage viewing
- Budget and token cost tracking
- User feedback (thumbs up / thumbs down)
"""

import time
import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Agentic RAG - Annual Reports",
    page_icon="📊",
    layout="wide"
)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = f"conv_{int(time.time())}"
if "mode" not in st.session_state:
    st.session_state.mode = "agent"

# Sidebar: Document Management & Controls
with st.sidebar:
    st.title("⚙️ RAG Configuration")
    
    # Mode selection
    mode = st.radio(
        "Retrieval Architecture:",
        ["agent", "pipeline"],
        format_func=lambda x: "🤖 Agent (Multi-step ReAct)" if x == "agent" else "⚡ Pipeline (Single-pass Baseline)",
        index=0 if st.session_state.mode == "agent" else 1
    )
    st.session_state.mode = mode

    st.divider()
    st.subheader("📁 Annual Reports Corpus")
    
    # Document uploader
    uploaded_file = st.file_uploader("Upload new PDF Report", type=["pdf"])
    if uploaded_file is not None:
        if st.button("Ingest Uploaded PDF"):
            with st.spinner("Uploading and indexing PDF in background..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    resp = requests.post(f"{API_URL}/documents/upload", files=files, timeout=30)
                    if resp.status_code == 200:
                        st.success(f"Uploaded {uploaded_file.name}! Processing in background.")
                    else:
                        st.error(f"Upload failed: {resp.text}")
                except Exception as e:
                    st.error(f"Cannot connect to API: {e}")

    # List indexed documents
    try:
        docs_resp = requests.get(f"{API_URL}/documents", timeout=5)
        if docs_resp.status_code == 200:
            docs = docs_resp.json()
            st.caption(f"Total Reports: {len(docs)}")
            for d in docs:
                status_color = "🟢" if d["status"] == "ready" else "🟡"
                st.text(f"{status_color} {d['filename']} ({d['page_count']} p.)")
    except Exception:
        st.warning("Backend API is currently offline. Start API on port 8000.")

    st.divider()
    if st.button("🔄 Rebuild Vector Index"):
        with st.spinner("Rebuilding index from database..."):
            try:
                r = requests.post(f"{API_URL}/documents/reindex", timeout=10)
                st.info("Re-indexing started in background.")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.conversation_id = f"conv_{int(time.time())}"
        st.rerun()

# Main Chat View
st.title("📊 Financial Report Agentic RAG")
st.caption("Ask questions across 14 corporate annual reports. Backed by hybrid retrieval, RRF, and multi-step ReAct.")

# Render existing messages
for msg_idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display Agent Execution Steps if available
        if msg.get("agent_steps"):
            with st.expander("🔍 View Agent Reasoning & Tool Calls", expanded=False):
                for s in msg["agent_steps"]:
                    st.markdown(f"**Step {s.get('step')}:** {s.get('thought')}")
                    if s.get("action") and s.get("action") != "Final Answer":
                        st.code(f"Action: {s.get('action')}\nArgs: {s.get('action_args')}", language="json")
                        if s.get("observation"):
                            st.caption(f"Observation: {s.get('observation')[:500]}...")
                            
        # Display Grounded Citations
        if msg.get("citations"):
            st.markdown("##### 📌 Citations")
            for c in msg["citations"]:
                doc = c.get("document") or c.get("doc", "Unknown")
                page = c.get("page", "?")
                st.badge(f"📄 {doc} — Page {page}")
                
        # Display Usage
        if msg.get("usage"):
            u = msg["usage"]
            cost = u.get("usd", 0.0)
            st.caption(f"⏱️ Tokens: {u.get('prompt_tokens', 0)} in / {u.get('completion_tokens', 0)} out | Est. Cost: ${cost:.5f}")

# User Input
if prompt := st.chat_input("Ask a question about the annual reports..."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        with st.spinner(f"Analyzing documents using {st.session_state.mode}..."):
            try:
                payload = {
                    "question": prompt,
                    "conversation_id": st.session_state.conversation_id,
                    "mode": st.session_state.mode
                }
                headers = {"X-User-Id": "streamlit_user"}
                resp = requests.post(f"{API_URL}/chat", json=payload, headers=headers, timeout=120)
                
                if resp.status_code == 200:
                    data = resp.json()
                    answer = data.get("answer", "")
                    citations = data.get("citations", [])
                    steps = data.get("agent_steps", [])
                    usage = data.get("usage", {})
                    
                    st.markdown(answer)
                    
                    if steps:
                        with st.expander("🔍 View Agent Reasoning & Tool Calls", expanded=True):
                            for s in steps:
                                st.markdown(f"**Step {s.get('step')}:** {s.get('thought')}")
                                if s.get("action") and s.get("action") != "Final Answer":
                                    st.code(f"Action: {s.get('action')}\nArgs: {s.get('action_args')}", language="json")
                                    if s.get("observation"):
                                        st.caption(f"Observation: {s.get('observation')[:500]}...")
                                        
                    if citations:
                        st.markdown("##### 📌 Citations")
                        for c in citations:
                            doc = c.get("document") or c.get("doc", "Unknown")
                            page = c.get("page", "?")
                            st.badge(f"📄 {doc} — Page {page}")
                            
                    if usage:
                        cost = usage.get("usd", 0.0)
                        st.caption(f"Tokens: {usage.get('prompt_tokens', 0)} in / {usage.get('completion_tokens', 0)} out | Cost: ${cost:.5f}")
                        
                    # Save assistant message to session
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "citations": citations,
                        "agent_steps": steps,
                        "usage": usage,
                        "message_id": data.get("message_id")
                    })
                else:
                    err_msg = f"API Error ({resp.status_code}): {resp.text}"
                    st.error(err_msg)
                    st.session_state.messages.append({"role": "assistant", "content": err_msg})
            except Exception as e:
                err_msg = f"Failed to connect to backend: {e}"
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
