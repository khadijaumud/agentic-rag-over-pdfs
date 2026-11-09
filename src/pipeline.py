"""Single-Pass RAG Baseline Pipeline.
Performs exactly ONE retrieval step and ONE LLM call.
Used as the benchmark against the multi-step agent.
"""

import re
from typing import Dict, Any, List, Optional
from src.retrieval import hybrid_search
from src.llm import call_llm_with_fallback

CITATION_REGEX = re.compile(
    r"(?P<document>[A-Za-z0-9][A-Za-z0-9\s,.\-_&()']*?\.pdf)\s*,\s*"
    r"p(?:age|\.)?\s*(?P<page>\d+)",
    re.IGNORECASE,
)
BRACKET_CITATION_REGEX = re.compile(r"\[([^\]]+)\]")
PAGE_REGEX = re.compile(r"\bp(?:age|\.)?\s*(\d+)", re.IGNORECASE)

def extract_citations_from_text(text: str, fallback_chunks: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Extract [Doc.pdf, p.X] citations from text; fallback to retrieved chunks if none in text."""
    found = []
    seen = set()
    
    bracket_spans = []
    for bracket in BRACKET_CITATION_REGEX.finditer(text):
        body = bracket.group(1)
        document_match = re.search(
            r"(?P<document>[A-Za-z0-9][A-Za-z0-9\s,.\-_&()']*?\.pdf)",
            body,
            re.IGNORECASE,
        )
        if not document_match:
            continue
        doc = document_match.group("document").strip()
        doc = re.sub(r"(?:\.pdf)+$", ".pdf", doc, flags=re.IGNORECASE)
        for page_match in PAGE_REGEX.finditer(body[document_match.end():]):
            key = (doc.lower(), int(page_match.group(1)))
            if key not in seen:
                seen.add(key)
                found.append({"document": doc, "page": key[1]})
        bracket_spans.append(bracket.span())

    unbracketed = text
    for start, end in reversed(bracket_spans):
        unbracketed = unbracketed[:start] + unbracketed[end:]
    for match in CITATION_REGEX.finditer(unbracketed):
        doc = match.group("document").strip()
        page = int(match.group("page"))
        if not doc.lower().endswith(".pdf"):
            doc = doc + ".pdf"
        doc = re.sub(r"(?:\.pdf)+$", ".pdf", doc, flags=re.IGNORECASE)
        key = (doc.lower(), page)
        if key not in seen:
            seen.add(key)
            found.append({"document": doc, "page": page})
            
    # If model didn't format brackets explicitly, but cited in prose or used the retrieved passages
    if not found and fallback_chunks:
        for c in fallback_chunks[:3]:
            key = (c["document_id"].lower(), c["page_number"])
            if key not in seen:
                seen.add(key)
                found.append({"document": c["document_id"], "page": c["page_number"]})
                
    return found

def run_single_pass_pipeline(
    question: str,
    top_k: int = 8,
    document_filter: Optional[str] = None
) -> Dict[str, Any]:
    """Execute a single-pass RAG retrieval + LLM synthesis."""
    # 1. Single retrieval step
    retrieved = hybrid_search(question, document_filter=document_filter, top_k=top_k)
    
    if not retrieved:
        # Honest refusal if retrieval found nothing relevant
        return {
            "answer": "Data not available. The report does not contain information to answer this question.",
            "citations": [],
            "usage": {"usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0},
            "retrieved_chunks": []
        }
        
    # Format context
    context_blocks = []
    for idx, c in enumerate(retrieved, 1):
        context_blocks.append(f"--- Context {idx} [{c['document_id']}, p.{c['page_number']}] ---\n{c['content']}\n")
    context_str = "\n".join(context_blocks)
    
    prompt = f"""Use ONLY the following context to answer the question.
If the context does not contain the necessary facts, reply: "Data not available or not mentioned in the report."
Cite every claim with [document_name.pdf, p.X].

Context:
{context_str}

Question: {question}
Answer:"""

    messages = [{"role": "user", "content": prompt}]
    llm_resp = call_llm_with_fallback(messages, temperature=0.0)
    answer_text = llm_resp["text"]
    if llm_resp.get("provider") == "none":
        answer_text = (
            "The configured LLM is unavailable. The most relevant report passages are:\n\n"
            + "\n\n".join(c["content"] for c in retrieved)
        )
    
    citations = extract_citations_from_text(answer_text, fallback_chunks=retrieved)
    
    return {
        "answer": answer_text,
        "citations": citations,
        "usage": {
            "usd": llm_resp["cost_usd"],
            "prompt_tokens": llm_resp["prompt_tokens"],
            "completion_tokens": llm_resp["completion_tokens"],
            "provider": llm_resp.get("provider"),
            "model": llm_resp.get("model"),
        },
        "retrieved_chunks": retrieved
    }
