"""Multi-Step ReAct Agent for Agentic RAG.
Features:
- Dynamic tool-calling loop (Search, List Documents, Read Page, Calculate)
- Iteration cap, cycle/repeat guard, per-turn budget limit
- Robust error handling (tool errors returned as observations)
- Full step-by-step trace logging for UI and debugging
- Grounded citations parsing
"""

import json
import re
from typing import Dict, Any, List, Optional, Tuple

from src.config import MAX_AGENT_ITERATIONS, PER_TURN_BUDGET_USD
from src.retrieval import hybrid_search
from src.db import list_documents as db_list_docs, get_page_chunks
from src.llm import call_llm_with_fallback
from src.pipeline import extract_citations_from_text

# Tool definitions
def tool_list_documents() -> str:
    """List all available documents in the repository."""
    docs = db_list_docs()
    if not docs:
        return "No documents currently in the repository."
    res = [f"- {d['filename']} ({d['page_count']} pages)" for d in docs]
    return "Available annual reports:\n" + "\n".join(res)

def tool_search_documents(query: str, document_filter: Optional[str] = None) -> str:
    """Search documents using hybrid search."""
    try:
        results = hybrid_search(query, document_filter=document_filter, top_k=5)
        if not results:
            return f"No relevant passages found for query '{query}'. Try different search terms or check document name."
        
        output = []
        for idx, r in enumerate(results, 1):
            output.append(f"[{idx}] Source: [{r['document_id']}, p.{r['page_number']}]\n{r['content']}")
        return "\n\n".join(output)
    except Exception as e:
        return f"[Tool Error in search_documents] {str(e)}. Try a simpler search query."

def tool_read_page(document: str, page_number: int) -> str:
    """Read all text content from an exact 1-based page of a document."""
    try:
        # Normalize document name
        if not document.lower().endswith(".pdf"):
            document = document + ".pdf"
        chunks = get_page_chunks(document, int(page_number))
        if not chunks:
            return f"[Notice] No text found for {document} on page {page_number}. Verify the page number."
        return f"--- Content of {document} Page {page_number} ---\n" + "\n\n".join(c["content"] for c in chunks)
    except Exception as e:
        return f"[Tool Error in read_page] {str(e)}. Check the document name and page number."

def tool_calculate(expression: str) -> str:
    """Safely calculate basic arithmetic expressions."""
    # Allow only safe characters
    cleaned = re.sub(r"[^0-9\+\-\*\/\(\)\.\s]", "", expression)
    if not cleaned.strip():
        return "[Calculator Error] Expression must contain numbers and basic operators (+, -, *, /)."
    try:
        val = eval(cleaned, {"__builtins__": None}, {})
        return f"{expression} = {val}"
    except Exception as e:
        return f"[Calculator Error] Failed to evaluate: {e}"

TOOLS_MAP = {
    "search_documents": tool_search_documents,
    "list_documents": tool_list_documents,
    "read_page": tool_read_page,
    "calculate": tool_calculate
}

AGENT_SYSTEM_PROMPT = """You are an Agentic RAG Assistant answering factual and financial questions from annual reports.
You have access to the following tools:

1. `search_documents(query, document_filter=None)`: Hybrid search over PDF reports. Returns ranked passages with [Document.pdf, p.X].
2. `list_documents()`: Lists all indexed company reports.
3. `read_page(document, page_number)`: Reads full text of a specific page (1-based).
4. `calculate(expression)`: Computes math expressions.

RESPONSE FORMAT PROTOCOL:
At each turn, you must think and either call a tool OR provide your final answer using this EXACT format:

Thought: <explain your reasoning and next step>
Action: <one of: search_documents, list_documents, read_page, calculate>
Action Input: <valid JSON object of arguments, e.g. {"query": "power generation capacity", "document_filter": "RWE AG.pdf"}>

When you have gathered enough evidence, or determined that the report does not contain the answer, respond with:

Thought: I have enough evidence or verified that the information is absent.
Final Answer: <your cited answer or refusal>

GROUNDING AND REFUSAL RULES:
- Every claim must cite the exact source: [Document.pdf, p.X].
- If the report does not state the requested data, your Final Answer MUST explicitly say so (e.g. "Data not available", "The report does not mention...", "N/A").
- Do NOT guess. Always search first.
"""

def parse_agent_action(llm_output: str) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    """Parse Thought, Action, Action Input, or Final Answer from model response."""
    thought_match = re.search(r"Thought:\s*(.*?)(?=(Action:|Final Answer:|$))", llm_output, re.DOTALL | re.IGNORECASE)
    thought = thought_match.group(1).strip() if thought_match else ""
    
    final_answer_match = re.search(r"Final Answer:\s*(.*)", llm_output, re.DOTALL | re.IGNORECASE)
    if final_answer_match:
        final_answer = final_answer_match.group(1).strip()
        return None, None, thought, final_answer
        
    action_match = re.search(r"Action:\s*([a-zA-Z0-9_]+)", llm_output, re.IGNORECASE)
    action_input_match = re.search(r"Action Input:\s*(\{.*?\}|\[.*?\]|\".*?\"|'.*?'|[^\n]+)", llm_output, re.DOTALL | re.IGNORECASE)
    
    if action_match:
        action = action_match.group(1).strip()
        args = {}
        if action_input_match:
            raw_input = action_input_match.group(1).strip()
            # Try parsing JSON
            try:
                args = json.loads(raw_input)
                if not isinstance(args, dict):
                    args = {"input": args}
            except Exception:
                # If simple string or unquoted
                args = {"query": raw_input.strip("\"' ")}
        return action, args, thought, None
        
    return None, None, thought, llm_output.strip()

def run_agent(
    question: str,
    max_iterations: int = MAX_AGENT_ITERATIONS,
    conversation_history: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Execute the multi-step Agentic RAG loop."""
    steps = []
    seen_actions = set()
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost_usd = 0.0
    all_retrieved_chunks = []
    observations = []
    
    messages = []
    if conversation_history:
        messages.extend(conversation_history)
        
    # Initial user prompt
    messages.append({"role": "user", "content": f"Question: {question}"})
    
    final_answer = None
    
    for step_num in range(1, max_iterations + 1):
        # 1. Budget safeguard check
        if total_cost_usd >= PER_TURN_BUDGET_USD:
            steps.append({
                "step": step_num,
                "thought": "Budget ceiling reached.",
                "action": "force_synthesis",
                "observation": "Turn budget limit exceeded. Synthesizing answer with available information."
            })
            break
            
        # 2. Call LLM
        llm_resp = call_llm_with_fallback(messages, system_prompt=AGENT_SYSTEM_PROMPT, temperature=0.0)
        total_prompt_tokens += llm_resp["prompt_tokens"]
        total_completion_tokens += llm_resp["completion_tokens"]
        total_cost_usd += llm_resp["cost_usd"]

        if llm_resp.get("provider") == "none":
            # Keep the application useful and honest without silently inventing an
            # answer when no remote provider is configured.
            results = hybrid_search(question, top_k=5)
            if results:
                final_answer = (
                    "The configured LLM is unavailable. Relevant report passages:\n\n"
                    + "\n\n".join(r["content"] for r in results)
                )
                all_retrieved_chunks.extend(results)
            else:
                final_answer = "The configured LLM is unavailable, and the reports do not contain a relevant passage."
            steps.append({
                "step": step_num,
                "thought": "No configured LLM provider was available.",
                "action": "offline_fallback",
                "action_args": {},
                "observation": final_answer[:1500],
            })
            break
        
        reply_text = llm_resp["text"]
        action, action_args, thought, answer = parse_agent_action(reply_text)
        
        if answer is not None:
            final_answer = answer
            steps.append({
                "step": step_num,
                "thought": thought,
                "action": "Final Answer",
                "action_args": {},
                "observation": None
            })
            break
            
        if not action or action not in TOOLS_MAP:
            # If model didn't format action properly, send clarification observation
            observation = f"[Instruction] Please select one of the valid tools: {list(TOOLS_MAP.keys())} or provide 'Final Answer:'."
            steps.append({
                "step": step_num,
                "thought": thought,
                "action": action or "unknown",
                "action_args": action_args,
                "observation": observation
            })
            observations.append(observation)
            messages.append({"role": "assistant", "content": reply_text})
            messages.append({"role": "user", "content": f"Observation: {observation}"})
            continue

        # 3. Loop Guard: prevent executing the exact same tool call
        action_key = (action, json.dumps(action_args, sort_keys=True))
        if action_key in seen_actions:
            observation = "[Guard] You already executed this exact tool call with these arguments. Refine your query or proceed to 'Final Answer:'."
            steps.append({
                "step": step_num,
                "thought": thought,
                "action": action,
                "action_args": action_args,
                "observation": observation
            })
            messages.append({"role": "assistant", "content": reply_text})
            messages.append({"role": "user", "content": f"Observation: {observation}"})
            continue
            
        seen_actions.add(action_key)
        
        # 4. Execute tool
        tool_fn = TOOLS_MAP[action]
        try:
            if action == "list_documents":
                observation = tool_fn()
            elif action == "search_documents":
                q = action_args.get("query", question)
                doc_filt = action_args.get("document_filter") or action_args.get("document")
                observation = tool_fn(query=q, document_filter=doc_filt)
            elif action == "read_page":
                doc = action_args.get("document", "")
                page = int(action_args.get("page_number", action_args.get("page", 1)))
                observation = tool_fn(document=doc, page_number=page)
            elif action == "calculate":
                expr = action_args.get("expression", action_args.get("expr", ""))
                observation = tool_fn(expression=expr)
            else:
                observation = tool_fn(**action_args)
        except Exception as e:
            observation = f"[Tool Error] {str(e)}"
            
        steps.append({
            "step": step_num,
            "thought": thought,
            "action": action,
            "action_args": action_args,
            "observation": observation[:1500] # Truncate observation in trace for brevity
        })
        
        messages.append({"role": "assistant", "content": reply_text})
        messages.append({"role": "user", "content": f"Observation:\n{observation}"})
        
    # If loop ended without final answer, force synthesis
    if final_answer is None and total_cost_usd < PER_TURN_BUDGET_USD:
        messages.append({"role": "user", "content": "You have reached the maximum search steps. Provide your Final Answer now based strictly on the observations above."})
        forced_resp = call_llm_with_fallback(messages, system_prompt=AGENT_SYSTEM_PROMPT, temperature=0.0)
        total_prompt_tokens += forced_resp["prompt_tokens"]
        total_completion_tokens += forced_resp["completion_tokens"]
        total_cost_usd += forced_resp["cost_usd"]
        
        _, _, _, forced_ans = parse_agent_action(forced_resp["text"])
        final_answer = forced_ans or forced_resp["text"]
    elif final_answer is None:
        final_answer = (
            "The per-turn budget was reached before a final answer could be generated. "
            "Available observations:\n\n" + "\n\n".join(observations[-3:])
        )

    citations = extract_citations_from_text(final_answer, fallback_chunks=all_retrieved_chunks)
    
    return {
        "answer": final_answer,
        "citations": citations,
        "agent_steps": steps,
        "usage": {
            "usd": round(total_cost_usd, 6),
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens
        }
    }
