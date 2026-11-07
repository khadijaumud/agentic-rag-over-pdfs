# Agentic RAG over PDFs

This project answers questions about 14 annual company reports with a
tool-calling research agent. The agent can search the corpus, list documents,
read an exact page, and calculate arithmetic. Answers cite the original PDF
filename and a one-based page number.

## Architecture

```text
Browser (Streamlit) ──HTTP/SSE──> FastAPI
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
        SQLite source       Agent loop           Background upload
        of truth + FTS5     search/read/calc      parse → chunk → embed
              │                   │                   │
              └──────────── hybrid retrieval ────────┘
                         dense + BM25 → RRF → reranker
                                      │
                         Gemini → Groq fallback
```

Documents and chunks are relational data in SQLite. FTS5 supplies lexical
retrieval and embeddings supply dense retrieval; the vector index can be
rebuilt from stored chunks without reopening PDFs. The included database is a
seeded convenience index for the starter corpus. New indexes can be rebuilt
with the ingestion script.

## Run locally

```powershell
Copy-Item .env.example .env
# Set GEMINI_API_KEY, or GROQ_API_KEY as a fallback, in .env
python -m pip install -r requirements.txt
python scripts/ingest_starter_corpus.py
python -m uvicorn src.app:app --reload --port 8000
```

In another terminal:

```powershell
streamlit run src/ui.py
```

The API is at `http://localhost:8000`, the UI at `http://localhost:8501`, and
`GET /health` is the readiness check. Without an LLM key the app remains
usable for retrieval and displays relevant passages, but it cannot synthesize
normal conversational answers.

## Docker

```powershell
Copy-Item .env.example .env
# Fill in the key(s), then:
docker compose up --build
```

Compose exposes the API on port 8000 and Streamlit on port 8501. SQLite and
evaluation runs use named volumes. `.env` is optional for startup, but an LLM
key is required for generated answers. Docker was not available in the
development environment used for this submission, so image build and
container startup still need to be run on a Docker-enabled machine.

## Agent and safety behavior

Each agent step is recorded in the response and persisted with the assistant
message. The loop has an iteration cap, a per-turn USD budget, and a repeated
tool-call guard. Tool failures are returned as observations so the model can
retry with a better query. Uploaded PDF text is treated as untrusted evidence
and cannot override the system instructions. A per-user budget is enforced
using the `X-User-Id` header.

The upload endpoint returns immediately with `pending`; `/documents` reports
`processing`, `ready`, or `failed`. Deleting a document cascades through
chunks and FTS5. `/documents/reindex` rebuilds embeddings from SQLite.
`/chat/stream` emits SSE events for agent steps, answer tokens, citations, and
usage.

## Evaluation

The fixed scorer is `starter/eval.py`; it must not be changed below its
`# scoring: do not edit` marker.

```powershell
python starter/eval.py --label agent
python starter/eval.py --label pipeline --api http://localhost:8001
```

The repository contains the golden set and starter PDFs. The locally executed
offline comparison is recorded in [`EVAL.md`](./EVAL.md) and the raw scorecards
are [`starter/runs/offline-agent.json`](./starter/runs/offline-agent.json) and
[`starter/runs/offline-pipeline.json`](./starter/runs/offline-pipeline.json).
Those numbers measure retrieval plus extractive fallback only: no LLM key was
available, so they are not a claim about provider-backed answer quality. The
intended comparison is agent versus the one-retrieval/one-call pipeline in
[`src/pipeline.py`](./src/pipeline.py).

## Tests

```powershell
python -m pytest -q
python -m compileall -q src scripts tests
```

The tests cover deterministic chunking, citation parsing, FTS sanitization,
RRF, calculator safety, cost estimation, and FastAPI chat/document/feedback
contracts with mocked LLM calls.

## Known limitations

FlashRank and local embeddings are optional at runtime; if unavailable, the
system uses normalized RRF results and lexical retrieval. The Streamlit UI
currently simulates token streaming from the completed SSE-compatible backend
response rather than rendering provider-native token callbacks. Authentication
is intentionally limited to the required user-id header. OCR for scanned pages,
structured table querying, and a separate task queue are not implemented.
