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
              └── Qdrant dense + SQLite BM25 ────────┘
                         dense + BM25 → RRF → reranker
                                      │
                         Gemini → Groq fallback
```

Documents and chunks are relational data in SQLite, which remains the source of
truth. Qdrant is the persistent dense vector database and FTS5 supplies lexical
retrieval; the vector index can be rebuilt from stored chunks and embeddings
without reopening PDFs. The included database is a
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
`GET /health` is the readiness check. A valid LLM key (or reachable Ollama
fallback) is required for conversational answers; missing or unreachable
providers return an explicit API error rather than an invented answer.

## Docker

```powershell
Copy-Item .env.example .env
# Fill in the key(s), then:
docker compose up --build
```

Compose exposes the API on port 8000, Streamlit on port 8501, and Qdrant on
port 6333. SQLite, uploaded PDFs, runtime data, and Qdrant vectors use named
volumes. On first startup, the
entrypoint seeds the named volume from the bundled starter index and reports
14 ready documents. `.env` is optional for startup, but an LLM key is required
for generated answers. Secrets are excluded by `.dockerignore` and supplied
through Compose environment files; they are not copied into the image.

Docker was verified from a running Docker Desktop daemon with
`docker compose up --build -d`, a healthy container, a real Gemini chat request,
and an upload/restart persistence test. The upload remained ready after
restart, and the original 14-document corpus remained available.

## Agent and safety behavior

Each agent step is recorded in the response and persisted with the assistant
message, including tool arguments/observations and per-LLM-call provider,
model, tokens, cost, and latency. The loop has an iteration cap, a per-turn
USD budget, and a repeated tool-call guard. Tool failures are returned as
observations so the model can retry with a better query. Uploaded PDF text is
treated as untrusted evidence and cannot override the system instructions. A
per-user budget is enforced using the `X-User-Id` header.

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

The scorer always runs the fixed 16-question golden set. The agent is a
multi-step tool-calling workflow; the pipeline is exactly one retrieval and
one synthesis call. The pipeline keeps the measured top-8 shortlist; the agent uses a wider,
page-deduplicated top-12 shortlist so its exploratory search has a better
chance of finding alternate report terminology. Runtime usage now records
provider/model, prompt tokens,
completion tokens, latency, citations, and estimated cost in each row.
Configure `PRIMARY_LLM_MODEL=gemini-3.1-flash-lite` and a valid
`GEMINI_API_KEY` before running the provider-backed scorecard. Use
`--resume --delay-seconds 20` for the agent on the free-tier request limit.
See [`EVAL.md`](./EVAL.md) for the executed real Gemini and offline scorecards,
per-question artifacts, before/after experiments, checkpoint behavior, quota
findings, and reproduction commands. The latest guarded experiment scored
0.154 agent accuracy / 0.417 evidence versus 0.538 / 0.583 for the pipeline,
with one Gemini HTTP 503 recorded in the agent run. Across the clean
same-model scorecards, the agent remains slower, more expensive, and less
accurate overall; it is not currently worth the additional latency and cost.
The latest retrieval-fix pipeline rerun completed at 0.462 accuracy and 0.583
evidence. Its paired agent rerun was interrupted by Gemini HTTP 429 quota
exhaustion after q07; the incomplete rows are preserved and are not presented
as a complete scorecard.

The latest complete paced evaluation used `gemini-3.1-flash-lite` after the
previous `gemini-flash-lite-latest` alias returned HTTP 404. Both 16-question
runs completed without errors: the agent scored 0.538 accuracy, 1.000 correct refusals, 0.583 evidence, 15.068
seconds/question, 142,540 prompt tokens,
7,056 completion tokens, and $0.0128; the pipeline scored 0.538 accuracy,
0.667 correct refusals, 0.667 evidence, 14.934 seconds/question, 68,599
prompt tokens, 859 completion tokens, and $0.0054. On this run the agent was not worth its additional cost and produced
no accuracy improvement. It had better refusal accuracy, but lower evidence
coverage (0.583 versus 0.667).

## Tests

```powershell
python -m pytest -q
python -m compileall -q src scripts tests
```

The tests cover deterministic chunking, citation parsing, FTS sanitization,
RRF, calculator safety, cost estimation, prompt-injection instructions,
token-bounded history, citation passage retrieval, and FastAPI
chat/document/feedback contracts with mocked LLM calls.

## Known limitations

FlashRank and local embeddings are optional at runtime; if unavailable, the
system uses normalized RRF results and lexical retrieval. Citation controls
open the original PDF with a `#page=N` fragment and show the extracted passage;
browser PDF behavior depends on the installed browser viewer. Ollama is a
configurable local fallback (`FALLBACK_PROVIDER=ollama`) but requires a
separately running Ollama server and pulled model. Authentication is
intentionally limited to the required user-id header. OCR for scanned pages,
structured table querying, and a separate task queue are not implemented.
