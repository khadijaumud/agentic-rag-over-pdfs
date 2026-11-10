# Agentic RAG over PDFs

This project answers questions about 14 annual company reports with a
tool-calling research agent. The agent can search the corpus, list documents,
read an exact page, and calculate arithmetic. Answers cite the original PDF
filename and a one-based page number.

## What the app does

- Upload PDF reports with a real progress bar; they are parsed, chunked and
  indexed in the background, and each one shows `pending`, `processing`, `ready`
  or `failed`.
- Ask questions in **Agent** mode (the model decides what to search, reads pages,
  searches again, then answers) or **Single pass** mode (one retrieval, one model
  call).
- Every answer cites document and page. Selecting a citation shows the extracted
  passage and can open the original PDF at that page.
- The **Research steps** panel lists every search, page read and calculation the
  agent made; the **Usage** panel shows tokens and estimated cost.
- Conversations are saved, follow-up questions work, and each user has a cost limit.

## Architecture

```text
Browser (web UI)   ──HTTP/SSE──> FastAPI
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
without reopening PDFs. The included database is a seeded convenience index for
the starter corpus. New indexes can be rebuilt with the ingestion script.

Repository layout: `src/` (API, agent, retrieval, ingestion, storage),
`frontend/` (web UI), `starter/` (corpus, golden set, scorer, recorded runs),
`tests/`, `scripts/`, `EVAL.md` (evaluation log).

## Run with Docker

```powershell
Copy-Item .env.example .env      # Linux/macOS: cp .env.example .env
# Set GEMINI_API_KEY (or GROQ_API_KEY as a fallback) in .env, then:
docker compose up --build
```

Then open `http://localhost:8000` (use `http://127.0.0.1:8000` if `localhost`
does not respond on Windows). Compose exposes the API and the web UI on port 8000
and Qdrant on port 6333. SQLite, uploaded PDFs, runtime data and Qdrant vectors
use named volumes. On first startup the entrypoint seeds the named volume from the
bundled starter index, waits for Qdrant, rebuilds the vector index from SQLite
(a few seconds for the 2,867 seeded chunks) and starts the server; `GET /health`
reports 14 ready documents.

The first build takes several minutes because `sentence-transformers` needs
PyTorch (the image installs the CPU-only build). Later builds use the Docker
cache. `.env` is optional for startup, but an LLM key is required for generated
answers. Secrets are excluded by `.dockerignore` and supplied through Compose
environment files; they are not copied into the image.

A real Gemini chat request and an upload/restart persistence test were run
against an earlier build of this stack: the uploaded document stayed `ready`
after a restart and the original 14-document corpus remained available.

## Run locally

```powershell
Copy-Item .env.example .env
# Set GEMINI_API_KEY, or GROQ_API_KEY as a fallback, in .env
python -m pip install -r requirements.txt
python scripts/ingest_starter_corpus.py
python -m uvicorn src.app:app --port 8000
```

The web UI and the API are both served by FastAPI at `http://localhost:8000`, and
`GET /health` is the readiness check. A valid LLM key (or reachable Ollama
fallback) is required for conversational answers; missing or unreachable
providers return an explicit API error rather than an invented answer.

## Agent and safety behavior

Each agent step is recorded in the response and persisted with the assistant
message, including tool arguments/observations and per-LLM-call provider, model,
tokens, cost and latency. The loop has an iteration cap, a per-turn USD budget and
a repeated tool-call guard. Tool failures are returned as observations so the
model can retry with a better query. Uploaded PDF text is treated as untrusted
evidence and cannot override the system instructions. A per-user budget is
enforced using the `X-User-Id` header.

The upload endpoint returns immediately with `pending`; `/documents` reports
`processing`, `ready` or `failed`. Deleting a document cascades through chunks and
FTS5. `/documents/reindex` rebuilds embeddings from SQLite. `/chat/stream` emits
SSE events for agent steps, answer tokens, citations and usage.

The web UI (`frontend/`: plain HTML, CSS and JavaScript served by FastAPI, no
build step) uploads with `XMLHttpRequest`, so the progress bar shows real
transferred bytes. After the upload it polls `/documents` and shows the backend's
status. It replaced an earlier Streamlit UI, whose `st.file_uploader` could not
report byte-level progress.

## Evaluation

The fixed scorer is `starter/eval.py`; it must not be changed below its
`# scoring: do not edit` marker. It always runs the fixed 16-question golden set
(three questions cannot be answered from the reports).

```powershell
python starter/eval.py --label agent
python starter/eval.py --label pipeline --api http://localhost:8001
```

The agent is a multi-step tool-calling workflow; the pipeline is exactly one
retrieval and one synthesis call. The pipeline keeps the measured top-8 shortlist;
the agent uses a wider, page-deduplicated top-12 shortlist so its exploratory
search has a better chance of finding alternate report terminology. Configure
`PRIMARY_LLM_MODEL=gemini-3.1-flash-lite` and a valid `GEMINI_API_KEY` before
running the provider-backed scorecard. Use `--resume --delay-seconds 20` for the
agent on the free-tier request limit.

### Agent versus single pass

Latest complete paced run, `gemini-3.1-flash-lite`, all 16 questions, no errors:

| | Agent | Single pass |
|---|---|---|
| Answer accuracy | 0.538 | 0.538 |
| Correct refusals (3 unanswerable) | 1.000 | 0.667 |
| Evidence cited | 0.583 | 0.667 |
| Seconds per question | 15.1 | 14.9 |
| Prompt / completion tokens | 142,540 / 7,056 | 68,599 / 859 |
| Cost for the set | $0.0128 | $0.0054 |

Source: `starter/runs/gemini-agent-complete.json` and
`starter/runs/gemini-pipeline-complete.json`. The previous
`gemini-flash-lite-latest` model alias returned HTTP 404, which is why this run
uses the pinned model.

**Is the agent worth it?** Not by these numbers. Accuracy is equal, the agent
costs about 2.4 times more and cites fewer pages. It refused all three
unanswerable questions while the pipeline refused two, and on the five-company
comparison question (q10) the agent was right and the pipeline was wrong in this
run, but each of these is a difference of a single question on a 16-question set,
so it is weak evidence. In another experiment (the protocol-guard run) the agent
scored 0.154 accuracy / 0.417 evidence versus 0.538 / 0.583 for the pipeline, with
one Gemini HTTP 503 recorded in the agent run. The latest retrieval-fix pipeline
rerun completed at 0.462 accuracy and 0.583 evidence; its paired agent rerun was
interrupted by Gemini HTTP 429 quota exhaustion after q07, and the incomplete rows
are preserved and not presented as a complete scorecard.

See [`EVAL.md`](./EVAL.md) for every executed scorecard, per-question artifacts,
before/after experiments, checkpoint behavior, quota findings and reproduction
commands.

## Tests

```powershell
python -m pytest -q
python -m compileall -q src scripts tests
```

The tests cover deterministic chunking, citation parsing, FTS sanitization, RRF,
calculator safety, cost estimation, prompt-injection instructions, token-bounded
history, citation passage retrieval, serving of the web UI, and FastAPI
chat/document/feedback contracts with mocked LLM calls.

## Known limitations

- **Latency and streaming.** `/chat/stream` runs the agent to completion and then
  streams the finished answer word by word, so tokens are not produced while the
  model is still thinking. The UI shows an elapsed-time indicator meanwhile, and
  agent answers often take 15 to 40 seconds.
- **Small evaluation set.** 16 questions cannot separate small differences; see
  the comparison above.
- **Citations.** The agent cites fewer pages than the single-pass pipeline, and
  the UI turns only citations written as `[File.pdf, p.12]` into clickable chips.
  Citation controls open the original PDF with a `#page=N` fragment; browser PDF
  behavior depends on the installed viewer.
- **Optional components.** FlashRank and local embeddings are optional at runtime;
  if unavailable, the system falls back to normalized RRF results and lexical
  retrieval. Ollama is a configurable local fallback (`FALLBACK_PROVIDER=ollama`)
  but needs a separately running Ollama server and pulled model.
- **Web UI.** It has no automated browser tests; it was checked manually against
  the running API.
- **Not implemented.** Authentication beyond the required user-id header, OCR for
  scanned pages, structured table querying and a separate task queue.
- **First Docker build** takes several minutes (PyTorch download).