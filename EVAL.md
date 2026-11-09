# Evaluation report

## Scope and fixed methodology

The evaluation uses the unchanged 16-question dataset at
`starter/data/golden_set.json` and the unchanged scoring section in
`starter/eval.py`. There are 13 answerable questions, 3 unanswerable
questions, and one five-report comparison. The scorer measures answer accuracy,
refusal accuracy, evidence-page coverage (one page tolerance), wall-clock
latency, and the sum of reported estimated USD cost.

The two workflows are materially different:

* **Pipeline baseline:** one hybrid retrieval call, one reranking pass, and one
  Gemini synthesis call.
* **Agent:** a bounded ReAct loop where Gemini chooses search, list-documents,
  read-page, and calculator tools; it can search repeatedly before producing a
  cited answer. Repeated calls, iteration count, and per-turn cost are guarded.

Both use the same SQLite source-of-truth corpus, Qdrant dense index, FTS5 index, reranker settings,
questions, and Gemini model configuration. Retrieval tuning was measured before
the change: top-5 evidence-group recall was 0.625; top-8 was 0.688; top-10
did not improve it. The retained pipeline setting is top-8; the agent uses a
wider top-12 exploratory shortlist with page-level deduplication.

## Executed artifacts

The original valid real-provider scorecards used `gemini-3.1-flash-lite`:

* [`gemini-agent-final.json`](./starter/runs/gemini-agent-final.json)
* [`gemini-pipeline-final.json`](./starter/runs/gemini-pipeline-final.json)

The comparable same-model run after the grounding changes used
`gemini-flash-lite-latest`:

* [`gemini-latest-agent-grounded.json`](./starter/runs/gemini-latest-agent-grounded.json)
* [`gemini-latest-pipeline.json`](./starter/runs/gemini-latest-pipeline.json)

* [`offline-agent-final.json`](./starter/runs/offline-agent-final.json)
* [`offline-pipeline-final.json`](./starter/runs/offline-pipeline-final.json)

The latest complete paced evaluation, after the shortlist-diversity and
refusal-guard changes, is saved separately:

* [`gemini-agent-complete.json`](./starter/runs/gemini-agent-complete.json)
* [`gemini-pipeline-complete.json`](./starter/runs/gemini-pipeline-complete.json)

Each row contains the question id, answer, citations, latency, estimated cost,
provider/model fields, and prompt/completion token fields. All 16 ids are
present in each file.

| run | answer accuracy | correct refusals | evidence cited | seconds/question | cost |
|---|---:|---:|---:|---:|---:|
| gemini-agent-final | 0.538 | 1.000 | 0.333 | 18.247 | $0.0093 |
| gemini-pipeline-final | 0.462 | 1.000 | 0.583 | 5.982 | $0.0056 |
| gemini-latest-agent-grounded | 0.231 | 0.667 | 0.417 | 14.401 | $0.0115 |
| gemini-latest-pipeline | 0.462 | 1.000 | 0.583 | 3.666 | $0.0057 |
| gemini-stopfix-agent | 0.308 | 1.000 | 0.250 | 14.143 | $0.0086 |
| gemini-stopfix-pipeline | 0.462 | 1.000 | 0.500 | 2.910 | $0.0056 |
| gemini-protocolguard-agent | 0.154 | 0.333 | 0.417 | 22.154 | $0.0119 |
| gemini-protocolguard-pipeline | 0.538 | 1.000 | 0.583 | 3.171 | $0.0056 |
| gemini-retrievalfix-agent | 0.077* | 0.000* | 0.250* | 8.064* | $0.0052* |
| gemini-retrievalfix-pipeline | 0.462 | 1.000 | 0.583 | 2.517 | $0.0056 |
| offline-agent-final | 0.385 | 0.000 | 0.500 | 1.351 | $0.0000 |
| offline-pipeline-final | 0.538 | 0.667 | 0.500 | 1.399 | $0.0000 |
| gemini-agent-complete | 0.538 | 1.000 | 0.583 | 15.068 | $0.0128 |
| gemini-pipeline-complete | 0.538 | 0.667 | 0.667 | 14.934 | $0.0054 |

`*` The retrieval-fix agent artifact is incomplete as a provider experiment:
q08-q16 failed with Gemini HTTP 429 `RESOURCE_EXHAUSTED` after the provider
quota was exhausted. Those rows are retained as errors and must not be
interpreted as answer-quality measurements.

The Gemini agent improved answer accuracy over the Gemini baseline by 0.076
and matched its refusal score, but used 1.67x the estimated cost and 3.05x
the latency while citing fewer evidence groups. On this fixed set, the agent
is not an unqualified win: it improved answer correctness but needs better
evidence discipline and more efficient tool use.

The offline rows are retrieval/extractive-fallback measurements only and are
not comparable to provider-backed answer quality.

### Latest complete comparison

Both latest runs used the same `gemini-3.1-flash-lite` model, the same
16-question dataset, and the same SQLite corpus. They completed all 16 rows
with zero evaluator errors.

| metric | agent | pipeline | agent minus pipeline |
|---|---:|---:|---:|
| answer accuracy | 0.538 | 0.538 | 0.000 |
| correct refusals | 1.000 | 0.667 | +0.333 |
| evidence cited | 0.583 | 0.667 | -0.084 |
| seconds/question | 15.068 | 14.934 | +0.134 |
| prompt tokens | 142,540 | 68,599 | +73,941 |
| completion tokens | 7,056 | 859 | +6,197 |
| estimated USD | $0.0128 | $0.0054 | +$0.0074 |

The agent used about 2.08x the prompt tokens, 8.21x the completion tokens,
and 2.37x the estimated cost. It was marginally slower in this run, did not
improve answer accuracy, and cited fewer evidence groups. The agent improved
correct refusals, but under the assignment's accuracy, grounding, latency,
and cost criteria it is not worth the additional expense on this fixed set.

## Improvement experiments

### 1. Structured agent evidence and citation normalization — kept

**Problem observed:** agent search/read observations were sent to Gemini but
were not retained in `all_retrieved_chunks`. When the model omitted a citation,
the final response could not recover the pages it had actually read. The
citation parser also treated strings such as `Westwater Resources, Inc..pdf,
p.16` as part of the filename and produced malformed names ending in
`.pdf.pdf`.

**Change:** search results and exact read-page chunks are now retained as
structured evidence; duplicate retrieval in the same tool execution was
removed; citation parsing supports multiple pages in one bracket and
normalizes repeated extensions. The agent also rejects empty or uncited
non-refusal final responses and asks the model to search/read again.

The same-model healthy-alias run provides the measurable before/after:

| agent run | accuracy | refusals | evidence | seconds/q | cost |
|---|---:|---:|---:|---:|---:|
| before: `gemini-latest-agent` | 0.077 | 0.000 | 0.000 | 4.696 | $0.0024 |
| after: `gemini-latest-agent-grounded` | 0.231 | 0.667 | 0.417 | 14.401 | $0.0115 |

The citation/reliability improvement was kept, despite the cost/latency
increase and the remaining accuracy gap. The run is not evidence that the
agent is better than the pipeline overall.

### 2. Forced-synthesis stopping bug — kept, targeted verification only

The persisted q02 trace showed that when the agent reached its iteration cap,
the final synthesis call still received the tool-calling protocol and returned
another `Action` block. The loop then stored that protocol text as the answer.
The forced-synthesis call now uses an answer-only system prompt, while the
normal multi-step tool loop is unchanged. A parser regression test covers the
boundary between tool calls and final answers.

This fix was verified with unit tests and a real Docker chat smoke test, but a
new 16-question provider-backed scorecard was not run after this change. The
scorecards above therefore remain the latest comparable measurements and are
not silently relabeled as post-fix results.

### 3. Retrieval shortlist measurement — previously kept

Golden evidence-group recall was measured at 0.625 with top-5 and 0.688 with
top-8; top-10 did not improve it. The default shortlist was changed to top-8
for both workflows. This is general retrieval tuning, not a question-specific
rule.

### 4. Generic lexical vocabulary expansion — kept, retrieval-only benefit

The failure audit found that report wording such as “people employed” was not
retrieved for a question using “healthcare professionals”, and leadership and
product questions used related vocabulary absent from the query. Lexical
retrieval now adds small generic synonym sets for workforce, leadership,
product, acquisition, and operating-activity terms. It does not contain
company names or golden answers.

Measured evidence-group recall improved from 11/16 to 12/16 on the fixed
questions. The fresh pipeline scorecard remained 0.462 accuracy and 0.583
evidence, so the change was retained as a retrieval reliability improvement,
not as a claimed end-to-end accuracy gain. q03, q07, q09, and q10 remain
retrieval/coverage weaknesses.

### 5. Agent shortlist diversity and refusal guard — kept after complete rerun

The local retrieval audit showed that several expected pages were present in
the database but were hidden by duplicate chunks on the same page or by the
short agent shortlist. The agent now requests a wider top-12 shortlist and
removes duplicate document/page entries from the observation sent to Gemini.
Its prompt also requires separate document-filtered searches for named
comparisons. Finally, a refusal after a search is not accepted until at least
one exact page has been read, reducing unsupported refusals when search already
found promising evidence.

These general changes are covered by regression tests. The complete rerun is
the `gemini-agent-complete.json` artifact above. It improved the agent's
evidence score relative to the earlier clean agent-final run (0.333 to 0.583)
and preserved 1.000 correct refusals, but did not improve accuracy over the
pipeline and increased total cost.

### 6. Citation passage access, per-call traces, and history budget — kept

The API now exposes `GET /documents/{document}/pages/{page}` for the extracted
passage behind a citation. The Streamlit citation panel loads and displays
that passage instead of showing metadata only. Each agent LLM call is recorded
as a trace step with provider, model, prompt tokens, completion tokens, and
estimated cost. Conversation history is capped by both turns and an estimated
token budget. These changes are covered by regression tests and do not alter
the fixed evaluator or scorecards.

## Question-level comparison

For the final same-model run, `OK/BAD` is the fixed scorer's answer result;
evidence is the scorer's page-group result:

| id | agent | pipeline | agent evidence | pipeline evidence | diagnosis |
|---|---|---|---:|---:|---|
| q01 | BAD | BAD | no | no | expected page absent from top-8; both refused |
| q02 | BAD | OK | yes | yes | agent ended on an action text; pipeline extracted cash-flow fact |
| q03 | BAD | BAD | yes | no | agent found page but refused R&D value; generation failure |
| q04 | BAD | BAD | no | yes | agent refused despite page evidence; citation omission |
| q05 | BAD | OK | yes | yes | agent selected a wrong balance-sheet value/page |
| q06 | OK | OK | yes | yes | both retrieved and cited exact page |
| q07 | BAD | BAD | no | no | leadership-change page absent from initial retrieval |
| q08 | BAD | BAD | no | yes | agent refused; pipeline cited non-answer evidence |
| q09 | BAD | BAD | no | no | expected page absent; agent did not recover it |
| q10 | BAD | BAD | no | no | multi-report comparison needs complete per-report evidence |
| q11 | OK | OK | yes | yes | correct boolean/refusal-style answer |
| q12 | BAD | OK | no | yes | agent refused a supported acquisition question |
| q13 | OK | OK | yes | yes | correct dividend-policy answer |
| q14 | OK | OK | yes | yes | correct refusal |
| q15 | OK | OK | yes | yes | correct refusal |
| q16 | BAD | OK | yes | yes | agent answered the wrong product/research interpretation |

The additional agent steps improved q06 and citation recovery, but did not
overcome retrieval misses on q01/q07/q09 or multi-document completeness on
q10. The pipeline remains faster, cheaper, and more consistently evidenced.

## Gemini execution and blocker

The configured Gemini credential was verified with model listing and real
inference. The stale `gemini-flash-lite-latest` alias returned HTTP 404, so
the configuration was corrected to the available `gemini-3.1-flash-lite`
model; the Gemini provider and key were not changed. The prior concurrent run
hit the provider's free-tier limit of 15 generate-content requests per minute
and returned HTTP 429. The latest runs used checkpointing and 20-second pacing;
all 16 rows in both complete files have no errors. The Groq fallback credential
remains invalid (HTTP 401), but no Groq or offline fallback was used in the
complete Gemini scorecards.

The later `gemini-protocolguard-agent` run recorded one real provider failure
on q15: Gemini HTTP 503, “This model is currently experiencing high demand.”
The evaluator preserved that row as an error rather than retrying it silently.
The corresponding pipeline run had 16 successful provider rows. These later
runs are retained as additional experiments and do not replace the earlier
clean scorecards.

The fresh `gemini-retrievalfix-agent` run encountered the provider's quota:
q08-q16 each returned HTTP 429 `RESOURCE_EXHAUSTED` after q01-q07.
The bounded retry handles transient 503/UNAVAILABLE responses once, but does
not retry quota errors; this avoids multiplying rate-limit traffic. The exact
failed requests and rows remain in
[`gemini-retrievalfix-agent.json`](./starter/runs/gemini-retrievalfix-agent.json).

The evaluator writes each row atomically and supports `--resume`; successful
rows are skipped while error rows are retried. This prevents a quota
interruption from repeating completed provider calls.

## Reproduction

From `agentic-rag-over-pdfs`:

```powershell
python -m pip install -r requirements.txt
python scripts/ingest_starter_corpus.py
python -m uvicorn src.app:app --host 127.0.0.1 --port 8000
python starter/eval.py --label gemini-agent-complete --api http://127.0.0.1:8000 --resume --delay-seconds 20
python starter/eval.py --label gemini-pipeline-complete --api http://127.0.0.1:8000 --resume --delay-seconds 20
```

Set `GEMINI_API_KEY` and `PRIMARY_LLM_MODEL=gemini-3.1-flash-lite` in `.env`.
Use a valid `GROQ_API_KEY` only if fallback coverage is desired. Do not treat
the offline commands or mocked tests as provider-backed evaluation.

## Validation

Executed successfully:

```powershell
python -m pytest -q
python -m compileall -q src scripts tests
docker compose config --quiet
docker version
docker compose up --build -d
docker compose ps
curl http://localhost:8000/health
```

The current regression suite result is 26 passed with two existing FastAPI deprecation
warnings. Tests mock LLM calls where noted in the test module; they do not
prove Gemini quality. Docker Desktop was running, the rebuilt container
reported healthy, and `/health` reported 14 ready documents before and after
the evaluations.

The running-container integration flow used a real Gemini pipeline request for
the RWE capacity question. It returned provider `gemini`, model
`gemini-flash-lite-latest`, citation `[RWE AG.pdf, p.47]`, 437 prompt tokens,
31 completion tokens, and estimated cost `$0.000042`.

Persistence was tested, not inferred from Compose configuration: a PDF was
uploaded and reached `ready` with 240 pages, the container was restarted, and
the PDF remained ready. The 14-document health count was restored after the
temporary test document was deleted. The conversation created by the real
chat request was also present after restart. Container logs showed startup
messages but no `ERROR`, `Traceback`, or `WARNING` lines in the final check.

## Professor requirement checklist

| Requirement | Status | Verification |
|---|---|---|
| Agent tool-calling loop and bounded guards | Implemented and verified | Persisted traces, unit/integration tests |
| PDF upload, background ingestion, statuses | Implemented and verified | Upload reached `ready` in Docker |
| SQLite source of truth and persistent vector rebuild | Implemented and verified | SQLite stores documents/chunks; Qdrant stores 2,867 dense points and `/documents/reindex` rebuilds the index without PDF parsing |
| Hybrid retrieval, reranking, threshold | Implemented and verified | Unit tests and prior measured retrieval experiment |
| Grounded answers, citations, passage opening | Implemented and verified | Real RWE citation and API/UI code paths |
| FastAPI, SSE streaming, health endpoint | Implemented and verified | Health, chat, and live `start`/`token`/`done` SSE events verified |
| Conversations, budgets, usage, safety, fallback | Implemented and verified | Persisted conversations, token-bounded history, usage records, prompt defense, Groq fallback, and tested Ollama failure handling |
| Streamlit UI and agent trace display | Implemented; UI load checked | Citation source controls, original PDF page links, extracted passages, per-call usage display, and agent timeline are wired to API responses; a full browser citation click requires a browser PDF viewer |
| Docker secrets and named-volume persistence | Implemented and verified | `.dockerignore`, fresh image, upload/restart test |
| Unit/integration tests and traces | Implemented and verified | 23 tests passed; detailed traces persist in assistant messages |
| Evaluation and before/after scorecards | Implemented and verified | Complete agent and pipeline runs saved with zero evaluator errors |
| README and reproducibility instructions | Implemented and verified | Updated with current Docker and evaluation facts |
| OCR, structured table querying, task queue, real authentication | Missing | Explicit project limitations |
