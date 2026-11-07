# Evaluation log

The fixed scorer in `starter/eval.py` was run against the local API on
2026-11-07. No Gemini or Groq key was configured, so both modes used the
honest offline passage fallback. This is a real reproducible baseline, not a
provider-backed quality claim.

| run | answer accuracy | correct refusals | evidence cited | seconds/question | cost |
|---|---:|---:|---:|---:|---:|
| offline-agent | 0.385 | 0.0 | 0.5 | 1.001 | $0.0000 |
| offline-pipeline | 0.385 | 0.0 | 0.5 | 1.021 | $0.0000 |

The agent and single-pass pipeline tie in this offline mode because neither can
synthesize answers without an LLM. With a configured provider, rerun:

```powershell
python starter/eval.py --label agent
python starter/eval.py --label pipeline --api http://localhost:8001
```

The useful improvement retained in the implementation is normalized fallback
scoring when FlashRank is unavailable: lexical/RRF results no longer all fail
the relevance threshold. The evaluation also exposed the limitation of an
extractive fallback for comparisons and refusals, which is why the README does
not claim a high-quality final score.
