# 3–5 Minute Demo Runbook

1. Start the stack with `docker compose up --build`; open
   `http://localhost:8501`. Confirm the health indicator and 14 ready reports.
2. Upload a PDF in the sidebar and wait for its status to become `ready`.
3. Ask a five-company comparison question. Show the streamed answer, source
   citations, and the Research activity trace with searches, arguments, and
   per-call usage.
4. Click a citation, choose **Open original PDF at this page**, and verify the
   browser opens the report with the 1-based page fragment. Show the extracted
   passage beside it.
5. Ask an unanswerable question and show the grounded refusal. Open a previous
   conversation to demonstrate persisted history.

Useful checks:

```powershell
Invoke-RestMethod http://localhost:8000/health
docker compose ps
docker compose logs --tail=40 rag-app
```

The demo requires a valid `GEMINI_API_KEY` (or a configured Groq key). For a
local fallback, run Ollama separately and set `FALLBACK_PROVIDER=ollama`.
