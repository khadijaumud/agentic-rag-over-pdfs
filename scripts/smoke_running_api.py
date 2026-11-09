"""Smoke-test a running Docker API without touching seeded documents.

Usage:
    python scripts/smoke_running_api.py
    python scripts/smoke_running_api.py --base-url http://localhost:8000 --skip-chat
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pymupdf
import requests


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--skip-chat", action="store_true", help="Avoid provider cost during smoke testing.")
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")
    health = None
    deadline = time.time() + 90
    while time.time() < deadline:
        try:
            response = requests.get(f"{base_url}/health", timeout=10)
            if response.ok:
                health = response.json()
                break
        except requests.RequestException:
            pass
        time.sleep(2)
    if health is None:
        raise RuntimeError("Running API did not become reachable within 90 seconds.")
    assert health["status"] == "ok"

    documents = requests.get(f"{base_url}/documents", timeout=10)
    documents.raise_for_status()
    seeded_names = {item["filename"] for item in documents.json()}

    filename = f"api-smoke-{int(time.time())}.pdf"
    pdf_path = Path(__file__).with_name(filename)
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Running API smoke test document.")
    pdf.save(pdf_path)
    pdf.close()
    document_id = None
    try:
        with pdf_path.open("rb") as handle:
            uploaded = requests.post(
                f"{base_url}/documents/upload",
                files={"file": (filename, handle, "application/pdf")},
                timeout=30,
            )
        uploaded.raise_for_status()
        assert uploaded.json()["status"] == "pending"

        deadline = time.time() + 60
        while time.time() < deadline:
            current = requests.get(f"{base_url}/documents", timeout=10)
            current.raise_for_status()
            match = next((item for item in current.json() if item["filename"] == filename), None)
            if match:
                document_id = match["id"]
                if match["status"] == "ready":
                    break
                if match["status"] == "failed":
                    raise RuntimeError("Uploaded smoke-test PDF entered failed status.")
            time.sleep(2)
        assert document_id, "Uploaded document did not appear in the document list."

        if not args.skip_chat:
            chat = requests.post(
                f"{base_url}/chat",
                json={"question": "What text is in the smoke-test document?", "mode": "pipeline"},
                headers={"X-User-Id": "running_api_smoke"},
                timeout=180,
            )
            chat.raise_for_status()
            assert "answer" in chat.json()
    finally:
        if document_id:
            deleted = requests.delete(f"{base_url}/documents/{document_id}", timeout=30)
            deleted.raise_for_status()
        pdf_path.unlink(missing_ok=True)

    remaining = requests.get(f"{base_url}/documents", timeout=10)
    remaining.raise_for_status()
    assert seeded_names.issubset({item["filename"] for item in remaining.json()})
    print("Running API smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
