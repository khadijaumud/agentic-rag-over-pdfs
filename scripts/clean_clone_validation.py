"""Validate Docker startup in an isolated temporary project and volume set.

The source tree is copied to a temporary directory, so the current database,
working tree, and existing Compose volumes are never modified. The command
also checks seeded documents, API health, and the web UI.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import requests


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> int:
    source = Path(__file__).resolve().parents[1]
    project_name = f"rag_clean_{uuid.uuid4().hex[:10]}"
    api_port, qdrant_port = free_port(), free_port()
    with tempfile.TemporaryDirectory(prefix="rag-clean-clone-") as directory:
        clone = Path(directory) / "project"
        shutil.copytree(
            source,
            clone,
            ignore=shutil.ignore_patterns(".git", ".env", "__pycache__", ".pytest_cache", "*.pyc"),
        )
        compose_file = clone / "docker-compose.yml"
        compose_text = compose_file.read_text(encoding="utf-8")
        compose_text = compose_text.replace('"6333:6333"', f'"{qdrant_port}:6333"')
        compose_text = compose_text.replace('"8000:8000"', f'"{api_port}:8000"')
        compose_file.write_text(compose_text, encoding="utf-8")
        compose = ["docker", "compose", "-p", project_name, "up", "--build", "-d"]
        try:
            subprocess.run(compose, cwd=clone, check=True)
            deadline = time.time() + 180
            health = None
            while time.time() < deadline:
                try:
                    response = requests.get(f"http://localhost:{api_port}/health", timeout=5)
                    if response.ok:
                        health = response.json()
                        break
                except requests.RequestException:
                    pass
                time.sleep(3)
            if not health or health.get("status") != "ok":
                raise RuntimeError("Isolated API did not become healthy.")
            documents = requests.get(f"http://localhost:{api_port}/documents", timeout=10)
            documents.raise_for_status()
            assert len(documents.json()) >= 1
            web_ui = requests.get(f"http://localhost:{api_port}/", timeout=10)
            web_ui.raise_for_status()
            assert "Research Desk" in web_ui.text
            print(f"Clean-clone validation passed: {len(documents.json())} seeded documents.")
        finally:
            subprocess.run(
                ["docker", "compose", "-p", project_name, "down", "-v", "--remove-orphans"],
                cwd=clone,
                check=False,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
