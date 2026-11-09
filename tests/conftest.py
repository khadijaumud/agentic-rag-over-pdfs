"""Test configuration: isolate every test run from the seeded production database.

The application reads DB_PATH/DATA_DIR from the environment at import time, so
they are redirected to a throwaway directory before any ``src`` module loads.
Running the suite must never mutate ``data/rag_database.db`` or the PDFs.
"""

import os
import tempfile
from pathlib import Path

_TEST_ROOT = Path(tempfile.mkdtemp(prefix="agentic-rag-tests-"))
os.environ["DATA_DIR"] = str(_TEST_ROOT)
os.environ["DB_PATH"] = str(_TEST_ROOT / "test_rag_database.db")
os.environ["PDF_DIR"] = str(_TEST_ROOT / "pdfs")
os.environ.setdefault("VECTOR_DB_URL", "http://127.0.0.1:1")
os.environ.setdefault("VECTOR_DB_TIMEOUT_SECONDS", "0.2")
