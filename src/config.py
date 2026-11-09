import os
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Load the project configuration first, with the starter location as a
# compatibility fallback for existing local setups.
load_dotenv(BASE_DIR / ".env")
if not (BASE_DIR / ".env").exists():
    load_dotenv(BASE_DIR / "starter" / ".env")

DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
PDF_DIR = Path(os.getenv("PDF_DIR", str(BASE_DIR / "starter" / "data" / "pdfs")))
DB_PATH = Path(os.getenv("DB_PATH", str(DATA_DIR / "rag_database.db")))

# LLM & API configurations
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

PRIMARY_LLM_MODEL = os.getenv("PRIMARY_LLM_MODEL", "gemini-flash-lite-latest")
FALLBACK_LLM_MODEL = os.getenv("FALLBACK_LLM_MODEL", "llama-3.3-70b-versatile")
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Retrieval & RAG Hyperparameters
DEFAULT_TOP_K_DENSE = int(os.getenv("TOP_K_DENSE", "10"))
DEFAULT_TOP_K_LEXICAL = int(os.getenv("TOP_K_LEXICAL", "10"))
DEFAULT_RRF_K = int(os.getenv("RRF_K", "60"))
DEFAULT_RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "8"))
DEFAULT_RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.05"))

# Agent Constraints
MAX_AGENT_ITERATIONS = int(os.getenv("MAX_AGENT_ITERATIONS", "6"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "6"))
MAX_HISTORY_TOKENS = int(os.getenv("MAX_HISTORY_TOKENS", "6000"))
PER_TURN_BUDGET_USD = float(os.getenv("PER_TURN_BUDGET_USD", "0.05"))
MAX_USER_BUDGET_USD = float(os.getenv("MAX_USER_BUDGET_USD", "5.0"))
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "120"))
VECTOR_DB_URL = os.getenv("VECTOR_DB_URL", "http://qdrant:6333")
VECTOR_COLLECTION = os.getenv("VECTOR_COLLECTION", "pdf_chunks")
VECTOR_DB_TIMEOUT_SECONDS = float(os.getenv("VECTOR_DB_TIMEOUT_SECONDS", "5"))
