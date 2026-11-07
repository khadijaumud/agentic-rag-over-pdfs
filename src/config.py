import os
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "starter" / "data"
PDF_DIR = DATA_DIR / "pdfs"
DB_PATH = BASE_DIR / "data" / "rag_database.db"

# LLM & API configurations
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

PRIMARY_LLM_MODEL = os.getenv("PRIMARY_LLM_MODEL", "gemini-2.5-flash")
FALLBACK_LLM_MODEL = os.getenv("FALLBACK_LLM_MODEL", "llama-3.3-70b-versatile")
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Force offline mode for local models to prevent network retry delays
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Retrieval & RAG Hyperparameters
DEFAULT_TOP_K_DENSE = int(os.getenv("TOP_K_DENSE", "10"))
DEFAULT_TOP_K_LEXICAL = int(os.getenv("TOP_K_LEXICAL", "10"))
DEFAULT_RRF_K = int(os.getenv("RRF_K", "60"))
DEFAULT_RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))
DEFAULT_RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.05"))

# Agent Constraints
MAX_AGENT_ITERATIONS = int(os.getenv("MAX_AGENT_ITERATIONS", "6"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "6"))
PER_TURN_BUDGET_USD = float(os.getenv("PER_TURN_BUDGET_USD", "0.05"))
