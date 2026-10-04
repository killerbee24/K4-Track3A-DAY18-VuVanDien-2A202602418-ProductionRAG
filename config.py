"""Shared configuration for Lab 18."""

import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# MWAPI uses an Anthropic-compatible API by default. Legacy key names are
# accepted so an existing .env does not need to expose/copy its secret.
LLM_API_FORMAT = os.getenv("LLM_API_FORMAT", "anthropic").strip().lower()
LLM_API_KEY = (os.getenv("LLM_API_KEY") or os.getenv("MWAPI_API_KEY")
               or os.getenv("ANTHROPIC_API_KEY") or OPENAI_API_KEY).strip()
LLM_BASE_URL = (os.getenv("LLM_BASE_URL") or os.getenv("ANTHROPIC_BASE_URL")
                or os.getenv("OPENAI_BASE_URL") or "https://api.mwapi.dev").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))

# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
EVAL_EMBEDDING_MODEL = os.getenv("EVAL_EMBEDDING_MODEL", EMBEDDING_MODEL)

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
