"""
Centralized configuration — loads environment variables and exposes constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# ── Reddit API ──────────────────────────────────────────────────────────────
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "wsb-sentiment-agent/1.0")

# ── Ollama ──────────────────────────────────────────────────────────────────
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ── Paths ───────────────────────────────────────────────────────────────────
DATA_DIR = _PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

CHROMA_DIR = str(DATA_DIR / "chroma_db")

# ── Model defaults ──────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SENTIMENT_MODEL = "ProsusAI/finbert"
SPACY_MODEL = "en_core_web_sm"
