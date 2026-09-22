"""Central configuration for the RAG demo.

Every tunable setting in the pipeline lives here, so the audience can see in one
place which knobs exist (chunk size, top-k, model names, ...). CLI flags override
these defaults at run time.

Secrets (API keys) are read from a `.env` file in the project root via
python-dotenv. Key values are never printed: callers should only ever report
whether a key is "found" or "missing".
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# --- Paths -----------------------------------------------------------------

# src/rag_demo/config.py -> project root is two levels above the package.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = PROJECT_ROOT / "index"
ENV_FILE = PROJECT_ROOT / ".env"

# Load .env once, on import. Variables already set in the shell take precedence.
load_dotenv(ENV_FILE)

# --- Pipeline defaults -----------------------------------------------------

CHUNK_STRATEGY = "fixed"  # "fixed" or "paragraph"
CHUNK_SIZE = 800  # characters per chunk
CHUNK_OVERLAP = 150  # characters shared between neighbouring fixed-size chunks
EMBEDDER = os.getenv("EMBEDDER", "local")  # "local" (sentence-transformers) or "voyage"
# A Hugging Face model ID, or a path to a downloaded copy of the model folder.
LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "all-MiniLM-L6-v2")
VOYAGE_EMBED_MODEL = "voyage-3.5"
INDEX_BACKEND = "numpy"  # "numpy" or "chroma"
TOP_K = 5
CLAUDE_MODEL = "claude-sonnet-5"
MAX_TOKENS = 1024


# --- Secrets ---------------------------------------------------------------


def _read_key(name: str) -> str | None:
    """Return the named key from the environment, or None if unset or blank."""
    value = os.getenv(name, "").strip()
    return value or None


def anthropic_api_key() -> str | None:
    """The Anthropic key, used only for the generation stage (Claude)."""
    return _read_key("ANTHROPIC_API_KEY")


def voyage_api_key() -> str | None:
    """The optional Voyage AI key, used only when EMBEDDER is "voyage"."""
    return _read_key("VOYAGE_API_KEY")


class MissingAPIKeyError(RuntimeError):
    """Raised when a required API key is not configured."""


def require_anthropic_api_key() -> str:
    """Return the Anthropic key, or raise a clear error that never echoes a value."""
    key = anthropic_api_key()
    if key is None:
        raise MissingAPIKeyError(
            "ANTHROPIC_API_KEY is missing. Copy .env.example to .env in the project "
            f"root ({PROJECT_ROOT}) and set ANTHROPIC_API_KEY=<your key>."
        )
    return key
