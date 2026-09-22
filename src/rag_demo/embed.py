"""Stage 3 of the RAG pipeline: embedding (text -> vectors).

An embedding model turns a piece of text into a fixed-length list of numbers
(a vector) such that texts with similar *meaning* get vectors pointing in
similar directions. "How long should I brine a turkey?" and a chunk about
salting poultry overnight end up close together, even with few words in common.
This is what lets retrieval search by meaning rather than by keyword.

Every vector is normalised to unit length. Then the cosine similarity of two
vectors is simply their dot product, which keeps the retrieval maths simple.

Anthropic does not offer an embedding model, so two options are provided:

- LocalEmbedder: `all-MiniLM-L6-v2` via sentence-transformers. Free, offline,
  384 numbers per vector. Downloaded (~90 MB) the first time it is used.
- VoyageEmbedder: Voyage AI's hosted models (Anthropic's recommended embeddings
  provider). Needs VOYAGE_API_KEY. Voyage embeds documents and queries slightly
  differently (`input_type`), which improves retrieval.

The same embedder must be used for the chunks and the questions: vectors from
different models live in different spaces and cannot be compared.
"""

import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

import numpy as np

from rag_demo import config

# Silence a harmless Hugging Face warning about symlinks on Windows, and the
# library's own progress bars (the CLI shows its own).
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

EMBEDDERS = ("local", "voyage")


class EmbedderUnavailableError(RuntimeError):
    """Raised when an embedding model cannot be loaded."""


def normalise(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to unit length, so dot product == cosine similarity."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return (vectors / np.where(norms == 0, 1, norms)).astype(np.float32)


class Embedder(ABC):
    """Turns text into unit-length vectors."""

    #: Identifies the model; stored with the index so a mismatch can be detected.
    name: str
    batch_size: int = 64

    @abstractmethod
    def _embed(self, texts: list[str], input_type: str) -> np.ndarray:
        """Embed one batch. `input_type` is "document" or "query"."""

    def embed_documents(
        self, texts: list[str], on_batch: Callable[[int], None] | None = None
    ) -> np.ndarray:
        """Embed chunks in batches. Returns an (n_texts x dimension) matrix."""
        batches = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batches.append(self._embed(batch, input_type="document"))
            if on_batch:
                on_batch(len(batch))
        return normalise(np.vstack(batches))

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single question. Returns a vector of length `dimension`."""
        return normalise(self._embed([text], input_type="query"))[0]


class LocalEmbedder(Embedder):
    """sentence-transformers running on this machine. No API key needed."""

    def __init__(self, model_name: str = config.LOCAL_EMBED_MODEL):
        # Imported here: loading PyTorch takes a few seconds, and commands that
        # don't embed anything shouldn't pay for it.
        from sentence_transformers import SentenceTransformer
        from transformers.utils import logging as transformers_logging

        transformers_logging.disable_progress_bar()

        # model_name is a Hugging Face model ID, or a folder holding a downloaded
        # copy (for machines that can't reach huggingface.co). Either way the
        # index records the model's name, not where it was loaded from.
        self.name = f"local:{Path(model_name).name}"
        try:
            self.model = SentenceTransformer(model_name, device="cpu")
        except OSError as err:
            raise EmbedderUnavailableError(
                f"Could not load the local embedding model '{model_name}'. The first run downloads it "
                "(~90 MB) from huggingface.co, so that site must be reachable. Alternatively, download "
                "the model folder on another network and set LOCAL_EMBED_MODEL=<path to folder> in .env."
            ) from err

    def _embed(self, texts: list[str], input_type: str) -> np.ndarray:
        # all-MiniLM-L6-v2 treats documents and queries the same way.
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


class VoyageEmbedder(Embedder):
    """Voyage AI's hosted embedding models. Needs VOYAGE_API_KEY."""

    batch_size = 128

    def __init__(self, model_name: str = config.VOYAGE_EMBED_MODEL):
        import voyageai

        key = config.voyage_api_key()
        if key is None:
            raise config.MissingAPIKeyError(
                "EMBEDDER is 'voyage' but VOYAGE_API_KEY is missing. Add it to .env, "
                "or use the local embedder (--embedder local)."
            )
        self.name = f"voyage:{model_name}"
        self.model_name = model_name
        self.client = voyageai.Client(api_key=key)

    def _embed(self, texts: list[str], input_type: str) -> np.ndarray:
        result = self.client.embed(texts, model=self.model_name, input_type=input_type)
        return np.asarray(result.embeddings, dtype=np.float32)


def get_embedder(kind: str = config.EMBEDDER) -> Embedder:
    """Build the embedder named by `kind` ("local" or "voyage")."""
    if kind == "local":
        return LocalEmbedder()
    if kind == "voyage":
        return VoyageEmbedder()
    raise ValueError(f"Unknown embedder {kind!r}; choose from {', '.join(EMBEDDERS)}.")
