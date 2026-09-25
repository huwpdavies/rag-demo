"""Stage 3 of the RAG pipeline: embedding (text -> vectors).

An embedding model turns a piece of text into a fixed-length list of numbers
(a vector) such that texts with similar *meaning* get vectors pointing in
similar directions. "How long should I brine a turkey?" and a chunk about
salting poultry overnight end up close together, even with few words in common.
This is what lets retrieval search by meaning rather than by keyword.

Every vector is normalised to unit length. Then the cosine similarity of two
vectors is simply their dot product, which keeps the retrieval maths simple.

Anthropic does not offer an embedding model, so embeddings are computed
locally with `all-MiniLM-L6-v2` via sentence-transformers: free, offline, no
API key, 384 numbers per vector. It is downloaded (~90 MB) on first use.

The same model must be used for the chunks and the questions: vectors from
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


class EmbedderUnavailableError(RuntimeError):
    """Raised when an embedding model cannot be loaded."""


def normalise(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to unit length, so dot product == cosine similarity."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return (vectors / np.where(norms == 0, 1, norms)).astype(np.float32)


def embedder_name(model_name: str = config.EMBED_MODEL) -> str:
    """The name recorded in the index manifest, e.g. "all-MiniLM-L6-v2".

    Available without loading the model, so an index can be checked cheaply.
    A folder path is reduced to the folder's name: the index records which
    model was used, not where it was loaded from.
    """
    return Path(model_name).name


class Embedder(ABC):
    """Turns text into unit-length vectors.

    Documents and queries have separate methods because some embedding models
    encode them differently; all-MiniLM-L6-v2 happens to treat them the same.
    """

    #: Identifies the model; stored with the index so a mismatch can be detected.
    name: str
    batch_size: int = 64

    @abstractmethod
    def _embed(self, texts: list[str]) -> np.ndarray:
        """Embed one batch of texts."""

    def embed_documents(
        self, texts: list[str], on_batch: Callable[[int], None] | None = None
    ) -> np.ndarray:
        """Embed chunks in batches. Returns an (n_texts x dimension) matrix."""
        batches = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batches.append(self._embed(batch))
            if on_batch:
                on_batch(len(batch))
        return normalise(np.vstack(batches))

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single question. Returns one vector."""
        return normalise(self._embed([text]))[0]


class LocalEmbedder(Embedder):
    """sentence-transformers running on this machine. No API key needed."""

    def __init__(self, model_name: str = config.EMBED_MODEL):
        # Imported here: loading PyTorch takes a few seconds, and commands that
        # don't embed anything shouldn't pay for it.
        from sentence_transformers import SentenceTransformer
        from transformers.utils import logging as transformers_logging

        transformers_logging.disable_progress_bar()

        # model_name is a Hugging Face model ID, or a folder holding a downloaded
        # copy (for machines that can't reach huggingface.co).
        self.name = embedder_name(model_name)
        # Use the copy saved on disk if there is one. Otherwise the library
        # checks huggingface.co for updates on every load, which is slow, and on
        # a network that blocks the site, hangs until it times out.
        try:
            self.model = SentenceTransformer(model_name, device="cpu", local_files_only=True)
            return
        except OSError:
            pass  # not downloaded yet
        try:
            self.model = SentenceTransformer(model_name, device="cpu")
        except OSError as err:
            raise EmbedderUnavailableError(
                f"Could not load the embedding model '{model_name}'. The first run downloads it "
                "(~90 MB) from huggingface.co, so that site must be reachable. Alternatively, download "
                "the model folder on another network and set EMBED_MODEL=<path to folder> in .env."
            ) from err

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def get_embedder(model_name: str = config.EMBED_MODEL) -> Embedder:
    """Build the embedder used for both chunks and questions."""
    return LocalEmbedder(model_name)
