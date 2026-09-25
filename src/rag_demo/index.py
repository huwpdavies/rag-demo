"""Stage 4 of the RAG pipeline: indexing (vectors + chunks -> searchable store).

Embedding every chunk is the slow part of the pipeline, so it is done once, at
ingest time, and the results are saved. The index is what later answers "which
chunks are most similar to this question?".

The NumPy index shows how simple a vector index can be:

- `vectors.npy`: one matrix, one row per chunk (e.g. 1,083 x 384).
- `chunks.json`: the text and metadata of each chunk, in the same row order.
- `manifest.json`: how the index was built (embedding model, chunking settings,
  which PDF). Stored so the index can be checked before it is trusted.

Searching is one matrix-vector product: because every vector has unit length,
`vectors @ query` gives the cosine similarity of the query to every chunk.

The manifest guards against a classic silent RAG bug: querying an index with a
different embedding model from the one that built it. The vectors would have
nothing to do with each other, yet search would still return confident-looking
(but meaningless) results. Loading refuses such an index instead.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rag_demo.chunk import Chunk

MANIFEST_FILE = "manifest.json"


class IndexNotFoundError(FileNotFoundError):
    """Raised when no index has been built yet."""


class IndexMismatchError(RuntimeError):
    """Raised when an index was built with a different embedding model."""


@dataclass
class Manifest:
    """How an index was built. Written next to the index files."""

    embedding_model: str
    dimension: int
    chunk_strategy: str
    chunk_size: int
    chunk_overlap: int
    chunk_count: int
    pdf_name: str
    pdf_sha256: str
    index_backend: str
    built_at: str  # ISO 8601, UTC

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / MANIFEST_FILE).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "Manifest":
        path = directory / MANIFEST_FILE
        if not path.is_file():
            raise IndexNotFoundError(f"No index found in {directory}. Build one with: rag-demo ingest")
        data = json.loads(path.read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def check_embedding_model(self, embedding_model: str) -> None:
        """Refuse to use this index with vectors from a different model."""
        if embedding_model != self.embedding_model:
            raise IndexMismatchError(
                f"This index was built with '{self.embedding_model}', but the current embedding model is "
                f"'{embedding_model}'. Vectors from different embedding models live in different "
                "spaces, so comparing them gives meaningless results. Rebuild the index with "
                "'rag-demo ingest', or set EMBED_MODEL back to the original model."
            )


class VectorIndex(ABC):
    """Stores chunk vectors and finds the chunks nearest to a query vector."""

    @abstractmethod
    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        """Add vectors (one row per chunk) and their chunks."""

    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        """Return the k most similar chunks, best first, with cosine similarity scores."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Write the index to a directory."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "VectorIndex":
        """Read an index back from a directory."""

    @abstractmethod
    def __len__(self) -> int: ...


class NumpyIndex(VectorIndex):
    """The simplest possible vector index: a matrix in memory, searched exhaustively."""

    VECTORS_FILE = "vectors.npy"
    CHUNKS_FILE = "chunks.json"

    def __init__(self) -> None:
        self.vectors: np.ndarray | None = None
        self.chunks: list[Chunk] = []

    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        if len(vectors) != len(chunks):
            raise ValueError(f"Got {len(vectors)} vectors but {len(chunks)} chunks.")
        vectors = np.asarray(vectors, dtype=np.float32)
        if self.vectors is not None and vectors.shape[1] != self.vectors.shape[1]:
            raise ValueError(f"Vector dimension {vectors.shape[1]} does not match index ({self.vectors.shape[1]}).")
        self.vectors = vectors if self.vectors is None else np.vstack([self.vectors, vectors])
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        if self.vectors is None or k <= 0:
            return []
        scores = self.vectors @ query_vector  # cosine similarity with every chunk at once
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]  # the k best, unordered
        top = top[np.argsort(-scores[top])]  # ...then sorted best first
        return [(self.chunks[i], float(scores[i])) for i in top]

    def save(self, path: Path) -> None:
        if self.vectors is None:
            raise ValueError("Cannot save an empty index.")
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / self.VECTORS_FILE, self.vectors)
        (path / self.CHUNKS_FILE).write_text(
            json.dumps([asdict(c) for c in self.chunks], ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "NumpyIndex":
        vectors_path, chunks_path = path / cls.VECTORS_FILE, path / cls.CHUNKS_FILE
        if not (vectors_path.is_file() and chunks_path.is_file()):
            raise IndexNotFoundError(f"No NumPy index found in {path}. Build one with: rag-demo ingest")
        index = cls()
        chunks = [Chunk(**c) for c in json.loads(chunks_path.read_text(encoding="utf-8"))]
        index.add(np.load(vectors_path), chunks)
        return index

    def __len__(self) -> int:
        return len(self.chunks)


BACKENDS: dict[str, type[VectorIndex]] = {"numpy": NumpyIndex}


def open_index(path: Path, embedding_model: str) -> tuple[VectorIndex, Manifest]:
    """Load the index at `path`, after checking it was built with `embedding_model`."""
    manifest = Manifest.load(path)
    manifest.check_embedding_model(embedding_model)
    index = BACKENDS[manifest.index_backend].load(path)
    if len(index) != manifest.chunk_count:
        raise IndexMismatchError(
            f"The manifest lists {manifest.chunk_count} chunks but the index holds {len(index)}. "
            "The index files are inconsistent; rebuild with 'rag-demo ingest'."
        )
    return index, manifest
