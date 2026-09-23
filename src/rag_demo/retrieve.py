"""Stage 5 of the RAG pipeline: retrieval (question -> most relevant chunks).

This is the "R" in RAG. The question is embedded with the *same* model that
embedded the chunks, so it lands in the same vector space. The index then
compares it with every chunk vector and returns the k closest.

Because all vectors have unit length, "closest" is measured by cosine
similarity, computed as a dot product: 1.0 means pointing the same way
(same meaning), around 0 means unrelated. Scores are relative, not absolute:
with all-MiniLM-L6-v2 a good match on this book scores around 0.6-0.75, while
the median chunk scores around 0.2 for any question.

The retrieved chunks are what Claude will later be shown as context, so the
quality of the final answer can never be better than the quality of this step.
"""

from dataclasses import dataclass
from pathlib import Path

from rag_demo import config
from rag_demo.chunk import Chunk
from rag_demo.embed import Embedder, embedder_name, get_embedder
from rag_demo.index import Manifest, VectorIndex, open_index


@dataclass
class Result:
    chunk: Chunk
    score: float  # cosine similarity between question and chunk


class Retriever:
    """An index plus the embedder that built it, ready to answer queries."""

    def __init__(self, index: VectorIndex, manifest: Manifest, embedder: Embedder):
        self.index = index
        self.manifest = manifest
        self.embedder = embedder

    @classmethod
    def open(cls, index_dir: Path = config.INDEX_DIR, embedder_kind: str = config.EMBEDDER) -> "Retriever":
        """Load the saved index and a matching embedder.

        The index manifest is checked against the embedder's name *before* the
        model is loaded, so a mismatch fails fast with a clear explanation.
        """
        index, manifest = open_index(index_dir, embedder_name(embedder_kind))
        return cls(index, manifest, get_embedder(embedder_kind))

    def retrieve(self, question: str, k: int = config.TOP_K) -> list[Result]:
        """Return the k chunks most similar to the question, best first."""
        query_vector = self.embedder.embed_query(question)
        return [Result(chunk, score) for chunk, score in self.index.search(query_vector, k)]
