"""Retrieval evaluation: does the search find the right part of the book?

A RAG answer can only be as good as the chunks it is given, so retrieval is
worth measuring on its own. eval/questions.json lists questions written from
the PDF, each with the page(s) that hold the answer.

For each question we retrieve the top chunks and look for the first one that
comes from an answer page. "Hit rate at k" is the share of questions where
such a chunk appears in the top k results:

- hit@1: the best match was right;
- hit@5: the right passage was somewhere in the five chunks Claude would see.

Re-running with different chunking settings (a temporary index is built in
memory) shows how those choices change retrieval quality. No API calls are made.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from rag_demo.chunk import Chunk
from rag_demo.embed import Embedder
from rag_demo.index import NumpyIndex, VectorIndex

HIT_RATE_KS = (1, 3, 5)


@dataclass
class EvalQuestion:
    question: str
    pages: list[int]  # PDF pages where the answer is found
    answer: str = ""  # for the reader; not used in scoring


@dataclass
class QuestionResult:
    item: EvalQuestion
    first_hit_rank: int | None  # 1-based rank of the first chunk from an answer page
    top_pages: list[int]  # pages of the best-ranked chunk


def load_questions(path: Path) -> list[EvalQuestion]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data["questions"] if isinstance(data, dict) else data
    return [EvalQuestion(q["question"], [int(p) for p in q["pages"]], q.get("answer", "")) for q in items]


def evaluate(index: VectorIndex, embedder: Embedder, questions: list[EvalQuestion]) -> list[QuestionResult]:
    """Retrieve the top chunks for each question and find where the first hit ranks."""
    k = max(HIT_RATE_KS)
    results = []
    for item in questions:
        hits = index.search(embedder.embed_query(item.question), k)
        answer_pages = set(item.pages)
        rank = next(
            (i for i, (chunk, _) in enumerate(hits, start=1) if answer_pages & set(chunk.pages)),
            None,
        )
        results.append(QuestionResult(item, rank, hits[0][0].pages if hits else []))
    return results


def hit_rate(results: list[QuestionResult], k: int) -> float:
    """Share of questions with a chunk from an answer page in the top k."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.first_hit_rank is not None and r.first_hit_rank <= k) / len(results)


def build_index(chunks: list[Chunk], embedder: Embedder, on_batch=None) -> NumpyIndex:
    """Embed chunks into a temporary, in-memory index (nothing is written to disk)."""
    index = NumpyIndex()
    index.add(embedder.embed_documents([c.text for c in chunks], on_batch=on_batch), chunks)
    return index
