"""NumPy index and manifest tests, using small hand-made vectors (no embedding model)."""

import numpy as np
import pytest

from rag_demo.chunk import Chunk
from rag_demo.index import IndexMismatchError, IndexNotFoundError, Manifest, NumpyIndex, open_index


def make_chunk(i: int) -> Chunk:
    text = f"chunk {i}"
    return Chunk(id=i, text=text, pages=[i + 1], start_char=0, end_char=len(text), length=len(text), strategy="fixed")


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def index() -> NumpyIndex:
    # Four chunks pointing along the four axes of a 4-dimensional space.
    idx = NumpyIndex()
    idx.add(np.eye(4, dtype=np.float32), [make_chunk(i) for i in range(4)])
    return idx


def make_manifest(model="model-a", chunk_count=4) -> Manifest:
    return Manifest(
        embedding_model=model, dimension=4, chunk_strategy="fixed", chunk_size=800, chunk_overlap=150,
        chunk_count=chunk_count, pdf_name="test.pdf", pdf_sha256="0" * 64, index_backend="numpy",
        built_at=Manifest.now(),
    )


def test_known_vectors_return_known_nearest_neighbours(index):
    results = index.search(unit([0.1, 0.2, 0.9, 0.3]), k=4)
    assert [c.id for c, _ in results] == [2, 3, 1, 0]
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == pytest.approx(unit([0.1, 0.2, 0.9, 0.3])[2])


def test_exact_match_scores_one(index):
    (chunk, score), *_ = index.search(unit([0, 1, 0, 0]), k=1)
    assert chunk.id == 1
    assert score == pytest.approx(1.0)


def test_k_larger_than_index_returns_everything(index):
    assert len(index.search(unit([1, 1, 1, 1]), k=10)) == 4


def test_save_load_round_trip(index, tmp_path):
    index.save(tmp_path)
    loaded = NumpyIndex.load(tmp_path)
    assert np.array_equal(loaded.vectors, index.vectors)
    assert loaded.chunks == index.chunks
    query = unit([0.5, 0.1, 0.1, 0.8])
    assert loaded.search(query, 4) == index.search(query, 4)


def test_add_rejects_mismatched_vectors_and_chunks():
    with pytest.raises(ValueError):
        NumpyIndex().add(np.eye(3, dtype=np.float32), [make_chunk(0)])


def test_loading_with_a_different_embedding_model_raises_a_clear_error(index, tmp_path):
    index.save(tmp_path)
    make_manifest(model="model-a").save(tmp_path)
    with pytest.raises(IndexMismatchError) as err:
        open_index(tmp_path, embedding_model="model-b")
    message = str(err.value)
    assert "model-a" in message and "model-b" in message
    assert "rag-demo index" in message


def test_loading_with_the_same_embedding_model_works(index, tmp_path):
    index.save(tmp_path)
    make_manifest(model="model-a").save(tmp_path)
    loaded, manifest = open_index(tmp_path, embedding_model="model-a")
    assert len(loaded) == 4 and manifest.embedding_model == "model-a"


def test_manifest_chunk_count_must_match_index(index, tmp_path):
    index.save(tmp_path)
    make_manifest(chunk_count=5).save(tmp_path)
    with pytest.raises(IndexMismatchError):
        open_index(tmp_path, embedding_model="model-a")


def test_missing_index_raises_not_found(tmp_path):
    with pytest.raises(IndexNotFoundError):
        open_index(tmp_path, embedding_model="model-a")
