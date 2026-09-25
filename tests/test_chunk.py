"""Chunker tests, on synthetic text so they run in milliseconds without the PDF."""

import random

import pytest

from rag_demo.chunk import build_document, chunk_document, chunk_pages
from rag_demo.extract import Page

WORDS = "salt fat acid heat butter flour water brine roast simmer season taste".split()


def make_pages(n_pages=6, paragraphs_per_page=4, seed=0) -> list[Page]:
    rng = random.Random(seed)
    pages = []
    for number in range(1, n_pages + 1):
        paragraphs = []
        for _ in range(paragraphs_per_page):
            sentences = [" ".join(rng.choices(WORDS, k=rng.randint(5, 14))).capitalize() + "." for _ in range(rng.randint(1, 8))]
            paragraphs.append(" ".join(sentences))
        pages.append(Page(page_number=number, text="\n\n".join(paragraphs)))
    return pages


def rebuild(doc_text, chunks):
    """Concatenate chunks minus their overlaps; gaps between chunks may only be whitespace."""
    parts, position = [], 0
    for c in chunks:
        if c.start_char >= position:
            assert not doc_text[position:c.start_char].strip(), "text was skipped between chunks"
            parts.append(doc_text[position:c.start_char] + c.text)
        else:
            parts.append(c.text[position - c.start_char :])
        position = c.end_char
    parts.append(doc_text[position:])
    return "".join(parts)


@pytest.mark.parametrize("size, overlap", [(800, 150), (300, 50), (200, 0)])
def test_fixed_overlap_length_is_correct(size, overlap):
    doc = build_document(make_pages())
    chunks = chunk_document(doc, "fixed", size, overlap)
    longest_word = max(len(w) for w in WORDS) + 1  # a word plus its trailing full stop
    for prev, nxt in zip(chunks, chunks[1:]):
        shared = prev.end_char - nxt.start_char
        # Overlap snaps forward to a word boundary, so it can fall short by at most one word.
        assert overlap - longest_word <= shared <= overlap
        if shared > 0:
            assert prev.text[-shared:] == nxt.text[:shared]


@pytest.mark.parametrize("strategy, size, overlap", [("fixed", 800, 150), ("fixed", 250, 40), ("paragraph", 800, 0), ("paragraph", 150, 0)])
def test_chunks_minus_overlaps_reproduce_source(strategy, size, overlap):
    doc = build_document(make_pages())
    chunks = chunk_document(doc, strategy, size, overlap)
    assert rebuild(doc.text, chunks) == doc.text


@pytest.mark.parametrize("strategy, size, overlap", [("fixed", 800, 150), ("paragraph", 800, 0), ("paragraph", 120, 0)])
def test_no_empty_or_oversized_chunks(strategy, size, overlap):
    chunks = chunk_pages(make_pages(), strategy, size, overlap)
    assert chunks
    for c in chunks:
        assert c.text.strip() and c.text == c.text.strip()
        assert c.length == len(c.text) <= size
        assert c.strategy == strategy


def test_no_words_are_split():
    doc = build_document(make_pages())
    for c in chunk_document(doc, "fixed", 300, 60):
        assert c.start_char == 0 or doc.text[c.start_char - 1].isspace()
        assert c.end_char == len(doc.text) or doc.text[c.end_char].isspace()


def test_chunk_spanning_a_page_break_lists_both_pages():
    pages = [Page(1, "The first page ends here."), Page(2, "The second page starts here.")]
    (chunk,) = chunk_pages(pages, "fixed", 800, 0)
    assert chunk.pages == [1, 2]


def test_sentence_running_across_a_page_break_is_joined_with_a_space():
    doc = build_document([Page(1, "Brine the turkey"), Page(2, "overnight in the fridge.")])
    assert doc.text == "Brine the turkey overnight in the fridge."


def test_empty_pages_are_skipped():
    doc = build_document([Page(1, ""), Page(2, "Text.")])
    assert doc.text == "Text."
    assert doc.page_spans == [(2, 0, 5)]


@pytest.mark.parametrize("strategy, size, overlap", [("words", 800, 0), ("fixed", 0, 0), ("fixed", 100, 100), ("fixed", 100, -1)])
def test_invalid_settings_raise(strategy, size, overlap):
    with pytest.raises(ValueError):
        chunk_pages(make_pages(), strategy, size, overlap)
