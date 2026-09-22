"""Stage 2 of the RAG pipeline: chunking (text -> chunks with metadata).

A whole book is far too long to embed as one vector or to paste into a prompt,
so it is cut into chunks. Each chunk becomes one searchable unit: it is
embedded, stored in the index, and (if retrieved) shown to Claude as context.

Chunk size is a trade-off. Small chunks give precise matches but may cut an
idea in half; large chunks keep more context but blur several topics into one
vector. Two strategies are provided so the difference can be compared:

- "fixed": a window of CHUNK_SIZE characters slides along the text, stepping
  by CHUNK_SIZE - CHUNK_OVERLAP. The overlap means a sentence cut at one
  chunk's edge still appears whole in its neighbour. Breaks are moved to the
  nearest whitespace so words are never split.
- "paragraph": the text is split on blank lines, small paragraphs are merged
  up to CHUNK_SIZE, and any paragraph longer than CHUNK_SIZE is split with the
  fixed-size window. Chunks follow the author's own structure; no overlap.

Every chunk records where it came from (character offsets and page numbers),
so answers can later cite their sources.
"""

import re
from dataclasses import dataclass

from rag_demo.extract import Page

STRATEGIES = ("fixed", "paragraph")


@dataclass
class Chunk:
    id: int
    text: str
    pages: list[int]  # every page this chunk's text comes from
    start_char: int  # offset of the chunk in the whole-document text
    end_char: int  # exclusive
    length: int
    strategy: str


@dataclass
class Document:
    """All page texts joined into one string, plus where each page sits in it."""

    text: str
    page_spans: list[tuple[int, int, int]]  # (page_number, start_char, end_char)

    def pages_for(self, start: int, end: int) -> list[int]:
        return [n for n, s, e in self.page_spans if s < end and start < e]


# A page that ends without closing its sentence, followed by one that starts
# in lower case, is one sentence broken by the page break.
_ENDS_SENTENCE = re.compile(r"[.!?:”\"’)]$")


def build_document(pages: list[Page]) -> Document:
    """Join page texts into one document, remembering each page's character span.

    Pages are separated by a blank line (a paragraph break), except where a
    sentence runs across the page break: then a single space keeps it whole.
    """
    parts: list[str] = []
    spans: list[tuple[int, int, int]] = []
    position = 0
    for page in pages:
        if not page.text:
            continue
        if parts:
            runs_on = not _ENDS_SENTENCE.search(parts[-1]) and page.text[0].islower()
            separator = " " if runs_on else "\n\n"
            parts.append(separator)
            position += len(separator)
        parts.append(page.text)
        spans.append((page.page_number, position, position + len(page.text)))
        position += len(page.text)
    return Document(text="".join(parts), page_spans=spans)


def _skip_whitespace(text: str, i: int, end: int) -> int:
    while i < end and text[i].isspace():
        i += 1
    return i


def _fixed_spans(text: str, start: int, end: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """Slide a window of `size` characters over text[start:end].

    Each window ends at a word boundary, and the next window starts at a word
    boundary about `overlap` characters before the previous one ended.
    """
    spans = []
    s = _skip_whitespace(text, start, end)
    while s < end:
        e = min(s + size, end)
        if e < end:
            # Pull the break back to the last whitespace, but not past half the window.
            for i in range(e, s + size // 2, -1):
                if text[i].isspace():
                    e = i
                    break
        while e > s and text[e - 1].isspace():
            e -= 1
        spans.append((s, e))

        if _skip_whitespace(text, e, end) >= end:
            break
        # Step back by the overlap, then forward to the start of a word.
        nxt = max(e - overlap, s + 1)
        while nxt < e and not text[nxt - 1].isspace():
            nxt += 1
        s = _skip_whitespace(text, nxt, end)
    return spans


def _paragraph_spans(text: str, size: int) -> list[tuple[int, int]]:
    """Split on blank lines, merge small paragraphs up to `size`, split large ones."""
    paragraphs = []
    position = 0
    for m in re.finditer(r"\n\s*\n", text):
        paragraphs.append((position, m.start()))
        position = m.end()
    paragraphs.append((position, len(text)))

    spans: list[tuple[int, int]] = []
    current: tuple[int, int] | None = None
    for p_start, p_end in paragraphs:
        length = p_end - p_start
        if length > size:
            if current:
                spans.append(current)
                current = None
            # Split into equal-sized pieces rather than full windows plus a tiny
            # leftover ("meat."). The slack absorbs pulling breaks back to whitespace.
            pieces = -(-length // size)  # ceiling division
            window = min(size, -(-length // pieces) + 40)
            spans.extend(_fixed_spans(text, p_start, p_end, window, overlap=0))
        elif current and p_end - current[0] <= size:
            current = (current[0], p_end)
        else:
            if current:
                spans.append(current)
            current = (p_start, p_end)
    if current:
        spans.append(current)
    return [(s, e) for s, e in spans if text[s:e].strip()]


def chunk_document(doc: Document, strategy: str, size: int, overlap: int) -> list[Chunk]:
    """Cut the document into chunks with the chosen strategy."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown chunk strategy {strategy!r}; choose from {', '.join(STRATEGIES)}.")
    if size <= 0:
        raise ValueError("Chunk size must be positive.")
    if not 0 <= overlap < size:
        raise ValueError("Chunk overlap must be at least 0 and smaller than the chunk size.")

    if strategy == "fixed":
        spans = _fixed_spans(doc.text, 0, len(doc.text), size, overlap)
    else:
        spans = _paragraph_spans(doc.text, size)

    return [
        Chunk(
            id=i,
            text=doc.text[s:e],
            pages=doc.pages_for(s, e),
            start_char=s,
            end_char=e,
            length=e - s,
            strategy=strategy,
        )
        for i, (s, e) in enumerate(spans)
    ]


def chunk_pages(pages: list[Page], strategy: str, size: int, overlap: int) -> list[Chunk]:
    """Convenience wrapper: pages -> document -> chunks."""
    return chunk_document(build_document(pages), strategy, size, overlap)
