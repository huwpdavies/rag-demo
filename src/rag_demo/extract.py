"""Stage 1 of the RAG pipeline: extraction (PDF -> text, page by page).

Before anything can be retrieved, the document has to become plain text. We
read the PDF one page at a time and keep the page number with each page's text,
so that later stages can cite *where* an answer came from.

PDF text is messy: a PDF stores positioned glyphs, not sentences. This module
does a small amount of cleaning so that chunks read like prose:

- Words separated by tabs or runs of spaces are collapsed to single spaces.
- Lines wrapped by the page layout are joined back into paragraphs.
- A word hyphenated across a line break is rejoined. In this kind of ebook,
  line-end hyphens are real compound words ("ice-\\ncold"), so the hyphen is kept.
- Paragraph breaks are kept as blank lines, for the paragraph-aware chunker.

Pages with no extractable text are reported, because they usually mean a
scanned or image-only page (OCR is out of scope for this demo).
"""

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from rag_demo import config


@dataclass
class Page:
    """The cleaned text of one PDF page. Page numbers are 1-based, as printed."""

    page_number: int
    text: str

    @property
    def is_empty(self) -> bool:
        return not self.text


def resolve_pdf_path(pdf: Path | None = None) -> Path:
    """Return the PDF to use: the one given, or the single PDF in data/."""
    if pdf is not None:
        if not pdf.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf}")
        return pdf

    pdfs = sorted(config.DATA_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF found. Put one PDF in {config.DATA_DIR}.")
    if len(pdfs) > 1:
        names = ", ".join(p.name for p in pdfs)
        raise ValueError(f"Several PDFs in {config.DATA_DIR} ({names}). Choose one with --pdf.")
    return pdfs[0]


# A line that starts with indentation or a bullet may open a new paragraph.
_INDENTED = re.compile(r"^(\s{2,}|\s*•)")
# ...but only if the previous line finished a sentence. This stops hanging
# indents (bulleted lists, recipe steps) from splitting a sentence in two.
_SENTENCE_END = re.compile(r"[.!?:”\"]$")
# A line ending in "word-" whose next line starts with a letter: a word split
# across the line break.
_HYPHEN_END = re.compile(r"[^\W\d_]-$")


def clean_page_text(raw: str) -> str:
    """Turn one page of layout-extracted PDF text into clean paragraphs.

    Paragraphs are separated by a blank line; lines within a paragraph are
    joined with single spaces.
    """
    paragraphs: list[list[str]] = []
    current: list[str] = []
    previous = ""

    for line in raw.replace("\t", " ").replace(" ", " ").splitlines():
        line = line.rstrip()
        starts_paragraph = (
            not line
            or line.lstrip().startswith("•")
            or (_INDENTED.match(line) and _SENTENCE_END.search(previous))
        )
        if starts_paragraph and current:
            paragraphs.append(current)
            current = []
        if line:
            current.append(line.strip())
        previous = line
    if current:
        paragraphs.append(current)

    return "\n\n".join(_join_lines(p) for p in paragraphs)


def _join_lines(lines: list[str]) -> str:
    """Join the wrapped lines of one paragraph, repairing line-end hyphenation."""
    text = lines[0]
    for line in lines[1:]:
        if _HYPHEN_END.search(text) and line[:1].isalpha():
            text += line  # "ice-" + "cold" -> "ice-cold"
        else:
            text += " " + line
    return re.sub(r" {2,}", " ", text)


def iter_pages(pdf_path: Path) -> Iterator[Page]:
    """Yield each page of the PDF as a cleaned Page, in order.

    pypdf's "layout" mode is used rather than its default mode: it keeps styled
    words (italics, links) on the same line as their sentence, and it preserves
    the indentation that marks the start of each paragraph.
    """
    reader = PdfReader(pdf_path)
    for number, page in enumerate(reader.pages, start=1):
        raw = page.extract_text(extraction_mode="layout") or ""
        yield Page(page_number=number, text=clean_page_text(raw))


def page_count(pdf_path: Path) -> int:
    return len(PdfReader(pdf_path).pages)


def extract_pages(
    pdf_path: Path, on_page: Callable[[Page], None] | None = None
) -> list[Page]:
    """Extract every page of the PDF. `on_page` is called after each page (for progress)."""
    pages = []
    for page in iter_pages(pdf_path):
        pages.append(page)
        if on_page:
            on_page(page)
    return pages
