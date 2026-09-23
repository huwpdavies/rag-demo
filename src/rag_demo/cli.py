"""Command-line interface: one command per stage of the RAG pipeline.

Each command runs a single stage and prints what it produced, so the pipeline
can be demonstrated step by step. Commands are added phase by phase.
"""

import statistics
import sys
import time
from pathlib import Path

import typer
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from rag_demo import chunk as chunking, config, extract as extraction

# PDFs are full of curly quotes and dashes; make sure redirected output on
# Windows doesn't choke on them.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

app = typer.Typer(
    help="A step-by-step Retrieval-Augmented Generation demo over a single PDF.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.command()
def check() -> None:
    """Check that API keys load from .env (shows only 'found' or 'missing')."""
    table = Table(title="API keys", show_header=True)
    table.add_column("Key")
    table.add_column("Status")
    table.add_column("Used for")

    def status(present: bool) -> str:
        return "[green]found[/green]" if present else "[red]missing[/red]"

    table.add_row(
        "ANTHROPIC_API_KEY",
        status(config.anthropic_api_key() is not None),
        "generation (required)",
    )
    table.add_row(
        "VOYAGE_API_KEY",
        status(config.voyage_api_key() is not None),
        "embeddings (optional, only when EMBEDDER=voyage)",
    )
    console.print(table)

    try:
        config.require_anthropic_api_key()
    except config.MissingAPIKeyError as err:
        console.print(f"[bold red]Error:[/bold red] {err}")
        raise typer.Exit(code=1)

    console.print("[green]Ready.[/green] The Anthropic key is configured.")


PdfOption = typer.Option(None, "--pdf", help="PDF to use (default: the single PDF in data/).")


def _resolve_pdf(pdf: Path | None) -> Path:
    try:
        return extraction.resolve_pdf_path(pdf)
    except (FileNotFoundError, ValueError) as err:
        console.print(f"[bold red]Error:[/bold red] {err}")
        raise typer.Exit(code=1)


def _extract_with_progress(pdf_path: Path, total: int) -> list[extraction.Page]:
    with Progress(console=console, transient=True) as progress:
        task = progress.add_task("Extracting pages", total=total)
        return extraction.extract_pages(pdf_path, on_page=lambda _: progress.advance(task))


@app.command()
def extract(
    pdf: Path = PdfOption,
    page: int = typer.Option(None, "--page", help="Page to preview (default: first page with text)."),
    show_all: bool = typer.Option(False, "--all", help="List characters for every page."),
    preview_chars: int = typer.Option(600, "--preview-chars", help="Length of the page preview."),
) -> None:
    """Stage 1: extract text from the PDF, page by page."""
    pdf_path = _resolve_pdf(pdf)
    total = extraction.page_count(pdf_path)
    if page is not None and not 1 <= page <= total:
        console.print(f"[bold red]Error:[/bold red] --page must be between 1 and {total}.")
        raise typer.Exit(code=1)

    console.print(f"Reading [bold]{pdf_path.name}[/bold]")
    pages = _extract_with_progress(pdf_path, total)

    lengths = [len(p.text) for p in pages]
    with_text = [n for n in lengths if n]
    summary = Table.grid(padding=(0, 2))
    summary.add_row("Pages", f"{len(pages)}")
    summary.add_row("Pages with text", f"{len(with_text)}")
    summary.add_row("Total characters", f"{sum(lengths):,}")
    if with_text:
        summary.add_row(
            "Characters per page (with text)",
            f"min {min(with_text):,} · median {int(statistics.median(with_text)):,} · max {max(with_text):,}",
        )
    console.print(summary)

    # Characters per page, as a compact grid.
    shown = pages if show_all else pages[:30]
    cells = [
        f"[dim]p{p.page_number:>3}[/dim] " + (f"{len(p.text):>5,}" if p.text else "[yellow]empty[/yellow]")
        for p in shown
    ]
    console.print()
    console.print("[bold]Characters per page[/bold]")
    console.print(Columns(cells, padding=(0, 3), column_first=True))
    if len(shown) < len(pages):
        console.print(f"[dim]... {len(pages) - len(shown)} more pages (use --all to list every page)[/dim]")

    empty = [p.page_number for p in pages if p.is_empty]
    if empty:
        console.print()
        console.print(
            f"[bold yellow]Warning:[/bold yellow] {len(empty)} page(s) returned no text: "
            f"{', '.join(map(str, empty))}"
        )
        console.print(
            "[yellow]These are probably images (illustrations or scanned pages). "
            "OCR is out of scope, so they will not be searchable.[/yellow]"
        )

    if page is not None:
        target = pages[page - 1]
    else:
        target = next((p for p in pages if p.text), None)
    if target is None:
        return
    preview = Text(target.text[:preview_chars])  # Text, not markup: book text may contain [brackets]
    if len(target.text) > preview_chars:
        preview.append(" …")
    console.print()
    console.print(
        Panel(
            preview if target.text else "[yellow](no text on this page)[/yellow]",
            title=f"Preview: page {target.page_number}",
            subtitle=f"{len(target.text):,} characters",
        )
    )


def _load_pages(pdf_path: Path) -> list[extraction.Page]:
    """Pages from the extraction cache if it is valid, otherwise extract now."""
    pages = extraction.load_cached_pages(pdf_path)
    if pages is not None:
        console.print(f"[dim]Using cached extraction of {pdf_path.name}[/dim]")
        return pages
    console.print(f"Extracting [bold]{pdf_path.name}[/bold] (cached for next time)")
    return _extract_with_progress(pdf_path, extraction.page_count(pdf_path))


def _histogram(values: list[int], bins: int = 10, width: int = 40) -> Table:
    """A text histogram of chunk lengths, with bins spanning the observed range."""
    low, high = min(values), max(values)
    step = max(1, -(-(high - low + 1) // bins))  # ceiling division
    counts = [0] * bins
    for v in values:
        counts[min((v - low) // step, bins - 1)] += 1
    table = Table.grid(padding=(0, 1))
    for i, n in enumerate(counts):
        bin_low = low + i * step
        if bin_low > high:
            break
        bar = "█" * max(1, round(width * n / max(counts))) if n else ""
        table.add_row(f"{bin_low:>5,}–{min(bin_low + step - 1, high):<5,}", f"[cyan]{bar}[/cyan]", f"{n:,}")
    return table


PREV_OVERLAP_STYLE = "black on yellow"
NEXT_OVERLAP_STYLE = "black on bright_cyan"


StrategyOption = typer.Option(config.CHUNK_STRATEGY, "--strategy", help="fixed or paragraph.")
ChunkSizeOption = typer.Option(config.CHUNK_SIZE, "--chunk-size", help="Maximum characters per chunk.")
OverlapOption = typer.Option(
    config.CHUNK_OVERLAP, "--overlap", help="Characters shared by neighbouring chunks (fixed only)."
)


def _load_chunks(pdf_path: Path, strategy: str, chunk_size: int, overlap: int) -> list[chunking.Chunk]:
    """Stages 1-2: pages (cached if possible) -> chunks. Exits on bad settings or no text."""
    try:
        pages = _load_pages(pdf_path)
        chunks = chunking.chunk_pages(pages, strategy, chunk_size, overlap)
    except ValueError as err:
        console.print(f"[bold red]Error:[/bold red] {err}")
        raise typer.Exit(code=1)
    if not chunks:
        console.print("[yellow]No chunks: the PDF has no extractable text.[/yellow]")
        raise typer.Exit(code=1)
    return chunks


@app.command()
def chunk(
    strategy: str = StrategyOption,
    chunk_size: int = ChunkSizeOption,
    overlap: int = OverlapOption,
    sample_from: int = typer.Option(None, "--sample-from", help="ID of the first sample chunk (default: middle)."),
    pdf: Path = PdfOption,
) -> None:
    """Stage 2: cut the text into chunks and show how they overlap."""
    pdf_path = _resolve_pdf(pdf)
    if strategy == "paragraph":
        overlap = 0
    chunks = _load_chunks(pdf_path, strategy, chunk_size, overlap)

    lengths = [c.length for c in chunks]
    summary = Table.grid(padding=(0, 2))
    summary.add_row("Strategy", strategy)
    summary.add_row("Chunk size", f"{chunk_size:,} characters")
    summary.add_row("Overlap", f"{overlap:,} characters" if strategy == "fixed" else "none (paragraph strategy)")
    summary.add_row("Chunks", f"{len(chunks):,}")
    summary.add_row(
        "Chunk length",
        f"min {min(lengths):,} · median {int(statistics.median(lengths)):,} · max {max(lengths):,}",
    )
    console.print(summary)

    console.print()
    console.print("[bold]Chunk length histogram[/bold] (characters → number of chunks)")
    console.print(_histogram(lengths))

    # Three consecutive sample chunks, with shared (overlapping) text highlighted.
    first = len(chunks) // 2 if sample_from is None else sample_from
    first = max(0, min(first, len(chunks) - 3))
    console.print()
    if strategy == "fixed":
        legend = Text("Sample chunks. ")
        legend.append("Shared with previous chunk", style=PREV_OVERLAP_STYLE)
        legend.append("  ")
        legend.append("Shared with next chunk", style=NEXT_OVERLAP_STYLE)
        console.print(legend)
    else:
        console.print("Sample chunks. Paragraph chunks do not overlap; blank lines mark merged paragraphs.")

    for i in range(first, min(first + 3, len(chunks))):
        c = chunks[i]
        body = Text(c.text)
        if i > 0 and chunks[i - 1].end_char > c.start_char:
            body.stylize(PREV_OVERLAP_STYLE, 0, chunks[i - 1].end_char - c.start_char)
        if i + 1 < len(chunks) and chunks[i + 1].start_char < c.end_char:
            body.stylize(NEXT_OVERLAP_STYLE, chunks[i + 1].start_char - c.start_char, c.length)
        pages_label = "page " + str(c.pages[0]) if len(c.pages) == 1 else f"pages {c.pages[0]}–{c.pages[-1]}"
        console.print(
            Panel(
                body,
                title=f"Chunk {c.id} · {pages_label}",
                subtitle=f"chars {c.start_char:,}–{c.end_char:,} · {c.length} long",
            )
        )


EmbedderOption = typer.Option(config.EMBEDDER, "--embedder", help="local or voyage.")


def _get_embedder(kind: str):
    from rag_demo import embed as embedding  # imports numpy; keep other commands fast

    try:
        with console.status(f"Loading the {kind} embedding model…"):
            return embedding.get_embedder(kind)
    except (ValueError, config.MissingAPIKeyError, embedding.EmbedderUnavailableError, ImportError) as err:
        console.print(f"[bold red]Error:[/bold red] {err}")
        raise typer.Exit(code=1)


def _embed_with_progress(embedder, chunks: list[chunking.Chunk]):
    columns = (TextColumn("Embedding chunks"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn())
    with Progress(*columns, console=console, transient=True) as progress:
        task = progress.add_task("embed", total=len(chunks))
        return embedder.embed_documents([c.text for c in chunks], on_batch=lambda n: progress.advance(task, n))


@app.command()
def embed(
    strategy: str = StrategyOption,
    chunk_size: int = ChunkSizeOption,
    overlap: int = OverlapOption,
    embedder_kind: str = EmbedderOption,
    show: int = typer.Option(0, "--show", help="ID of the chunk whose vector is printed."),
    pdf: Path = PdfOption,
) -> None:
    """Stage 3: turn every chunk into a vector, and show what one looks like."""
    import numpy as np

    pdf_path = _resolve_pdf(pdf)
    if strategy == "paragraph":
        overlap = 0
    chunks = _load_chunks(pdf_path, strategy, chunk_size, overlap)
    if not 0 <= show < len(chunks):
        console.print(f"[bold red]Error:[/bold red] --show must be between 0 and {len(chunks) - 1}.")
        raise typer.Exit(code=1)

    embedder = _get_embedder(embedder_kind)
    started = time.perf_counter()
    try:
        vectors = _embed_with_progress(embedder, chunks)
    except Exception as err:  # e.g. network or auth errors from a hosted embedder
        console.print(f"[bold red]Embedding failed:[/bold red] {type(err).__name__}: {err}")
        raise typer.Exit(code=1)
    elapsed = time.perf_counter() - started

    rows, dims = vectors.shape
    summary = Table.grid(padding=(0, 2))
    summary.add_row("Model", embedder.name)
    summary.add_row("Chunks", f"{len(chunks):,} ({strategy}, {chunk_size} chars)")
    summary.add_row("Matrix shape", f"[bold]{rows:,} x {dims}[/bold]  (one row per chunk, one column per dimension)")
    summary.add_row("Time taken", f"{elapsed:.1f}s ({rows / elapsed:,.0f} chunks/s)")
    console.print(summary)

    c, v = chunks[show], vectors[show]
    snippet = c.text[:150] + ("…" if c.length > 150 else "")
    numbers = ", ".join(f"{x:+.4f}" for x in v[:8])
    console.print()
    console.print(
        Panel(
            Text.assemble(
                ("Text: ", "bold"), snippet, "\n\n",
                ("Vector: ", "bold"), f"[{numbers}, … {dims - 8} more]", "\n\n",
                ("Length: ", "bold"), f"{np.linalg.norm(v):.4f} (normalised to 1, so a dot product is a cosine similarity)",
            ),
            title=f"Chunk {c.id} as an embedding",
        )
    )


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.1f} {unit}"
        n /= 1024


@app.command()
def ingest(
    strategy: str = StrategyOption,
    chunk_size: int = ChunkSizeOption,
    overlap: int = OverlapOption,
    embedder_kind: str = EmbedderOption,
    fresh: bool = typer.Option(False, "--fresh", help="Re-extract the PDF even if a cached extraction exists."),
    pdf: Path = PdfOption,
) -> None:
    """Stages 1-4: extract, chunk, embed, and save the index (run this once per PDF/settings)."""
    from rag_demo import index as indexing

    pdf_path = _resolve_pdf(pdf)
    if strategy == "paragraph":
        overlap = 0
    started = time.perf_counter()

    console.rule("[bold]1. Extract")
    if fresh:
        pages = _extract_with_progress(pdf_path, extraction.page_count(pdf_path))
    else:
        pages = _load_pages(pdf_path)
    with_text = sum(1 for p in pages if p.text)
    console.print(f"{len(pages)} pages, {with_text} with text")

    console.rule("[bold]2. Chunk")
    try:
        chunks = chunking.chunk_pages(pages, strategy, chunk_size, overlap)
    except ValueError as err:
        console.print(f"[bold red]Error:[/bold red] {err}")
        raise typer.Exit(code=1)
    if not chunks:
        console.print("[yellow]No chunks: the PDF has no extractable text.[/yellow]")
        raise typer.Exit(code=1)
    console.print(f"{len(chunks):,} chunks ({strategy}, size {chunk_size}, overlap {overlap})")

    console.rule("[bold]3. Embed")
    embedder = _get_embedder(embedder_kind)
    embed_started = time.perf_counter()
    try:
        vectors = _embed_with_progress(embedder, chunks)
    except Exception as err:  # e.g. network or auth errors from a hosted embedder
        console.print(f"[bold red]Embedding failed:[/bold red] {type(err).__name__}: {err}")
        raise typer.Exit(code=1)
    console.print(
        f"{vectors.shape[0]:,} x {vectors.shape[1]} matrix with {embedder.name} "
        f"in {time.perf_counter() - embed_started:.1f}s"
    )

    console.rule("[bold]4. Index")
    index = indexing.NumpyIndex()
    index.add(vectors, chunks)
    # Remove any old manifest first, so a half-written index never looks valid.
    (config.INDEX_DIR / indexing.MANIFEST_FILE).unlink(missing_ok=True)
    index.save(config.INDEX_DIR)
    manifest = indexing.Manifest(
        embedding_model=embedder.name,
        dimension=int(vectors.shape[1]),
        chunk_strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        chunk_count=len(chunks),
        pdf_name=pdf_path.name,
        pdf_sha256=extraction.pdf_sha256(pdf_path),
        index_backend="numpy",
        built_at=indexing.Manifest.now(),
    )
    manifest.save(config.INDEX_DIR)  # written last: a manifest means the index is complete
    console.print(f"Saved to {config.INDEX_DIR}")
    console.print(f"[green]Done in {time.perf_counter() - started:.1f}s.[/green] Inspect it with: rag-demo index-info")


@app.command("index-info")
def index_info() -> None:
    """Show how the saved index was built (its manifest) and the files it uses."""
    from rag_demo import embed as embedding, index as indexing

    try:
        manifest = indexing.Manifest.load(config.INDEX_DIR)
    except indexing.IndexNotFoundError as err:
        console.print(f"[yellow]{err}[/yellow]")
        raise typer.Exit(code=1)

    table = Table(title="Index manifest", show_header=True)
    table.add_column("Field")
    table.add_column("Value")
    for field, value in vars(manifest).items():
        table.add_row(field, f"{value:,}" if isinstance(value, int) else str(value))
    console.print(table)

    files = Table(title="Index files", show_header=True)
    files.add_column("File")
    files.add_column("Size", justify="right")
    for name in (indexing.NumpyIndex.VECTORS_FILE, indexing.NumpyIndex.CHUNKS_FILE, indexing.MANIFEST_FILE):
        path = config.INDEX_DIR / name
        files.add_row(name, _human_size(path.stat().st_size) if path.is_file() else "[red]missing[/red]")
    console.print(files)
    console.print(f"[dim]in {config.INDEX_DIR}[/dim]")

    # Is the index still valid for the current PDF and embedder?
    current = embedding.embedder_name(config.EMBEDDER)
    if current == manifest.embedding_model:
        console.print(f"[green]✓[/green] Current embedder ({current}) matches the index.")
    else:
        console.print(
            f"[red]✗[/red] Current embedder is {current}, but the index was built with "
            f"{manifest.embedding_model}. Searches will be refused until you rebuild with 'rag-demo ingest'."
        )
    pdf_path = config.DATA_DIR / manifest.pdf_name
    if not pdf_path.is_file():
        console.print(f"[yellow]![/yellow] {manifest.pdf_name} is no longer in {config.DATA_DIR}.")
    elif extraction.pdf_sha256(pdf_path) == manifest.pdf_sha256:
        console.print(f"[green]✓[/green] {manifest.pdf_name} is unchanged since the index was built.")
    else:
        console.print(
            f"[yellow]![/yellow] {manifest.pdf_name} has changed since the index was built. "
            "Rebuild with 'rag-demo ingest'."
        )


@app.callback()
def main() -> None:
    """A step-by-step Retrieval-Augmented Generation demo over a single PDF."""


if __name__ == "__main__":
    app()
