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
from rich.console import Console, Group
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


def _get_embedder():
    from rag_demo import embed as embedding  # imports numpy; keep other commands fast

    try:
        with console.status(f"Loading the embedding model ({config.EMBED_MODEL})…"):
            return embedding.get_embedder()
    except (embedding.EmbedderUnavailableError, ImportError) as err:
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

    embedder = _get_embedder()
    started = time.perf_counter()
    try:
        vectors = _embed_with_progress(embedder, chunks)
    except Exception as err:  # e.g. running out of memory
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


@app.command("index")
def build_index(
    strategy: str = StrategyOption,
    chunk_size: int = ChunkSizeOption,
    overlap: int = OverlapOption,
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
    embedder = _get_embedder()
    embed_started = time.perf_counter()
    try:
        vectors = _embed_with_progress(embedder, chunks)
    except Exception as err:  # e.g. running out of memory
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
    current = embedding.embedder_name()
    if current == manifest.embedding_model:
        console.print(f"[green]✓[/green] Current embedding model ({current}) matches the index.")
    else:
        console.print(
            f"[red]✗[/red] Current embedding model is {current}, but the index was built with "
            f"{manifest.embedding_model}. Searches will be refused until you rebuild with 'rag-demo index'."
        )
    pdf_path = config.DATA_DIR / manifest.pdf_name
    if not pdf_path.is_file():
        console.print(f"[yellow]![/yellow] {manifest.pdf_name} is no longer in {config.DATA_DIR}.")
    elif extraction.pdf_sha256(pdf_path) == manifest.pdf_sha256:
        console.print(f"[green]✓[/green] {manifest.pdf_name} is unchanged since the index was built.")
    else:
        console.print(
            f"[yellow]![/yellow] {manifest.pdf_name} has changed since the index was built. "
            "Rebuild with 'rag-demo index'."
        )


def _open_retriever():
    """Load the saved index and its embedder, or exit with a clear message."""
    from rag_demo import embed as embedding, index as indexing
    from rag_demo.retrieve import Retriever

    try:
        with console.status("Loading the index and embedding model…"):
            return Retriever.open(config.INDEX_DIR)
    except indexing.IndexNotFoundError as err:
        console.print(f"[yellow]{err}[/yellow]")
    except (indexing.IndexMismatchError, embedding.EmbedderUnavailableError) as err:
        console.print(f"[bold red]Error:[/bold red] {err}")
    raise typer.Exit(code=1)


def _pages_label(pages: list[int]) -> str:
    return str(pages[0]) if len(pages) == 1 else f"{pages[0]}–{pages[-1]}"


@app.command()
def search(
    question: str = typer.Argument(..., help="The question to search for."),
    top_k: int = typer.Option(config.TOP_K, "--top-k", "-k", help="Number of chunks to return."),
    preview_chars: int = typer.Option(150, "--preview-chars", help="Characters of each chunk to show."),
) -> None:
    """Stage 5: find the chunks most similar to a question (no Claude involved)."""
    if top_k < 1:
        console.print("[bold red]Error:[/bold red] --top-k must be at least 1.")
        raise typer.Exit(code=1)
    retriever = _open_retriever()

    started = time.perf_counter()
    results = retriever.retrieve(question, top_k)
    elapsed_ms = (time.perf_counter() - started) * 1000

    table = Table(title=f"Top {len(results)} chunks for: “{question}”", show_header=True, show_lines=True)
    table.add_column("Rank", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Page(s)", justify="right", no_wrap=True)
    table.add_column("Chunk", justify="right")
    table.add_column(f"Text (first {preview_chars} characters)")
    for rank, r in enumerate(results, start=1):
        snippet = " ".join(r.chunk.text.split())  # flatten paragraph breaks for the table
        if len(snippet) > preview_chars:
            snippet = snippet[:preview_chars] + "…"
        table.add_row(str(rank), f"{r.score:.3f}", _pages_label(r.chunk.pages), str(r.chunk.id), Text(snippet))
    console.print(table)
    console.print(
        f"[dim]Searched {len(retriever.index):,} chunks with {retriever.manifest.embedding_model} "
        f"in {elapsed_ms:.0f} ms. Score = cosine similarity (1.0 = same direction, ~0 = unrelated).[/dim]"
    )


def _print_prompt(prompt) -> None:
    """Show the exact system prompt and user message, with a token estimate."""
    console.print(Panel(Text(prompt.system), title="System prompt", title_align="left", border_style="magenta"))
    console.print(Panel(Text(prompt.user), title="User message", title_align="left", border_style="cyan"))
    tokens = prompt.estimated_tokens
    from rag_demo.augment import CHARS_PER_TOKEN

    table = Table(title=f"Estimated input tokens (characters ÷ {CHARS_PER_TOKEN})", show_header=True)
    table.add_column("Part")
    table.add_column("Characters", justify="right")
    table.add_column("≈ Tokens", justify="right")
    table.add_row("System prompt", f"{len(prompt.system):,}", f"{tokens['system']:,}")
    table.add_row("User message", f"{len(prompt.user):,}", f"{tokens['user']:,}")
    table.add_row("[bold]Total[/bold]", f"{len(prompt.system) + len(prompt.user):,}", f"[bold]{tokens['total']:,}[/bold]")
    console.print(table)


def _answer_panel(answer, title: str, border_style: str) -> Panel:
    from rich.markdown import Markdown

    body = Markdown(answer.text)
    if answer.stop_reason == "max_tokens":
        body = Group(body, Text("[Answer cut off: it reached the --max-tokens limit.]", style="yellow"))
    usage = (
        f"{answer.input_tokens:,} in · {answer.output_tokens:,} out · {answer.seconds:.1f}s · {answer.model}"
    )
    return Panel(body, title=title, title_align="left", subtitle=usage, border_style=border_style)


def _print_citations(answer, prompt) -> None:
    """List the chunks the answer cites, and flag citations to chunks Claude wasn't shown."""
    by_id = {r.chunk.id: (rank, r) for rank, r in enumerate(prompt.results, start=1)}
    if not answer.citations:
        console.print("[yellow]The answer cites no chunks.[/yellow]")
        return
    table = Table(title="Cited chunks", show_header=True)
    table.add_column("Chunk", justify="right")
    table.add_column("Page(s)", justify="right", no_wrap=True)
    table.add_column("Retrieval rank", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Text (first 100 characters)")
    for c in answer.citations:
        if not c.in_context:
            table.add_row(str(c.chunk_id), c.pages, "[red]not supplied[/red]", "", "[red]Claude cited a chunk it was never shown[/red]")
            continue
        rank, r = by_id[c.chunk_id]
        snippet = " ".join(r.chunk.text.split())[:100] + "…"
        table.add_row(str(c.chunk_id), _pages_label(r.chunk.pages), f"{rank} of {len(prompt.results)}", f"{r.score:.3f}", Text(snippet))
    console.print(table)
    uncited = len(prompt.results) - sum(c.in_context for c in answer.citations)
    if uncited:
        console.print(f"[dim]{uncited} of the {len(prompt.results)} retrieved chunks were not cited.[/dim]")


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to ask about the PDF."),
    top_k: int = typer.Option(config.TOP_K, "--top-k", "-k", help="Number of chunks to include as context."),
    show_prompt: bool = typer.Option(False, "--show-prompt", help="Print the full prompt; send nothing to the API."),
    no_rag: bool = typer.Option(False, "--no-rag", help="Ask Claude with no document context."),
    compare: bool = typer.Option(False, "--compare", help="Show the RAG and no-RAG answers side by side."),
    model: str = typer.Option(config.CLAUDE_MODEL, "--model", help="Claude model to use."),
    max_tokens: int = typer.Option(config.MAX_TOKENS, "--max-tokens", help="Maximum length of each answer, in tokens."),
) -> None:
    """Stages 5-7: retrieve chunks, build the prompt, and get Claude's cited answer."""
    from rag_demo import generate as generation
    from rag_demo.augment import build_prompt

    if top_k < 1:
        console.print("[bold red]Error:[/bold red] --top-k must be at least 1.")
        raise typer.Exit(code=1)
    if no_rag and compare:
        console.print("[bold red]Error:[/bold red] use either --no-rag or --compare, not both.")
        raise typer.Exit(code=1)
    if not show_prompt:
        try:
            config.require_anthropic_api_key()  # fail before loading the embedding model
        except config.MissingAPIKeyError as err:
            console.print(f"[bold red]Error:[/bold red] {err}")
            raise typer.Exit(code=1)

    try:
        if no_rag and not show_prompt:  # no retrieval needed
            with console.status(f"Asking {model} with no context…"):
                answer = generation.generate_without_context(question, model, max_tokens)
            console.print(_answer_panel(answer, "Answer without RAG (no document context)", "red"))
            return

        retriever = _open_retriever()
        results = retriever.retrieve(question, top_k)
        prompt = build_prompt(question, results, retriever.manifest.pdf_name)
        if show_prompt:
            _print_prompt(prompt)
            console.print("[green]Nothing has been sent to the API.[/green]")
            return

        with console.status(f"Asking {model} with {len(results)} retrieved chunks…"):
            answer = generation.generate_answer(prompt, model, max_tokens)
        rag_panel = _answer_panel(answer, f"Answer with RAG ({len(results)} chunks)", "green")

        if compare:
            with console.status(f"Asking {model} the same question with no context…"):
                plain = generation.generate_without_context(question, model, max_tokens)
            side_by_side = Table.grid(expand=True, padding=(0, 1))
            side_by_side.add_column(ratio=1)
            side_by_side.add_column(ratio=1)
            side_by_side.add_row(rag_panel, _answer_panel(plain, "Without RAG (no document context)", "red"))
            console.print(side_by_side)
        else:
            console.print(rag_panel)
    except generation.GenerationError as err:
        console.print(f"[bold red]Error:[/bold red] {err}")
        raise typer.Exit(code=1)

    _print_citations(answer, prompt)
    from rag_demo.augment import CHARS_PER_TOKEN

    estimate = prompt.estimated_tokens["total"]
    console.print(
        f"[dim]Input tokens: {answer.input_tokens:,} reported by the API "
        f"(estimated {estimate:,} from characters ÷ {CHARS_PER_TOKEN}).[/dim]"
    )


@app.command("eval")
def eval_retrieval(
    strategy: str = typer.Option(None, "--strategy", help="Rebuild a temporary index with this chunk strategy."),
    chunk_size: int = typer.Option(None, "--chunk-size", help="Rebuild a temporary index with this chunk size."),
    overlap: int = typer.Option(None, "--overlap", help="Rebuild a temporary index with this overlap."),
    questions_path: Path = typer.Option(config.EVAL_QUESTIONS, "--questions", help="Questions file (JSON)."),
) -> None:
    """Measure retrieval: how often the right page is in the top 1, 3 and 5 chunks."""
    from rag_demo import evaluate as evaluation

    try:
        questions = evaluation.load_questions(questions_path)
    except (OSError, ValueError, KeyError, TypeError) as err:
        console.print(f"[bold red]Error:[/bold red] could not read questions from {questions_path}: {err}")
        raise typer.Exit(code=1)
    if not questions:
        console.print(f"[yellow]{questions_path} has no questions.[/yellow]")
        raise typer.Exit(code=1)

    if strategy is None and chunk_size is None and overlap is None:
        # Evaluate the saved index as built by 'rag-demo index'.
        retriever = _open_retriever()
        index, embedder, m = retriever.index, retriever.embedder, retriever.manifest
        label = f"saved index: {m.chunk_strategy}, size {m.chunk_size}, overlap {m.chunk_overlap}, {m.chunk_count:,} chunks"
    else:
        # Build a temporary index in memory with the requested settings; the saved one is untouched.
        strategy = strategy or config.CHUNK_STRATEGY
        chunk_size = chunk_size or config.CHUNK_SIZE
        overlap = 0 if strategy == "paragraph" else (config.CHUNK_OVERLAP if overlap is None else overlap)
        chunks = _load_chunks(_resolve_pdf(None), strategy, chunk_size, overlap)
        embedder = _get_embedder()
        columns = (TextColumn("Building temporary index"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn())
        with Progress(*columns, console=console, transient=True) as progress:
            task = progress.add_task("build", total=len(chunks))
            index = evaluation.build_index(chunks, embedder, on_batch=lambda n: progress.advance(task, n))
        label = f"temporary index: {strategy}, size {chunk_size}, overlap {overlap}, {len(chunks):,} chunks"

    results = evaluation.evaluate(index, embedder, questions)

    k_max = max(evaluation.HIT_RATE_KS)
    table = Table(title=f"Retrieval evaluation ({label})", show_header=True)
    table.add_column("#", justify="right")
    table.add_column("Question")
    table.add_column("Answer page(s)", justify="right")
    table.add_column("First hit", justify="right")
    table.add_column("Top result page(s)", justify="right", no_wrap=True)
    for i, r in enumerate(results, start=1):
        if r.first_hit_rank is None:
            rank = f"[red]miss (not in top {k_max})[/red]"
        else:
            colour = "green" if r.first_hit_rank == 1 else "yellow"
            rank = f"[{colour}]rank {r.first_hit_rank}[/{colour}]"
        table.add_row(
            str(i), Text(r.item.question), ", ".join(map(str, r.item.pages)), rank,
            _pages_label(r.top_pages) if r.top_pages else "",
        )
    console.print(table)

    summary = Table(title="Hit rate", show_header=True)
    summary.add_column("k", justify="right")
    summary.add_column("Questions with an answer page in the top k", justify="right")
    for k in evaluation.HIT_RATE_KS:
        hits = sum(1 for r in results if r.first_hit_rank is not None and r.first_hit_rank <= k)
        summary.add_row(str(k), f"{hits} / {len(results)}  ({evaluation.hit_rate(results, k):.0%})")
    console.print(summary)


@app.callback()
def main() -> None:
    """A step-by-step Retrieval-Augmented Generation demo over a single PDF."""


if __name__ == "__main__":
    app()
