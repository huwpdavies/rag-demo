"""Command-line interface: one command per stage of the RAG pipeline.

Each command runs a single stage and prints what it produced, so the pipeline
can be demonstrated step by step. Commands are added phase by phase.
"""

import statistics
import sys
from pathlib import Path

import typer
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table

from rag_demo import config, extract as extraction

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
    preview = target.text[:preview_chars]
    if len(target.text) > preview_chars:
        preview += " …"
    console.print()
    console.print(
        Panel(
            preview or "[yellow](no text on this page)[/yellow]",
            title=f"Preview: page {target.page_number}",
            subtitle=f"{len(target.text):,} characters",
        )
    )


@app.callback()
def main() -> None:
    """A step-by-step Retrieval-Augmented Generation demo over a single PDF."""


if __name__ == "__main__":
    app()
