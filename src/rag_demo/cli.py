"""Command-line interface: one command per stage of the RAG pipeline.

Each command runs a single stage and prints what it produced, so the pipeline
can be demonstrated step by step. Commands are added phase by phase.
"""

import typer
from rich.console import Console
from rich.table import Table

from rag_demo import config

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


@app.callback()
def main() -> None:
    """A step-by-step Retrieval-Augmented Generation demo over a single PDF."""


if __name__ == "__main__":
    app()
