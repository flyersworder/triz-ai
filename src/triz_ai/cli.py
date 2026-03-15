"""CLI entry point for triz-ai."""

import json
import logging
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Suppress litellm noise before any imports trigger it
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

app = typer.Typer(
    name="triz-ai",
    help="AI-Powered TRIZ Innovation Engine — analyze problems, classify patents, "
    "and discover new inventive principles.",
    no_args_is_help=True,
)
console = Console(stderr=True)


def _get_llm_client(model: str | None = None):
    from triz_ai.llm.client import LLMClient

    return LLMClient(model=model)


def _get_store():
    from triz_ai.patents.store import PatentStore

    store = PatentStore()
    store.init_db()
    return store


def _output(data: dict, fmt: str) -> None:
    """Output data in the requested format."""
    if fmt == "json":
        typer.echo(json.dumps(data, indent=2))
    elif fmt == "markdown":
        for key, value in data.items():
            if isinstance(value, list):
                typer.echo(f"\n## {key.replace('_', ' ').title()}\n")
                for item in value:
                    if isinstance(item, dict):
                        parts = [f"**{k}**: {v}" for k, v in item.items()]
                        typer.echo(f"- {', '.join(parts)}")
                    else:
                        typer.echo(f"- {item}")
            else:
                typer.echo(f"**{key.replace('_', ' ').title()}**: {value}")
    # text format handled per-command with rich


@app.command()
def analyze(
    problem: str = typer.Argument(help="Technical problem description"),
    model: str = typer.Option(None, help="LLM model string (overrides config)"),
    format: str = typer.Option("text", help="Output format: text, json, markdown"),
) -> None:
    """Analyze a technical problem using TRIZ methodology."""
    from triz_ai.engine.analyzer import analyze as run_analyze

    llm_client = _get_llm_client(model)
    store = _get_store()

    try:
        result = run_analyze(problem, llm_client=llm_client, store=store)
    except Exception as e:
        console.print(f"[red]Analysis failed: {e}[/red]")
        raise typer.Exit(1) from None

    if format != "text":
        _output(result.model_dump(), format)
        return

    # Rich text output
    console.print(Panel(problem, title="[bold]Problem[/bold]", border_style="blue"))

    console.print(
        f"\n[bold]Contradiction:[/bold] Improving [cyan]{result.improving_param['name']}[/cyan] "
        f"worsens [red]{result.worsening_param['name']}[/red]"
    )
    console.print(f"[dim]{result.reasoning}[/dim]\n")

    if result.recommended_principles:
        table = Table(title="Recommended TRIZ Principles")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="bold")
        table.add_column("Description")
        for p in result.recommended_principles:
            table.add_row(str(p["id"]), p["name"], p["description"])
        console.print(table)

    if result.patent_examples:
        console.print("\n[bold]Related Patents:[/bold]")
        for p in result.patent_examples:
            console.print(f"  • [cyan]{p['id']}[/cyan] — {p['title']}")


@app.command()
def classify(
    source: str = typer.Argument(help="Patent file path, quoted text, or '-' for stdin"),
    model: str = typer.Option(None, help="LLM model string (overrides config)"),
    format: str = typer.Option("text", help="Output format: text, json, markdown"),
) -> None:
    """Classify a patent through TRIZ lens."""
    from triz_ai.engine.classifier import classify as run_classify
    from triz_ai.knowledge.principles import load_principles

    # Determine input source
    if source == "-":
        patent_text = sys.stdin.read()
    elif Path(source).exists():
        patent_text = Path(source).read_text(encoding="utf-8")
    else:
        patent_text = source

    llm_client = _get_llm_client(model)
    store = _get_store()

    try:
        result = run_classify(patent_text, llm_client=llm_client, store=store)
    except Exception as e:
        console.print(f"[red]Classification failed: {e}[/red]")
        raise typer.Exit(1) from None

    if format != "text":
        _output(result.model_dump(), format)
        return

    # Rich text output
    principles = {p.id: p for p in load_principles()}
    console.print(f"\n[bold]Confidence:[/bold] {result.confidence:.0%}")
    console.print(f"[bold]Contradiction:[/bold] {result.contradiction}")
    console.print(f"\n[dim]{result.reasoning}[/dim]\n")

    table = Table(title="Identified TRIZ Principles")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="bold")
    for pid in result.principle_ids:
        p = principles.get(pid)
        name = p.name if p else "Unknown"
        table.add_row(str(pid), name)
    console.print(table)


@app.command()
def discover(
    domain: str = typer.Option(..., help="Technology domain to analyze"),
    model: str = typer.Option(None, help="LLM model string (overrides config)"),
    format: str = typer.Option("text", help="Output format: text, json, markdown"),
) -> None:
    """Discover underused TRIZ principles in a domain."""
    from triz_ai.engine.generator import discover as run_discover

    llm_client = _get_llm_client(model)
    store = _get_store()

    try:
        result = run_discover(domain, llm_client=llm_client, store=store)
    except Exception as e:
        console.print(f"[red]Discovery failed: {e}[/red]")
        raise typer.Exit(1) from None

    if format != "text":
        _output(result.model_dump(), format)
        return

    # Rich text output
    panel_text = f"Domain: [cyan]{result.domain}[/cyan]\nPatents in store: {result.total_patents}"
    panel_text += f"\nClassified: {result.classified_patents}"
    if result.classified_patents == 0:
        panel_text += "\n[yellow]No classified patents yet. Run 'triz-ai evolve' first.[/yellow]"
    console.print(
        Panel(
            panel_text,
            title="[bold]Discovery Report[/bold]",
            border_style="green",
        )
    )

    # Principle usage table (only show used ones + top underused)
    used = [p for p in result.principle_usage if p["count"] > 0]
    if used:
        table = Table(title="Principle Usage")
        table.add_column("ID", style="cyan")
        table.add_column("Principle", style="bold")
        table.add_column("Count", justify="right")
        for p in sorted(used, key=lambda x: x["count"], reverse=True):
            table.add_row(str(p["id"]), p["name"], str(p["count"]))
        console.print(table)

    if result.underused_principles:
        console.print(f"\n[bold]Underused Principles ({len(result.underused_principles)}):[/bold]")
        for p in result.underused_principles[:10]:
            console.print(f"  • [yellow]{p['id']}[/yellow] — {p['name']}")

    if result.ideas:
        console.print("\n")
        table = Table(title="Generated Ideas")
        table.add_column("Principle", style="cyan")
        table.add_column("Idea", style="bold")
        table.add_column("Reasoning")
        for idea in result.ideas:
            table.add_row(str(idea["principle_id"]), idea["idea"], idea["reasoning"])
        console.print(table)


@app.command()
def evolve(
    review: bool = typer.Option(False, help="Interactive review of candidate principles"),
    model: str = typer.Option(None, help="LLM model string (overrides config)"),
    format: str = typer.Option("text", help="Output format: text, json, markdown"),
) -> None:
    """Discover candidate new TRIZ principles from patents."""
    store = _get_store()

    if review:
        from triz_ai.evolution.review import interactive_review

        interactive_review(store=store)
        return

    from triz_ai.evolution.pipeline import run_evolution

    llm_client = _get_llm_client(model)

    try:
        candidates = run_evolution(llm_client=llm_client, store=store)
    except Exception as e:
        console.print(f"[red]Evolution failed: {e}[/red]")
        raise typer.Exit(1) from None

    if format != "text":
        _output(
            {"candidates": [c.model_dump() for c in candidates]},
            format,
        )
        return

    if not candidates:
        console.print("[yellow]No new candidate principles discovered.[/yellow]")
        return

    table = Table(title="New Candidate Principles")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="bold")
    table.add_column("Confidence", justify="right")
    table.add_column("Evidence")
    for c in candidates:
        table.add_row(c.id, c.name, f"{c.confidence:.0%}", f"{len(c.evidence_patent_ids)} patents")
    console.print(table)
    console.print("\nRun [cyan]triz-ai evolve --review[/cyan] to accept or reject candidates.")


@app.command()
def ingest(
    source: str = typer.Argument(help="File or directory path to ingest"),
    model: str = typer.Option(None, help="LLM model string (overrides config)"),
) -> None:
    """Ingest patent data from files."""
    from triz_ai.patents.ingest import ingest_directory, ingest_file

    path = Path(source)
    llm_client = _get_llm_client(model)
    store = _get_store()

    try:
        if path.is_dir():
            patents = ingest_directory(path, store, llm_client=llm_client)
        else:
            patents = ingest_file(path, store, llm_client=llm_client)
    except Exception as e:
        console.print(f"[red]Ingestion failed: {e}[/red]")
        raise typer.Exit(1) from None

    console.print(f"[green]Successfully ingested {len(patents)} patent(s).[/green]")
    for p in patents:
        console.print(f"  • [cyan]{p.id}[/cyan] — {p.title}")


@app.command()
def init(
    force: bool = typer.Option(False, help="Recreate database (destroys existing data)"),
) -> None:
    """Initialize the triz-ai database."""
    from triz_ai.config import get_db_path
    from triz_ai.patents.store import PatentStore

    db_path = get_db_path()
    store = PatentStore(db_path=db_path)

    if force:
        console.print(f"[yellow]Recreating database at {db_path}...[/yellow]")
    else:
        console.print(f"Initializing database at {db_path}...")

    store.init_db(force=force)
    console.print("[green]Database initialized successfully.[/green]")
    store.close()
