"""CLI entry point for triz-ai."""

import json
import logging
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
    help="AI-Powered TRIZ Innovation Engine — analyze problems, ingest patents, "
    "and discover new inventive principles.",
    no_args_is_help=True,
)
console = Console(stderr=True)


def _get_llm_client(model: str | None = None, classify_model: str | None = None):
    from triz_ai.llm.client import LLMClient

    return LLMClient(model=model, classify_model=classify_model)


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

    conf = result.contradiction_confidence
    conf_str = f" [dim](confidence: {conf:.0%})[/dim]"
    console.print(
        f"\n[bold]Contradiction:[/bold] Improving [cyan]{result.improving_param['name']}[/cyan] "
        f"worsens [red]{result.worsening_param['name']}[/red]{conf_str}"
    )
    if conf < 0.5:
        console.print(
            "[yellow]Low confidence — consider rephrasing your problem "
            "as 'improve X without worsening Y'.[/yellow]"
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
        patent_table = Table(title="Related Patents")
        patent_table.add_column("ID", style="cyan")
        patent_table.add_column("Assignee", style="bold")
        patent_table.add_column("Title")
        patent_table.add_column("Matched Principles", style="green")
        for p in result.patent_examples:
            assignee = p.get("assignee") or "—"
            matched = ", ".join(p.get("matched_principles", [])) or "—"
            patent_table.add_row(p["id"], assignee, p["title"], matched)
            if p.get("filing_date"):
                patent_table.add_row("", f"[dim]Filed: {p['filing_date']}[/dim]", "", "")
        console.print(patent_table)
    else:
        console.print(
            "\n[dim]Tip: run [cyan]triz-ai ingest <file>[/cyan] "
            "to get patent-backed examples.[/dim]"
        )

    if result.solution_directions:
        console.print("\n[bold]Solution Directions:[/bold]")
        for i, d in enumerate(result.solution_directions, 1):
            principles_str = ", ".join(d.get("principles_applied", []))
            console.print(f"\n  [cyan]{i}.[/cyan] [bold]{d['title']}[/bold]")
            console.print(f"     {d['description']}")
            if principles_str:
                console.print(f"     [dim]Applies: {principles_str}[/dim]")


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
        table.add_column("Source Patent", style="dim")
        for idea in result.ideas:
            source = idea.get("source_patent_id") or "—"
            table.add_row(str(idea["principle_id"]), idea["idea"], idea["reasoning"], source)
        console.print(table)


@app.command()
def evolve(
    review: bool = typer.Option(False, help="Interactive review of candidates"),
    parameters: bool = typer.Option(
        False, help="Discover candidate parameters instead of principles"
    ),
    model: str = typer.Option(None, help="LLM model string (overrides config)"),
    format: str = typer.Option("text", help="Output format: text, json, markdown"),
) -> None:
    """Discover candidate new TRIZ principles or parameters from patents."""
    store = _get_store()

    if review:
        if parameters:
            from triz_ai.evolution.review import interactive_parameter_review

            interactive_parameter_review(store=store)
        else:
            from triz_ai.evolution.review import interactive_review

            interactive_review(store=store)
        return

    llm_client = _get_llm_client(model)

    if parameters:
        from triz_ai.evolution.pipeline import run_parameter_evolution

        try:
            candidates = run_parameter_evolution(llm_client=llm_client, store=store)
        except Exception as e:
            console.print(f"[red]Parameter evolution failed: {e}[/red]")
            raise typer.Exit(1) from None

        if format != "text":
            _output(
                {"candidates": [c.model_dump() for c in candidates]},
                format,
            )
            return

        if not candidates:
            console.print("[yellow]No new candidate parameters discovered.[/yellow]")
            return

        table = Table(title="New Candidate Parameters")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="bold")
        table.add_column("Confidence", justify="right")
        table.add_column("Evidence")
        for c in candidates:
            table.add_row(
                c.id, c.name, f"{c.confidence:.0%}", f"{len(c.evidence_patent_ids)} patents"
            )
        console.print(table)
        console.print(
            "\nRun [cyan]triz-ai evolve --parameters --review[/cyan] "
            "to accept or reject candidates."
        )
        return

    from triz_ai.evolution.pipeline import run_evolution

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
    classify_model: str = typer.Option(
        None,
        help="Model for classification (default: smaller model from config)",
    ),
) -> None:
    """Ingest and classify patent data from files."""
    from triz_ai.patents.ingest import ingest_directory, ingest_file

    path = Path(source)
    llm_client = _get_llm_client(model, classify_model=classify_model)
    store = _get_store()

    try:
        if path.is_dir():
            patents, classified = ingest_directory(path, store, llm_client=llm_client)
        else:
            patents, classified = ingest_file(path, store, llm_client=llm_client)
    except Exception as e:
        console.print(f"[red]Ingestion failed: {e}[/red]")
        raise typer.Exit(1) from None

    console.print(f"[green]Successfully ingested {len(patents)} patent(s).[/green]")
    skipped = len(patents) - classified
    if classified > 0:
        console.print(f"[green]Classified {classified} patent(s) through TRIZ lens.[/green]")
    if skipped > 0:
        console.print(
            f"[yellow]{skipped} patent(s) could not be classified "
            "(will retry on next evolve).[/yellow]"
        )
    for p in patents:
        console.print(f"  • [cyan]{p.id}[/cyan] — {p.title}")


@app.command()
def init(
    force: bool = typer.Option(False, help="Recreate database (destroys existing data)"),
) -> None:
    """Initialize the triz-ai database (only needed with --force to reset)."""
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


# --- Matrix subcommands ---
matrix_app = typer.Typer(
    name="matrix",
    help="Manage the TRIZ contradiction matrix.",
    no_args_is_help=True,
)
app.add_typer(matrix_app, name="matrix")


@matrix_app.command()
def seed(
    force: bool = typer.Option(False, help="Re-seed all cells involving params 40-50"),
    model: str = typer.Option(None, help="LLM model string (overrides config)"),
) -> None:
    """LLM-seed missing contradiction matrix cells."""
    from triz_ai.knowledge.matrix_builder import seed_matrix

    llm_client = _get_llm_client(model)

    try:
        cells_added = seed_matrix(llm_client, force=force)
    except Exception as e:
        console.print(f"[red]Matrix seeding failed: {e}[/red]")
        raise typer.Exit(1) from None

    if cells_added == 0:
        from triz_ai.knowledge.contradictions import load_matrix

        matrix = load_matrix()
        total_possible = 50 * 49
        if len(matrix) >= total_possible:
            console.print("[yellow]No missing cells to seed.[/yellow]")
        else:
            console.print(
                "[yellow]No cells were seeded. LLM calls may have failed — "
                "check your API key and rate limits.[/yellow]"
            )
    else:
        console.print(f"[green]Seeded {cells_added} matrix cells.[/green]")


@matrix_app.command()
def stats() -> None:
    """Show contradiction matrix fill rate and observation stats."""
    from triz_ai.knowledge.contradictions import load_matrix

    matrix = load_matrix()
    total_possible = 50 * 49  # 50 params, excluding diagonal
    filled = len(matrix)
    fill_pct = filled / total_possible * 100

    console.print(
        Panel(
            f"Total possible cells: {total_possible}\n"
            f"Filled cells (static): {filled}\n"
            f"Fill rate: {fill_pct:.1f}%",
            title="[bold]Contradiction Matrix Stats[/bold]",
            border_style="blue",
        )
    )

    # Show observation stats if DB is available
    try:
        store = _get_store()
        observations = store.get_matrix_observations(min_count=1)
        total_obs_cells = len(observations)
        obs_with_min3 = len(store.get_matrix_observations(min_count=3))

        console.print("\n[bold]Patent Observations:[/bold]")
        console.print(f"  Cells with observations: {total_obs_cells}")
        console.print(f"  Cells with 3+ observations (active): {obs_with_min3}")

        # Top pairs by observation count
        if observations:
            from triz_ai.knowledge.parameters import get_parameter

            table = Table(title="Top Observed Parameter Pairs")
            table.add_column("Improving", style="cyan")
            table.add_column("Worsening", style="red")
            table.add_column("Principles", style="bold")
            table.add_column("Observations", justify="right")

            # Sort pairs by total observations
            pairs = sorted(
                observations.items(),
                key=lambda item: sum(count for _, count, _ in item[1]),
                reverse=True,
            )
            for (imp, wor), entries in pairs[:10]:
                imp_p = get_parameter(imp)
                wor_p = get_parameter(wor)
                imp_name = imp_p.name if imp_p else str(imp)
                wor_name = wor_p.name if wor_p else str(wor)
                total = sum(count for _, count, _ in entries)
                prins = ", ".join(str(pid) for pid, _, _ in entries)
                table.add_row(imp_name, wor_name, prins, str(total))
            console.print(table)
        store.close()
    except Exception:
        console.print("[dim]No patent store available for observation stats.[/dim]")
