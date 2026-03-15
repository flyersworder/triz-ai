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
    method: str = typer.Option(
        None,
        help="Force a specific TRIZ method: technical-contradiction, physical-contradiction, "
        "su-field, function-analysis, trimming, trends",
    ),
    router_model: str = typer.Option(
        None,
        help="Model for problem classification (default: classify_model from config)",
    ),
    format: str = typer.Option("text", help="Output format: text, json, markdown"),
) -> None:
    """Analyze a technical problem using TRIZ methodology.

    Auto-classifies the problem and routes to the best TRIZ tool,
    or use --method to force a specific analysis method.
    """
    from triz_ai.config import load_config
    from triz_ai.engine.router import route

    config = load_config()
    effective_router_model = router_model or config.llm.router_model
    llm_client = _get_llm_client(model)
    store = _get_store()

    try:
        result = route(
            problem,
            llm_client=llm_client,
            store=store,
            method=method,
            router_model=effective_router_model,
        )
    except Exception as e:
        console.print(f"[red]Analysis failed: {e}[/red]")
        raise typer.Exit(1) from None

    if format != "text":
        _output(result.model_dump(), format)
        return

    # --- Common header ---
    console.print(Panel(problem, title="[bold]Problem[/bold]", border_style="blue"))

    method_label = result.method.replace("_", " ").title()
    conf_str = f" [dim](confidence: {result.method_confidence:.0%})[/dim]"
    console.print(f"\n[bold]Method:[/bold] [cyan]{method_label}[/cyan]{conf_str}")

    if result.ideal_final_result:
        console.print(f"[bold]Ideal Final Result:[/bold] {result.ideal_final_result}")

    console.print(f"[dim]{result.reasoning}[/dim]\n")

    # --- Method-specific rendering ---
    _render_method_details(result)

    # --- Patents (common across all methods) ---
    _render_patents(result)

    # --- Solution directions (common across all methods) ---
    _render_solution_directions(result)

    # --- Secondary method tip ---
    if result.secondary_method:
        secondary_label = result.secondary_method.replace("_", "-")
        console.print(
            f"\n[dim]Tip: Try [cyan]--method {secondary_label}[/cyan] "
            f"for a different perspective.[/dim]"
        )


def _render_method_details(result) -> None:
    """Render method-specific details."""
    details = result.details
    if not details:
        return

    if result.method == "technical_contradiction":
        _render_contradiction_details(result, details)
    elif result.method == "physical_contradiction":
        _render_physical_details(details)
    elif result.method == "su_field":
        _render_su_field_details(details)
    elif result.method == "function_analysis":
        _render_function_details(details)
    elif result.method == "trimming":
        _render_trimming_details(details)
    elif result.method == "trends":
        _render_trends_details(details)


def _render_contradiction_details(result, details: dict) -> None:
    """Render technical contradiction specific output."""
    if result.improving_param and result.worsening_param:
        conf = details.get("contradiction_confidence", result.contradiction_confidence)
        conf_str = f" [dim](confidence: {conf:.0%})[/dim]"
        console.print(
            f"[bold]Contradiction:[/bold] Improving "
            f"[cyan]{result.improving_param['name']}[/cyan] "
            f"worsens [red]{result.worsening_param['name']}[/red]{conf_str}"
        )
        if conf < 0.5:
            console.print(
                "[yellow]Low confidence — consider rephrasing your problem "
                "as 'improve X without worsening Y'.[/yellow]"
            )

    if result.recommended_principles:
        table = Table(title="Recommended TRIZ Principles")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="bold")
        table.add_column("Description")
        for p in result.recommended_principles:
            table.add_row(str(p["id"]), p["name"], p["description"])
        console.print(table)


def _render_physical_details(details: dict) -> None:
    """Render physical contradiction specific output."""
    console.print(
        f"[bold]Physical Contradiction:[/bold] "
        f"[cyan]{details.get('property', '?')}[/cyan] must be "
        f"'[green]{details.get('requirement_a', '?')}[/green]' AND "
        f"'[red]{details.get('requirement_b', '?')}[/red]'"
    )
    sep_type = details.get("separation_type", "").replace("_", " ").title()
    console.print(f"[bold]Separation Type:[/bold] {sep_type}\n")

    sep_principles = details.get("separation_principles", [])
    if sep_principles:
        table = Table(title="Separation Principles")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="bold")
        table.add_column("Technique")
        for sp in sep_principles:
            table.add_row(str(sp.get("id", "")), sp.get("name", ""), sp.get("technique", ""))
        console.print(table)


def _render_su_field_details(details: dict) -> None:
    """Render Su-Field analysis specific output."""
    substances = details.get("substances", [])
    field = details.get("field", "?")
    problem_type = details.get("problem_type", "?")

    console.print(
        f"[bold]Su-Field Model:[/bold] "
        f"Substances: [cyan]{', '.join(substances)}[/cyan] | "
        f"Field: [green]{field}[/green] | "
        f"Problem: [red]{problem_type}[/red]\n"
    )

    solutions = details.get("standard_solutions", [])
    if solutions:
        table = Table(title="Recommended Standard Solutions")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="bold")
        table.add_column("Applicability")
        for ss in solutions:
            table.add_row(ss.get("id", ""), ss.get("name", ""), ss.get("applicability", ""))
        console.print(table)


def _render_function_details(details: dict) -> None:
    """Render function analysis specific output."""
    components = details.get("components", [])
    if components:
        table = Table(title="System Components")
        table.add_column("Component", style="cyan")
        table.add_column("Role")
        for c in components:
            table.add_row(c.get("name", ""), c.get("role", ""))
        console.print(table)

    functions = details.get("functions", [])
    if functions:
        table = Table(title="Functions")
        table.add_column("Subject", style="cyan")
        table.add_column("Action", style="bold")
        table.add_column("Object")
        table.add_column("Type", style="yellow")
        for f in functions:
            table.add_row(
                f.get("subject", ""),
                f.get("action", ""),
                f.get("object", ""),
                f.get("type", ""),
            )
        console.print(table)

    problem_functions = details.get("problem_functions", [])
    if problem_functions:
        console.print("\n[bold]Problematic Functions:[/bold]")
        for pf in problem_functions:
            console.print(
                f"  [red]•[/red] {pf.get('subject', '')} → {pf.get('action', '')} → "
                f"{pf.get('object', '')}: [red]{pf.get('problem', '')}[/red]"
            )

    recommendations = details.get("recommendations", [])
    if recommendations:
        console.print("\n[bold]Recommendations:[/bold]")
        for i, rec in enumerate(recommendations, 1):
            console.print(f"  {i}. {rec}")
    console.print()


def _render_trimming_details(details: dict) -> None:
    """Render trimming analysis specific output."""
    components = details.get("components", [])
    if components:
        table = Table(title="System Components")
        table.add_column("Component", style="cyan")
        table.add_column("Function")
        table.add_column("Cost", style="yellow")
        for c in components:
            table.add_row(c.get("name", ""), c.get("function", ""), c.get("cost", ""))
        console.print(table)

    candidates = details.get("trimming_candidates", [])
    if candidates:
        table = Table(title="Trimming Candidates")
        table.add_column("Component", style="red")
        table.add_column("Reason")
        table.add_column("Rule", style="cyan")
        for tc in candidates:
            table.add_row(tc.get("component", ""), tc.get("reason", ""), tc.get("rule", ""))
        console.print(table)

    redistributed = details.get("redistributed_functions", [])
    if redistributed:
        console.print("\n[bold]Function Redistribution:[/bold]")
        for rf in redistributed:
            console.print(
                f"  [cyan]{rf.get('function', '')}[/cyan]: "
                f"[red]{rf.get('from', '')}[/red] → [green]{rf.get('to', '')}[/green]"
            )
    console.print()


def _render_trends_details(details: dict) -> None:
    """Render trends analysis specific output."""
    current = details.get("current_stage", {})
    console.print(
        f"[bold]Current Position:[/bold] [cyan]{details.get('trend_name', '?')}[/cyan] — "
        f"Stage {current.get('stage', '?')}: {current.get('stage_name', '?')}\n"
    )

    next_stages = details.get("next_stages", [])
    if next_stages:
        table = Table(title="Next Evolutionary Stages")
        table.add_column("Stage", style="cyan", justify="center")
        table.add_column("Name", style="bold")
        table.add_column("Description")
        for ns in next_stages:
            table.add_row(str(ns.get("stage", "")), ns.get("name", ""), ns.get("description", ""))
        console.print(table)

    predictions = details.get("predictions", [])
    if predictions:
        console.print("\n[bold]Predictions:[/bold]")
        for i, pred in enumerate(predictions, 1):
            console.print(f"  {i}. {pred}")
    console.print()


def _render_patents(result) -> None:
    """Render patent examples (common across all methods)."""
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


def _render_solution_directions(result) -> None:
    """Render solution directions (common across all methods)."""
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
