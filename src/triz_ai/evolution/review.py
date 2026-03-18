"""Interactive review of candidate TRIZ principles."""

import logging

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

from triz_ai.patents.repository import PatentRepository
from triz_ai.patents.store import PatentStore

logger = logging.getLogger(__name__)
console = Console()


def interactive_review(store: PatentRepository | None = None) -> None:
    """Interactively review pending candidate principles.

    Displays each candidate with evidence and prompts for accept/reject.
    """
    if store is None:
        store = PatentStore()
        store.init_db()

    candidates = store.get_pending_candidates()
    if not candidates:
        console.print("[yellow]No pending candidates to review.[/yellow]")
        return

    console.print(f"\n[bold]Found {len(candidates)} candidate(s) to review.[/bold]\n")

    for candidate in candidates:
        # Display candidate details
        panel = Panel(
            f"[bold]{candidate.description or 'No description'}[/bold]\n\n"
            f"Confidence: {candidate.confidence:.0%}\n"
            f"Evidence patents: {len(candidate.evidence_patent_ids)}",
            title=f"[cyan]{candidate.id}: {candidate.name}[/cyan]",
            border_style="blue",
        )
        console.print(panel)

        # Show evidence patents
        if candidate.evidence_patent_ids:
            table = Table(title="Supporting Patents")
            table.add_column("Patent ID")
            table.add_column("Title")
            for pid in candidate.evidence_patent_ids:
                patent = store.get_patent(pid)
                title = patent.title if patent else "Unknown"
                table.add_row(pid, title)
            console.print(table)

        # Prompt for decision
        accept = Confirm.ask("Accept this candidate principle?")
        status = "accepted" if accept else "rejected"
        store.update_candidate_status(candidate.id, status)
        console.print(f"  → Marked as [{'green' if accept else 'red'}]{status}[/]\n")

    console.print("[bold green]Review complete.[/bold green]")


def interactive_parameter_review(store: PatentRepository | None = None) -> None:
    """Interactively review pending candidate parameters.

    Displays each candidate with evidence and prompts for accept/reject.
    """
    if store is None:
        store = PatentStore()
        store.init_db()

    candidates = store.get_pending_candidate_parameters()
    if not candidates:
        console.print("[yellow]No pending candidate parameters to review.[/yellow]")
        return

    console.print(f"\n[bold]Found {len(candidates)} candidate parameter(s) to review.[/bold]\n")

    for candidate in candidates:
        panel = Panel(
            f"[bold]{candidate.description or 'No description'}[/bold]\n\n"
            f"Confidence: {candidate.confidence:.0%}\n"
            f"Evidence patents: {len(candidate.evidence_patent_ids)}",
            title=f"[cyan]{candidate.id}: {candidate.name}[/cyan]",
            border_style="blue",
        )
        console.print(panel)

        if candidate.evidence_patent_ids:
            table = Table(title="Supporting Patents")
            table.add_column("Patent ID")
            table.add_column("Title")
            for pid in candidate.evidence_patent_ids:
                patent = store.get_patent(pid)
                title = patent.title if patent else "Unknown"
                table.add_row(pid, title)
            console.print(table)

        accept = Confirm.ask("Accept this candidate parameter?")
        status = "accepted" if accept else "rejected"
        store.update_candidate_parameter_status(candidate.id, status)
        console.print(f"  → Marked as [{'green' if accept else 'red'}]{status}[/]\n")

    console.print("[bold green]Parameter review complete.[/bold green]")
