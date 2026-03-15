"""Hybrid matrix builder — LLM-seeds missing contradiction matrix cells."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from rich.progress import Progress

from triz_ai.knowledge.contradictions import load_matrix

if TYPE_CHECKING:
    from triz_ai.llm.client import LLMClient

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def seed_matrix(llm_client: LLMClient, force: bool = False) -> int:
    """Seed missing contradiction matrix cells via LLM.

    Args:
        llm_client: LLM client instance for generating matrix entries.
        force: If True, re-seed all cells involving params 40-50 even if present.

    Returns:
        Number of cells added.
    """
    # Load current matrix and raw JSON
    current_matrix = load_matrix()
    matrix_file = _DATA_DIR / "matrix.json"
    with open(matrix_file) as f:
        raw: dict[str, list[int]] = json.load(f)

    # Compute missing cells
    all_params = range(1, 51)
    missing: dict[int, list[int]] = {}  # improving -> [worsening, ...]

    for i in all_params:
        for j in all_params:
            if i == j:
                continue

            is_missing = (i, j) not in current_matrix
            if force and (i >= 40 or j >= 40):
                is_missing = True

            if is_missing:
                missing.setdefault(i, []).append(j)

    if not missing:
        return 0

    cells_added = 0

    with Progress() as progress:
        task = progress.add_task("Seeding matrix...", total=len(missing))

        for improving, worsening_list in missing.items():
            try:
                result = llm_client.seed_matrix_row(improving, worsening_list)
            except Exception:
                progress.advance(task)
                continue

            for entry in result.entries:
                # Validate entry
                if entry.improving != improving:
                    continue
                if entry.worsening not in worsening_list:
                    continue
                if not all(1 <= pid <= 40 for pid in entry.principles):
                    continue
                if len(entry.principles) > 4:
                    continue

                key = f"{entry.improving},{entry.worsening}"
                if entry.principles:
                    raw[key] = entry.principles
                    cells_added += 1

            progress.advance(task)

    # Write back to matrix.json
    with open(matrix_file, "w") as f:
        json.dump(raw, f, separators=(",", ":"))

    # Clear the lru_cache so subsequent lookups see new data
    load_matrix.cache_clear()

    return cells_added
