"""TRIZ contradiction matrix — asymmetric matrix loaded from triz_ai/data/matrix.json.

The original 39x39 Altshuller matrix covers parameters 1-39. Parameters 40-50
(modern extensions) have no historical matrix data; lookups return empty lists.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from triz_ai.patents.repository import PatentRepository

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@lru_cache(maxsize=1)
def load_matrix() -> dict[tuple[int, int], list[int]]:
    """Load the contradiction matrix.

    Returns:
        dict mapping (improving_param, worsening_param) -> list of principle IDs.
    """
    data_file = _DATA_DIR / "matrix.json"
    with open(data_file) as f:
        data = json.load(f)

    matrix: dict[tuple[int, int], list[int]] = {}
    for key, principles in data.items():
        parts = key.split(",")
        improving = int(parts[0])
        worsening = int(parts[1])
        matrix[(improving, worsening)] = principles
    return matrix


def lookup(improving: int, worsening: int) -> list[int]:
    """Look up recommended principles for a contradiction.

    Args:
        improving: The parameter being improved (1-50).
        worsening: The parameter that worsens as a result (1-50).

    Returns:
        List of recommended principle IDs (up to 4).
    """
    matrix = load_matrix()
    return matrix.get((improving, worsening), [])


def lookup_with_observations(
    improving: int,
    worsening: int,
    store: PatentRepository | None = None,
) -> list[int]:
    """Look up principles, merging static matrix with patent observations.

    Args:
        improving: The parameter being improved (1-50).
        worsening: The parameter that worsens as a result (1-50).
        store: PatentRepository instance for observation data (optional).

    Returns:
        List of recommended principle IDs (up to 4).
    """
    static_principles = lookup(improving, worsening)

    if store is None:
        return static_principles

    # Try to get patent-observed data
    try:
        observations = store.get_matrix_observations(min_count=3)
    except Exception:
        return static_principles

    observed = observations.get((improving, worsening))
    if not observed:
        return static_principles

    # Merge: score each principle by observation count + static presence bonus
    scores: dict[int, float] = {}
    static_set = set(static_principles)

    for principle_id, count, _avg_conf in observed:
        scores[principle_id] = count + (2.0 if principle_id in static_set else 0.0)

    # Add static principles not in observations with a base score
    for pid in static_principles:
        if pid not in scores:
            scores[pid] = 1.0

    # Return top 4 by score
    ranked = sorted(scores, key=lambda pid: scores[pid], reverse=True)
    return ranked[:4]
