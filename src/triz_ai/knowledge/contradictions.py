"""TRIZ contradiction matrix — 39x39 asymmetric matrix loaded from data/triz/matrix.json."""

import json
from functools import lru_cache
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "triz"


@lru_cache(maxsize=1)
def load_matrix() -> dict[tuple[int, int], list[int]]:
    """Load the 39x39 contradiction matrix.

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
        improving: The parameter being improved (1-39).
        worsening: The parameter that worsens as a result (1-39).

    Returns:
        List of recommended principle IDs (up to 4).
    """
    matrix = load_matrix()
    return matrix.get((improving, worsening), [])
