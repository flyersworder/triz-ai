"""TRIZ inventive principles — 40 principles loaded from data/triz/principles.json."""

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel

_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "triz"


class Principle(BaseModel):
    """A TRIZ inventive principle."""

    id: int
    name: str
    description: str
    sub_principles: list[str] = []
    examples: list[str] = []
    keywords: list[str] = []


@lru_cache(maxsize=1)
def load_principles() -> list[Principle]:
    """Load all 40 TRIZ principles from data file."""
    data_file = _DATA_DIR / "principles.json"
    with open(data_file) as f:
        data = json.load(f)
    return [Principle(**item) for item in data]


def get_principle(principle_id: int) -> Principle | None:
    """Get a single principle by ID."""
    for p in load_principles():
        if p.id == principle_id:
            return p
    return None
