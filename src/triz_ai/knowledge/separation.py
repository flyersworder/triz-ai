"""Separation principles for physical contradictions — 4 categories loaded from data file."""

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class SeparationPrinciple(BaseModel):
    """A TRIZ separation principle for resolving physical contradictions."""

    id: int
    category: str
    name: str
    description: str
    techniques: list[str] = []
    examples: list[str] = []


@lru_cache(maxsize=1)
def load_separation_principles() -> list[SeparationPrinciple]:
    """Load all separation principles from data file."""
    data_file = _DATA_DIR / "separation_principles.json"
    with open(data_file) as f:
        data = json.load(f)
    return [SeparationPrinciple(**item) for item in data]


def get_separation_principle(principle_id: int) -> SeparationPrinciple | None:
    """Get a single separation principle by ID."""
    for p in load_separation_principles():
        if p.id == principle_id:
            return p
    return None
