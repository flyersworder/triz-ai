"""Evolution trends — 8 main TRIZ trends loaded from data file."""

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class TrendStage(BaseModel):
    """A single stage within an evolution trend."""

    stage: int
    name: str
    description: str


class EvolutionTrend(BaseModel):
    """A TRIZ evolution trend with stages."""

    id: int
    name: str
    description: str
    stages: list[TrendStage] = []


@lru_cache(maxsize=1)
def load_evolution_trends() -> list[EvolutionTrend]:
    """Load all evolution trends from data file."""
    data_file = _DATA_DIR / "evolution_trends.json"
    with open(data_file) as f:
        data = json.load(f)
    return [EvolutionTrend(**item) for item in data]


def get_trend(trend_id: int) -> EvolutionTrend | None:
    """Get a single trend by ID."""
    for t in load_evolution_trends():
        if t.id == trend_id:
            return t
    return None
