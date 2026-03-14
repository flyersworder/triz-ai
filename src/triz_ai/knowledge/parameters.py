"""TRIZ engineering parameters — 39 parameters loaded from data/triz/parameters.json."""

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel

_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "triz"


class Parameter(BaseModel):
    """A TRIZ engineering parameter."""

    id: int
    name: str
    description: str


@lru_cache(maxsize=1)
def load_parameters() -> list[Parameter]:
    """Load all 39 TRIZ engineering parameters from data file."""
    data_file = _DATA_DIR / "parameters.json"
    with open(data_file) as f:
        data = json.load(f)
    return [Parameter(**item) for item in data]


def get_parameter(param_id: int) -> Parameter | None:
    """Get a single parameter by ID."""
    for p in load_parameters():
        if p.id == param_id:
            return p
    return None
