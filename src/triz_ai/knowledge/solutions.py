"""Standard solutions (76 Su-Field standards) loaded from data file."""

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class StandardSolution(BaseModel):
    """A TRIZ standard solution (Su-Field standard)."""

    id: str
    class_id: int
    class_name: str
    name: str
    description: str
    applicability: str


@lru_cache(maxsize=1)
def load_standard_solutions() -> list[StandardSolution]:
    """Load all 76 standard solutions from data file."""
    data_file = _DATA_DIR / "standard_solutions.json"
    with open(data_file) as f:
        data = json.load(f)
    return [StandardSolution(**item) for item in data]


def get_solutions_by_class(class_id: int) -> list[StandardSolution]:
    """Get all standard solutions for a given class."""
    return [s for s in load_standard_solutions() if s.class_id == class_id]
