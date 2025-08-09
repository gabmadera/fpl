from __future__ import annotations

import json
from pathlib import Path
from typing import Set


def load_exclusions(path: str = "data/config/exclusions.json") -> Set[int]:
    p = Path(path)
    if not p.exists():
        return set()
    try:
        data = json.loads(p.read_text())
        if isinstance(data, list):
            return {int(x) for x in data if str(x).isdigit()}
        return set()
    except Exception:
        return set()

