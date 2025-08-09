from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


class JsonCache:
    def __init__(self, root: str = "data/cache", ttl_seconds: int = 3600) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl_seconds

    def _path(self, key: str) -> Path:
        safe = key.replace("/", "_")
        return self.root / f"{safe}.json"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        p = self._path(key)
        if not p.exists():
            return None
        if self.ttl > 0 and time.time() - p.stat().st_mtime > self.ttl:
            return None
        try:
            return json.loads(p.read_text())
        except Exception:
            return None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        p = self._path(key)
        with open(p, "w") as f:
            json.dump(value, f)


def ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def safe_write_text(path: str, content: str) -> None:
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)

