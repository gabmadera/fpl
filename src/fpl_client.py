from __future__ import annotations

import time
from typing import Any, Dict, Optional
import requests
from .config import config
from .security import SecurityManager
from .utils import JsonCache


class FPLClient:
    def __init__(self) -> None:
        self.security = SecurityManager()
        self.session = self.security.session
        self.base = config.FPL_BASE_URL.rstrip("/") + "/"
        self.cache = JsonCache(ttl_seconds=1800)

    def _get(self, endpoint: str, delay: Optional[float] = None) -> Dict[str, Any]:
        if not self.security.rate_limit_check():
            time.sleep(5)
        url = f"{self.base}{endpoint.lstrip('/')}"
        cache_key = endpoint.strip("/") or "root"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        time.sleep(delay if delay is not None else float(config.RATE_LIMIT_DELAY))
        resp = self.session.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        self.cache.set(cache_key, data)
        return data

    def bootstrap_static(self) -> Dict[str, Any]:
        return self._get("bootstrap-static/")

    def fixtures(self) -> Dict[str, Any]:
        return self._get("fixtures/")

    def element_summary(self, element_id: int) -> Dict[str, Any]:
        return self._get(f"element-summary/{element_id}/")

    def current_gameweek(self) -> int:
        data = self.bootstrap_static()
        events = data.get("events", [])
        for ev in events:
            if ev.get("is_current"):
                return int(ev.get("id", 1))
        # Fallback to next or first
        for ev in events:
            if ev.get("is_next"):
                return int(ev.get("id", 1))
        return 1

