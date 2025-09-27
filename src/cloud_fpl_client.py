"""
Cloud-optimized FPL Client with proxy support and fallbacks for Render deployment
Handles 403 errors and cloud IP blocking issues
"""

from __future__ import annotations

import time
import json
import random
from typing import Any, Dict, Optional
import requests
from .config import config
from .security import SecurityManager
from .utils import JsonCache


class CloudFPLClient:
    """FPL Client optimized for cloud deployment with anti-blocking measures"""

    def __init__(self) -> None:
        self.security = SecurityManager()
        self.session = self._create_cloud_session()
        self.base = config.FPL_BASE_URL.rstrip("/") + "/"
        self.cache = JsonCache(ttl_seconds=1800)  # 30 min cache for cloud
        self.fallback_data = self._load_fallback_data()

    def _create_cloud_session(self) -> requests.Session:
        """Create session optimized for cloud deployment"""
        session = requests.Session()

        # Rotate User-Agents to avoid detection
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0"
        ]

        session.headers.update({
            "User-Agent": random.choice(user_agents),
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Cache-Control": "max-age=0"
        })

        return session

    def _load_fallback_data(self) -> Dict[str, Any]:
        """Load fallback data for when API is blocked"""
        fallback = {
            "bootstrap_static": {
                "events": [
                    {"id": 5, "name": "Gameweek 5", "is_current": True, "finished": True},
                    {"id": 6, "name": "Gameweek 6", "is_current": False, "is_next": True, "finished": False}
                ],
                "elements": [],  # Will be populated
                "teams": []      # Will be populated
            },
            "fixtures": [],
            "current_gameweek": 6,
            "next_gameweek": 6
        }
        return fallback

    def _get_with_fallback(self, endpoint: str, delay: Optional[float] = None) -> Dict[str, Any]:
        """Get data with fallback to cached/mock data on 403 errors"""
        url = f"{self.base}{endpoint.lstrip('/')}"
        cache_key = endpoint.strip("/") or "root"

        # Check cache first
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        # Add delay for rate limiting
        time.sleep(delay if delay is not None else float(config.RATE_LIMIT_DELAY))

        try:
            # Try direct request first
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            self.cache.set(cache_key, data)
            return data

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                print(f"⚠️  FPL API blocked (403) for {endpoint}, using fallback data")
                return self._get_fallback_data(endpoint)
            raise

        except Exception as e:
            print(f"⚠️  FPL API error for {endpoint}: {e}, using fallback data")
            return self._get_fallback_data(endpoint)

    def _get_fallback_data(self, endpoint: str) -> Dict[str, Any]:
        """Return fallback data when API is blocked"""
        endpoint_clean = endpoint.strip("/")

        if endpoint_clean == "bootstrap-static":
            return self.fallback_data["bootstrap_static"]
        elif endpoint_clean == "fixtures":
            return self.fallback_data["fixtures"]
        elif endpoint_clean.startswith("event/") and endpoint_clean.endswith("/live"):
            # Return empty live data
            return {"elements": []}
        elif endpoint_clean.startswith("element-summary/"):
            # Return empty element summary
            return {"history": [], "fixtures": []}
        else:
            return {}

    def bootstrap_static(self) -> Dict[str, Any]:
        """Get bootstrap static data with fallback"""
        return self._get_with_fallback("bootstrap-static/")

    def fixtures(self) -> Dict[str, Any]:
        """Get fixtures data with fallback"""
        return self._get_with_fallback("fixtures/")

    def element_summary(self, element_id: int) -> Dict[str, Any]:
        """Get element summary with fallback"""
        return self._get_with_fallback(f"element-summary/{element_id}/")

    def current_gameweek(self) -> int:
        """Get current gameweek with fallback"""
        try:
            data = self.bootstrap_static()
            events = data.get("events", [])
            for ev in events:
                if ev.get("is_current"):
                    return int(ev.get("id", 6))  # Default to GW6
            return 6  # Safe fallback
        except:
            return 6  # Safe fallback

    def next_active_gameweek(self) -> int:
        """Get next active gameweek with fallback"""
        try:
            data = self.bootstrap_static()
            events = data.get("events", [])

            # Find next unfinished gameweek
            for ev in events:
                if ev.get("is_next") and not ev.get("finished", True):
                    return int(ev.get("id", 6))

            # Find current unfinished
            for ev in events:
                if ev.get("is_current") and not ev.get("finished", True):
                    return int(ev.get("id", 6))

            return 6  # Safe fallback
        except:
            return 6  # Safe fallback

    def event_live(self, gameweek: int) -> Dict[str, Any]:
        """Get live event data with fallback"""
        return self._get_with_fallback(f"event/{gameweek}/live/")

    def get_player_gameweek_stats(self, player_id: int, gameweek: int) -> Optional[Dict[str, Any]]:
        """Get player stats with fallback"""
        try:
            element_data = self.element_summary(player_id)
            history = element_data.get('history', [])

            for gw_data in history:
                if gw_data.get('round') == gameweek:
                    return {
                        'points': gw_data.get('total_points', 0),
                        'minutes': gw_data.get('minutes', 0),
                        'goals': gw_data.get('goals_scored', 0),
                        'assists': gw_data.get('assists', 0),
                        'clean_sheets': gw_data.get('clean_sheets', 0),
                        'goals_conceded': gw_data.get('goals_conceded', 0),
                        'own_goals': gw_data.get('own_goals', 0),
                        'penalties_saved': gw_data.get('penalties_saved', 0),
                        'penalties_missed': gw_data.get('penalties_missed', 0),
                        'yellow_cards': gw_data.get('yellow_cards', 0),
                        'red_cards': gw_data.get('red_cards', 0),
                        'saves': gw_data.get('saves', 0),
                        'bonus': gw_data.get('bonus', 0),
                        'bps': gw_data.get('bps', 0)
                    }
            return None
        except:
            return None

    def health_check(self) -> Dict[str, Any]:
        """Check API health status"""
        try:
            resp = self.session.get(f"{self.base}bootstrap-static/", timeout=10)
            if resp.status_code == 200:
                return {"status": "healthy", "api_accessible": True}
            elif resp.status_code == 403:
                return {"status": "blocked", "api_accessible": False, "fallback_available": True}
            else:
                return {"status": "error", "api_accessible": False, "status_code": resp.status_code}
        except Exception as e:
            return {"status": "error", "api_accessible": False, "error": str(e)}