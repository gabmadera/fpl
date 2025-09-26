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
        self.cache = JsonCache(ttl_seconds=600)  # Reduced from 30min to 10min for fresher data

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

    def next_active_gameweek(self) -> int:
        """Get the next gameweek that has unfinished fixtures for predictions"""
        data = self.bootstrap_static()
        events = data.get("events", [])

        # First try to find the next unfinished gameweek
        for ev in events:
            if ev.get("is_next") and not ev.get("finished", True):
                return int(ev.get("id", 1))

        # If current is not finished, use it
        for ev in events:
            if ev.get("is_current") and not ev.get("finished", True):
                return int(ev.get("id", 1))

        # Otherwise find first unfinished gameweek
        for ev in events:
            if not ev.get("finished", True):
                return int(ev.get("id", 1))

        # Fallback to current/next
        return self.current_gameweek()

    def event_live(self, gameweek: int) -> Dict[str, Any]:
        """Get live event data for a specific gameweek"""
        return self._get(f"event/{gameweek}/live/")

    def get_player_gameweek_stats(self, player_id: int, gameweek: int) -> Optional[Dict[str, Any]]:
        """Get detailed stats for a player in a specific gameweek"""
        try:
            element_data = self.element_summary(player_id)
            history = element_data.get('history', [])

            # Find stats for the specific gameweek
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
                        'bps': gw_data.get('bps', 0),  # Bonus Point System score
                        'influence': gw_data.get('influence', 0),
                        'creativity': gw_data.get('creativity', 0),
                        'threat': gw_data.get('threat', 0),
                        'ict_index': gw_data.get('ict_index', 0),
                        'selected': gw_data.get('selected', 0),
                        'transfers_in': gw_data.get('transfers_in', 0),
                        'transfers_out': gw_data.get('transfers_out', 0)
                    }
            return None
        except Exception as e:
            self.security.logger.error(f"Failed to get player {player_id} stats for GW {gameweek}: {e}")
            return None

    def get_current_event_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the current/next gameweek"""
        try:
            data = self.bootstrap_static()
            events = data.get("events", [])

            # Find current event
            for event in events:
                if event.get("is_current"):
                    return event

            # If no current, find next
            for event in events:
                if event.get("is_next"):
                    return event

            return None
        except Exception as e:
            self.security.logger.error(f"Failed to get current event info: {e}")
            return None

    def validate_fixture_freshness(self) -> Dict[str, Any]:
        """Validate that fixture data is fresh and suitable for predictions"""
        try:
            current_gw = self.current_gameweek()
            next_gw = self.next_active_gameweek()
            fixtures_data = self.fixtures()

            validation = {
                "status": "ok",
                "current_gw": current_gw,
                "next_active_gw": next_gw,
                "issues": [],
                "recommendations": []
            }

            # Check if current GW is finished but we're still using it
            current_event = self.get_current_event_info()
            if current_event and current_event.get("finished", False) and current_gw == next_gw:
                validation["issues"].append("Current gameweek is finished but no next gameweek detected")
                validation["status"] = "warning"

            # Count upcoming fixtures
            import pandas as pd
            fixtures_df = pd.DataFrame(fixtures_data)
            if not fixtures_df.empty:
                upcoming_fixtures = fixtures_df[
                    (fixtures_df.get("event", 0) >= next_gw) &
                    (~fixtures_df.get("finished", True))
                ]
                validation["upcoming_fixtures_count"] = len(upcoming_fixtures)

                if len(upcoming_fixtures) == 0:
                    validation["issues"].append("No upcoming unfinished fixtures found")
                    validation["status"] = "error"
                elif len(upcoming_fixtures) < 10:
                    validation["issues"].append(f"Only {len(upcoming_fixtures)} upcoming fixtures - may be incomplete")
                    validation["status"] = "warning"

            # Cache age check
            cache_key = "fixtures/"
            if hasattr(self.cache, '_cache') and cache_key in self.cache._cache:
                cache_entry = self.cache._cache[cache_key]
                if hasattr(cache_entry, 'created_at'):
                    import time
                    cache_age = time.time() - cache_entry.created_at
                    validation["cache_age_seconds"] = int(cache_age)
                    if cache_age > 900:  # 15 minutes
                        validation["recommendations"].append("Consider clearing fixture cache - data may be stale")

            return validation

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "issues": ["Failed to validate fixture data"],
                "recommendations": ["Check FPL API connectivity and data structure"]
            }

    def refresh_fixture_cache(self) -> bool:
        """Force refresh of fixture cache to get latest data"""
        try:
            # Clear fixture cache entries
            cache_keys_to_clear = ["fixtures/", "bootstrap-static/"]

            if hasattr(self.cache, '_cache'):
                for key in cache_keys_to_clear:
                    if key in self.cache._cache:
                        del self.cache._cache[key]
                        self.security.logger.info(f"Cleared cache for {key}")

            # Force fresh fetch
            self.fixtures()
            self.bootstrap_static()

            return True
        except Exception as e:
            self.security.logger.error(f"Failed to refresh fixture cache: {e}")
            return False

