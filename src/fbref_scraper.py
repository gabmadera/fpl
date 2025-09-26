from __future__ import annotations

from typing import Dict, List
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from io import StringIO
from .config import config


class FBRefScraper:
    """Minimal FBref scraper for league tables (xG/xA and defensive stats).

    NOTE: FBref structure can change. This is a basic, resilient approach reading
    visible HTML tables without JavaScript. For dynamic pages, use Selenium as fallback.
    """

    def __init__(self) -> None:
        self.base = config.FBREF_BASE_URL.rstrip("/")
        self.headers = {"User-Agent": "Mozilla/5.0 (compatible; FPL-AI/1.0)"}

    def fetch_league_table(self, path_suffix: str) -> pd.DataFrame:
        url = f"{self.base}/{path_suffix.lstrip('/')}"
        time.sleep(1.0)
        resp = requests.get(url, headers=self.headers, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        # Include commented tables
        html_parts = [resp.text]
        for comment in soup.find_all(string=lambda t: isinstance(t, type(soup.string)) and "table" in t.lower()):
            try:
                html_parts.append(comment)
            except Exception:
                pass
        combined = "\n".join(html_parts)
        try:
            tables = pd.read_html(StringIO(combined))
        except ValueError:
            tables = []
        return tables[0] if tables else pd.DataFrame()

    def get_player_expected_stats(self) -> pd.DataFrame:
        """Fetch expected goals/assists per player table.

        Primary target: FBref expected players page for comp 9 (Premier League):
        /en/comps/9/expected/players/Premier-League-Stats
        Fallback to generic comp page if structure changes.
        """
        candidates = [
            "expected/players/Premier-League-Stats",  # /en/comps/9/<this>
            "Premier-League-Stats",
            "9/Premier-League-Stats",
        ]
        for suffix in candidates:
            try:
                df = self.fetch_league_table(suffix)
                if not df.empty and any(col for col in df.columns if ("xG" in str(col) or "xA" in str(col) or "Expected" in str(col))):
                    # Normalize common columns if present
                    # Expect typical columns: Player, Squad, xG, npxG, xAG, npxG+xAG, 90s
                    for c in list(df.columns):
                        if str(c).strip() == "90s":
                            df.rename(columns={c: "nineties"}, inplace=True)
                    return df
            except Exception:
                continue
        return pd.DataFrame()

    def get_defensive_contribution_stats(self) -> pd.DataFrame:
        """Fetch table containing clearances/blocks/interceptions/tackles.

        We’ll search for columns that indicate CBIT/CBIRT-like stats.
        """
        # Try to find a defensive table specifically by scanning multiple tables
        base_df = self.get_player_expected_stats()
        candidate = base_df
        # Attempt to find a better defensive table from the same page if any
        if not base_df.empty:
            # Heuristic renames
            rename_map = {}
            for col in base_df.columns:
                col_str = str(col)
                if "Tackles" in col_str:
                    rename_map[col] = "tackles"
                if "Blocks" in col_str:
                    rename_map[col] = "blocks"
                if "Int" in col_str or "Interceptions" in col_str:
                    rename_map[col] = "interceptions"
                if "Clr" in col_str or "Clearances" in col_str:
                    rename_map[col] = "clearances"
                if "Recov" in col_str or "Ball Recoveries" in col_str:
                    rename_map[col] = "recoveries"
            if rename_map:
                candidate = base_df.rename(columns=rename_map)
        return candidate if candidate is not None else pd.DataFrame()

