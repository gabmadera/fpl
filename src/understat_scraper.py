from __future__ import annotations

import re
import json
from typing import Optional
import requests
import pandas as pd
from bs4 import BeautifulSoup


class UnderstatScraper:
    """Lightweight scraper for Understat league players page.

    Extracts playersData JSON from the league page and returns a DataFrame
    with per-90 and aggregate expected stats.
    """

    def __init__(self) -> None:
        self.base = "https://understat.com"
        self.headers = {"User-Agent": "Mozilla/5.0 (compatible; FPL-AI/1.0)"}

    def _extract_players_json(self, html: str) -> Optional[list]:
        # Pattern A: playersData = JSON.parse('...');
        m = re.search(r"playersData\s*=\s*JSON.parse\('(.*?)'\)\s*;", html, re.S)
        if m:
            raw = m.group(1)
            try:
                # Unescape JavaScript string into JSON
                txt = bytes(raw, "utf-8").decode("unicode_escape")
                return json.loads(txt)
            except Exception:
                pass
        # Pattern B: playersData = [ {...} ];
        m2 = re.search(r"playersData\s*=\s*(\[\{.*?\}\])\s*;", html, re.S)
        if m2:
            try:
                return json.loads(m2.group(1))
            except Exception:
                pass
        return None

    def get_league_players(self, league: str = "EPL", year: int = 2025) -> pd.DataFrame:
        # Try current year, then previous seasons as fallback (Understat uses start year)
        years = [year, year - 1, 2024, 2023]  # Extended fallback for pre-season
        data = None
        
        for y in years:
            try:
                url = f"{self.base}/league/{league}/{y}"
                resp = requests.get(url, headers=self.headers, timeout=30)
                resp.raise_for_status()
                data = self._extract_players_json(resp.text)
                if data and len(data) > 100:  # Ensure we got meaningful data
                    print(f"Understat: Successfully loaded {len(data)} players from {y} season")
                    break
                elif data:
                    print(f"Understat: Limited data from {y} ({len(data)} players), trying previous season")
            except Exception as e:
                print(f"Understat: Failed to fetch {y} data: {e}")
                continue
                
        if not data:
            print("Understat: No data available from any season")
            return pd.DataFrame()
        df = pd.json_normalize(data)
        # Typical keys: player_name, team_title, position, time, games, xG, xA, xG90, xA90
        # Normalize column names to consistent lowercase
        df.columns = [str(c).strip() for c in df.columns]
        # Compute per90 if missing (using modern pandas approach)
        if "xG90" not in df.columns and "xG" in df.columns and "time" in df.columns:
            xg_vals = pd.to_numeric(df["xG"], errors="coerce")
            time_vals = pd.to_numeric(df["time"], errors="coerce").replace(0, pd.NA)
            df["xG90"] = (xg_vals / time_vals * 90.0).replace([float('inf'), -float('inf')], pd.NA)
            
        if "xA90" not in df.columns and "xA" in df.columns and "time" in df.columns:
            xa_vals = pd.to_numeric(df["xA"], errors="coerce")
            time_vals = pd.to_numeric(df["time"], errors="coerce").replace(0, pd.NA)
            df["xA90"] = (xa_vals / time_vals * 90.0).replace([float('inf'), -float('inf')], pd.NA)
        # Standard output columns
        keep = [
            c for c in [
                "player_name", "team_title", "position", "games", "time", "xG", "xA", "xG90", "xA90",
            ] if c in df.columns
        ]
        out = df[keep].copy()
        out.rename(columns={"player_name": "name", "xG90": "xg_per90", "xA90": "xa_per90"}, inplace=True)
        
        # Clean the data - ensure numeric columns are properly formatted
        for col in ["xg_per90", "xa_per90", "xG", "xA"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors='coerce')
        
        print(f"Understat: Returning {len(out)} players with stats")
        return out


