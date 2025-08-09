from __future__ import annotations

import time
from typing import Dict
import pandas as pd
import requests

from .config import config
from .security import SecurityManager


class DataCollector:
    def __init__(self) -> None:
        self.security = SecurityManager()
        self.session = self.security.session

    def fetch_fpl_data(self, endpoint: str) -> Dict:
        """Fetch data from FPL API with basic rate limiting and retries."""
        if not self.security.rate_limit_check():
            time.sleep(5)
        url = f"{config.FPL_BASE_URL}{endpoint}"
        time.sleep(float(config.RATE_LIMIT_DELAY))
        response = self.session.get(url, timeout=20)
        response.raise_for_status()
        return response.json()

    def collect_all_data(self) -> dict[str, pd.DataFrame]:
        """Collect a minimal set of FPL data and return processed frames."""
        print("Starting data collection...")
        bootstrap = self.fetch_fpl_data("bootstrap-static/")
        fixtures = self.fetch_fpl_data("fixtures/")

        datasets: dict[str, pd.DataFrame] = {
            "players": self._process_player_data_2025_26(bootstrap),
            "fixtures": self._process_fixture_data(fixtures),
            "teams": self._process_teams(bootstrap),
        }
        print("Data collection complete")
        return datasets

    def _process_player_data_2025_26(self, bootstrap: Dict) -> pd.DataFrame:
        if not bootstrap or not bootstrap.get("elements"):
            return pd.DataFrame()
        df = pd.DataFrame(bootstrap["elements"]).copy()
        df.rename(
            columns={
                "web_name": "name",
                "element_type": "position_id",
                "team": "team_id",
                "now_cost": "price",
                "id": "player_id",
                "chance_of_playing_next_round": "chance_next",
                "status": "fpl_status",
            },
            inplace=True,
        )
        df["price"] = df["price"].astype(float) / 10.0
        # Minimal derived columns (placeholders)
        position_map = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
        df["position"] = df["position_id"].map(position_map)
        # Starting probability heuristic from FPL status/chance
        df["chance_next"] = pd.to_numeric(df.get("chance_next"), errors="coerce")
        df["starting_probability"] = 0.9
        df.loc[df["fpl_status"].isin(["s", "i"]), "starting_probability"] = 0.05
        df.loc[df["chance_next"].notna(), "starting_probability"] = (df.loc[df["chance_next"].notna(), "chance_next"] / 100.0).clip(0.0, 1.0)
        df["predicted_points"] = 0.0  # filled by model later
        df["afcon_risk"] = False      # TODO: implement nationality cross-ref
        return df[["player_id", "name", "position", "team_id", "price", "starting_probability", "fpl_status", "chance_next", "predicted_points", "afcon_risk"]]

    def _process_fixture_data(self, fixtures: Dict) -> pd.DataFrame:
        if not fixtures:
            return pd.DataFrame()
        df = pd.DataFrame(fixtures).copy()
        keep_cols = [
            "id",
            "event",
            "team_h",
            "team_a",
            "team_h_difficulty",
            "team_a_difficulty",
            "kickoff_time",
            "finished",
        ]
        existing = [c for c in keep_cols if c in df.columns]
        return df[existing]

    def _process_teams(self, bootstrap: Dict) -> pd.DataFrame:
        teams = pd.DataFrame(bootstrap.get("teams", []))
        if teams.empty:
            return teams
        # Keep id and name
        cols = [c for c in ["id", "name", "short_name", "strength_overall_home", "strength_overall_away"] if c in teams.columns]
        return teams[cols]

