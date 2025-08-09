from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional


class FeatureEngineer:
    def __init__(self) -> None:
        self.lookback_windows = [3, 5, 8, 12]

    def create_features(self, player_data: pd.DataFrame, fixture_data: pd.DataFrame) -> pd.DataFrame:
        df = player_data.copy()
        if "predicted_points" not in df.columns:
            df["predicted_points"] = 0.0
        # Ensure price exists and is numeric
        if "price" not in df.columns:
            if "value" in df.columns:
                with pd.option_context('mode.use_inf_as_na', True):
                    df["price"] = pd.to_numeric(df["value"], errors="coerce") / 10.0
            else:
                df["price"] = 10.0
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        if df["price"].isna().all():
            df["price"] = 10.0
        else:
            df["price"].fillna(df["price"].median(), inplace=True)

        # Simple price-based proxy features as placeholders
        try:
            uniques = max(1, int(df["price"].nunique()))
            df["price_band"] = pd.qcut(df["price"], q=min(5, uniques), labels=False, duplicates="drop")
        except Exception:
            df["price_band"] = 0
        df["is_premium"] = df["price"] >= df["price"].quantile(0.85)
        # Placeholder defensive contribution probability
        df["dc_point_probability"] = np.where(df["position"] == "DEF", 0.2, 0.05)
        # Fixture-based features (very basic): next opponent difficulty proxy
        if not fixture_data.empty and "team_id" in df.columns:
            opp_difficulty = self._compute_next_fixture_difficulty(fixture_data)
            df = df.merge(opp_difficulty, on="team_id", how="left")
            df["opp_difficulty"].fillna(df["opp_difficulty"].median() if not df["opp_difficulty"].isna().all() else 3, inplace=True)
        else:
            df["opp_difficulty"] = 3
        return df

    def _compute_next_fixture_difficulty(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        # Take nearest unfinished fixture; average symmetric difficulty
        upcoming = fixtures.copy()
        if "finished" in upcoming.columns:
            upcoming = upcoming[~upcoming["finished"].astype(bool)]
        cols = ["team_h", "team_a", "team_h_difficulty", "team_a_difficulty"]
        for c in cols:
            if c not in upcoming.columns:
                upcoming[c] = np.nan
        home_part = upcoming[["team_h", "team_a_difficulty"]].rename(columns={"team_h": "team_id", "team_a_difficulty": "opp_difficulty"})
        away_part = upcoming[["team_a", "team_h_difficulty"]].rename(columns={"team_a": "team_id", "team_h_difficulty": "opp_difficulty"})
        combined = pd.concat([home_part, away_part], axis=0, ignore_index=True)
        grouped = combined.groupby("team_id", as_index=False)["opp_difficulty"].median()
        return grouped

