from __future__ import annotations

import glob
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from .feature_engineer import FeatureEngineer
from .fbref_scraper import FBRefScraper
from .fpl_client import FPLClient
from .name_matching import PlayerNameMatcher
from .fpl_client import FPLClient
from .understat_scraper import UnderstatScraper


class DataPrep:
    """Prepares training data from historical GW CSVs if available.

    Expected folder layout (vaastav-style):
      data/historical/<season>/gws/gw1.csv ... gw38.csv

    If no GW data is present, returns empty frames so callers can gracefully skip.
    """

    def __init__(self) -> None:
        self.engineer = FeatureEngineer()
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        self.fbref = FBRefScraper()
        self.fpl_client = FPLClient()
        self.understat = UnderstatScraper()

    def _find_gw_csvs(self) -> List[str]:
        patterns = [
            "data/historical/*/gws/gw*.csv",
            "data/historical/gws/*/gw*.csv",
        ]
        files: List[str] = []
        for pat in patterns:
            files.extend(glob.glob(pat))
        return sorted(files)

    def load_historical_gw_frames(self) -> pd.DataFrame:
        files = self._find_gw_csvs()
        if not files:
            return pd.DataFrame()
        frames: List[pd.DataFrame] = []
        for f in files:
            try:
                df = pd.read_csv(f)
                # Expect columns like: name, position, team, total_points, minutes, value, etc.
                # Normalize minimal columns
                if "element" in df.columns:
                    df = df.rename(columns={"element": "player_id"})
                if "web_name" in df.columns and "name" not in df.columns:
                    df = df.rename(columns={"web_name": "name"})
                if "value" in df.columns and "price" not in df.columns:
                    # Vaastav value is in tenths usually
                    try:
                        df["price"] = df["value"].astype(float) / 10.0
                    except Exception:
                        df["price"] = np.nan
                frames.append(df)
            except Exception:
                continue
        if not frames:
            return pd.DataFrame()
        all_gw = pd.concat(frames, ignore_index=True, sort=False)
        return all_gw

    def build_training_table(self) -> Tuple[pd.DataFrame, pd.Series]:
        all_gw = self.load_historical_gw_frames()
        if all_gw.empty:
            return pd.DataFrame(), pd.Series(dtype=float)

        # Minimal normalization
        # Build player static frame (last known price, position text if present)
        static_cols = [c for c in ["player_id", "name", "position", "team", "price", "value"] if c in all_gw.columns]
        static = all_gw[static_cols].dropna(subset=[col for col in static_cols if col != "price"]).drop_duplicates(subset=["player_id"])
        if "team" in static.columns:
            static = static.rename(columns={"team": "team_id"})
        # Derive price from value if missing
        if "price" not in static.columns and "value" in static.columns:
            try:
                static["price"] = static["value"].astype(float) / 10.0
            except Exception:
                static["price"] = np.nan

        # Aggregate per player per GW
        key_cols = []
        if "player_id" in all_gw.columns:
            key_cols.append("player_id")
        if "round" in all_gw.columns:
            key_cols.append("round")
        elif "gw" in all_gw.columns:
            key_cols.append("gw")

        if not key_cols:
            return pd.DataFrame(), pd.Series(dtype=float)

        points_col = "total_points" if "total_points" in all_gw.columns else ("points" if "points" in all_gw.columns else None)
        if points_col is None:
            return pd.DataFrame(), pd.Series(dtype=float)

        keep_extra = [c for c in [
            "minutes", "goals_scored", "assists", "clean_sheets", "bps", "influence", "creativity", "threat",
            "was_home"
        ] if c in all_gw.columns]
        # Include opponent team id if present in GW CSVs
        extra_cols = keep_extra + (["opponent_team"] if "opponent_team" in all_gw.columns else [])
        per_gw = all_gw[key_cols + [points_col, "price"] + extra_cols].copy()
        per_gw = per_gw.rename(columns={points_col: "points"})
        per_gw = per_gw.sort_values(key_cols)

        # Next-GW target per player
        if "player_id" not in per_gw.columns:
            return pd.DataFrame(), pd.Series(dtype=float)
        per_gw["target"] = per_gw.groupby("player_id")["points"].shift(-1)
        per_gw = per_gw.dropna(subset=["target"])  # rows with known next gw points

        # Join static
        train = per_gw.merge(static, on="player_id", how="left")

        # Add opponent strength from FPL bootstrap teams if possible
        try:
            fpl = FPLClient()
            bs = fpl.bootstrap_static()
            tdf = pd.DataFrame(bs.get("teams", []))
            if not tdf.empty and "opponent_team" in train.columns:
                tdf = tdf.rename(columns={
                    "id": "opponent_team",
                    "strength_defence_home": "opp_def_home",
                    "strength_defence_away": "opp_def_away",
                    "strength_overall_home": "opp_overall_home",
                    "strength_overall_away": "opp_overall_away",
                })
                train = train.merge(tdf[[c for c in ["opponent_team", "opp_def_home", "opp_def_away", "opp_overall_home", "opp_overall_away"] if c in tdf.columns]], on="opponent_team", how="left")
                # Pick home/away appropriate opponent defense
                if "was_home" in train.columns:
                    train["opp_def_strength"] = train.apply(
                        lambda r: r.get("opp_def_home") if r.get("was_home") else r.get("opp_def_away"), axis=1
                    )
                else:
                    train["opp_def_strength"] = train.get("opp_overall_home")
        except Exception:
            pass

        # Build rolling features by player
        def add_roll(df: pd.DataFrame, col: str, windows=(3, 5)) -> pd.DataFrame:
            if col not in df.columns:
                return df
            for w in windows:
                df[f"{col}_sum_{w}"] = (
                    df.groupby("player_id")[col].rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
                )
                df[f"{col}_avg_{w}"] = (
                    df.groupby("player_id")[col].rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
                )
            return df

        for metric in ["points", "minutes", "goals_scored", "assists", "bps", "influence", "creativity", "threat"]:
            train = add_roll(train, metric)

        # Encode home/away
        if "was_home" in train.columns:
            train["home_flag"] = train["was_home"].astype(int)

        # Merge FBref expected metrics (xG/xA) via name matching
        try:
            fpl = FPLClient()
            bs = fpl.bootstrap_static()
            players_static = pd.DataFrame(bs.get("elements", [])).rename(columns={"web_name": "name", "id": "player_id"})
            matcher = PlayerNameMatcher(players_static[["player_id", "name"]])
            fb = self.fbref.get_player_expected_stats()
            if not fb.empty:
                # Heuristic: find name column and xG/xA columns
                name_col = next((c for c in fb.columns if str(c).lower() in ("player", "name")), None)
                if name_col:
                    link = matcher.bulk_match(fb, name_col)
                    xcols = [c for c in fb.columns if any(k in str(c) for k in ["xG", "Expected Goals", "xA", "Expected Assists", "xG/90", "xA/90"])][:8]
                    if xcols:
                        fb_small = fb[[name_col] + xcols].rename(columns={name_col: "other_name"})
                        joined = link.merge(fb_small, on="other_name", how="left")
                        agg = joined.groupby("player_id").mean(numeric_only=True).reset_index()
                        train = train.merge(agg, on="player_id", how="left")
                        # Normalize common column names if present
                        for col in list(train.columns):
                            lc = str(col).lower()
                            if "xg/90" in lc and "xg_per90" not in train.columns:
                                train.rename(columns={col: "xg_per90"}, inplace=True)
                            if "xa/90" in lc and "xa_per90" not in train.columns:
                                train.rename(columns={col: "xa_per90"}, inplace=True)
        except Exception:
            pass

        # Basic engineered features from current snapshot-like static (price bands etc.)
        features = self.engineer.create_features(train.rename(columns={"team": "team_id"}), pd.DataFrame())
        # Enrich with FBref expected/defensive aggregates if available via name matching
        try:
            fpl_bootstrap = self.fpl_client.get_bootstrap_data()
            players_static = pd.DataFrame(fpl_bootstrap.get("elements", []))
            players_static = players_static.rename(columns={"web_name": "name", "id": "player_id"})
            matcher = PlayerNameMatcher(players_static[["player_id", "name"]])
            fb_expected = self.fbref.get_player_expected_stats()
            if not fb_expected.empty:
                # Try common name column variants
                name_col = next((c for c in fb_expected.columns if str(c).lower() in ("player", "name")), None)
                if name_col:
                    link = matcher.bulk_match(fb_expected, name_col)
                    if not link.empty:
                        fb_cols = [c for c in fb_expected.columns if any(k in str(c) for k in ["xG", "xA", "Expected"])][:6]
                        fb_small = fb_expected[[name_col] + fb_cols].copy()
                        fb_small = fb_small.rename(columns={name_col: "other_name"})
                        fb_join = link.merge(fb_small, on="other_name", how="left")
                        agg = fb_join.groupby("player_id").mean(numeric_only=True).reset_index()
                        features = features.merge(agg, on="player_id", how="left")
        except Exception:
            pass
        # Enrich with Understat expected per-90 (historical priors)
        try:
            # Use previous season as a proxy prior if available
            us = self.understat.get_league_players("EPL", year=2024)
            if not us.empty:
                fpl_bootstrap = self.fpl_client.get_bootstrap_data()
                players_static = pd.DataFrame(fpl_bootstrap.get("elements", []))
                players_static = players_static.rename(columns={"web_name": "name", "id": "player_id"})
                matcher = PlayerNameMatcher(players_static[["player_id", "name"]])
                link = matcher.bulk_match(us, "name")
                us_small = us[[c for c in ["name", "xg_per90", "xa_per90"] if c in us.columns]].rename(columns={"name": "other_name"})
                joined = link.merge(us_small, on="other_name", how="left")
                agg = joined.groupby("player_id").mean(numeric_only=True).reset_index()
                features = features.merge(agg, on="player_id", how="left", suffixes=("", "_us"))
                for col in ["xg_per90", "xa_per90"]:
                    if f"{col}_us" in features.columns:
                        features[col] = features.get(col).fillna(features[f"{col}_us"]) if col in features.columns else features[f"{col}_us"]
                        features.drop(columns=[f"{col}_us"], inplace=True)
        except Exception:
            pass
        # Select numeric features
        X = features.select_dtypes(include=[np.number]).copy()
        y = train["target"].astype(float)
        # Persist training snapshot for debugging
        features_out = features.copy()
        features_out["target"] = y.values
        features_out.to_csv("data/processed/training_table_preview.csv", index=False)
        return X, y

