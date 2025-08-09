from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import joblib

from .feature_engineer import FeatureEngineer
from .model_trainer import FPLModelTrainer, XGBoostTrainer
from .pipeline import DataPipeline
from .data_prep import DataPrep
from .fpl_client import FPLClient
from .name_matching import PlayerNameMatcher
from .fbref_scraper import FBRefScraper
from .understat_scraper import UnderstatScraper


class MLPipeline:
    def __init__(self) -> None:
        Path("models/2025_26").mkdir(parents=True, exist_ok=True)
        self.engineer = FeatureEngineer()
        self.trainer = FPLModelTrainer()
        self.xgb = XGBoostTrainer()
        self.pipe = DataPipeline()
        self.prep = DataPrep()
        self.fpl = FPLClient()
        self.fbref = FBRefScraper()
        self.understat = UnderstatScraper()

    def prepare_training_data(self) -> pd.DataFrame:
        # Use latest FPL snapshot as current feature base
        self.pipe.collect_fpl_snapshots()
        datasets = self.fpl.bootstrap_static()
        players = pd.DataFrame(datasets.get("elements", []))
        teams = pd.DataFrame(datasets.get("teams", []))
        if players.empty:
            return pd.DataFrame()
        # Minimal columns to align with FeatureEngineer input path
        df = players.rename(columns={
            "web_name": "name",
            "element_type": "position_id",
            "team": "team_id",
            "id": "player_id",
            "now_cost": "price",
            "status": "fpl_status",
            "chance_of_playing_next_round": "chance_next",
        })
        df["price"] = pd.to_numeric(df["price"], errors="coerce").astype(float) / 10.0
        df["position"] = df["position_id"].map({1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"})

        # Merge team short names for convenience
        if not teams.empty:
            tdf = teams.rename(columns={"id": "team_id", "short_name": "team_short"})
            tdf["team_short"] = tdf.get("team_short", "").astype(str).str.upper()
            df = df.merge(tdf[["team_id", "team_short"]], on="team_id", how="left")

        # Merge FBref expected stats (xG/xA and per90) via name matching
        try:
            matcher = PlayerNameMatcher(df[["player_id", "name"]])
            fb = self.fbref.get_player_expected_stats()
            if not fb.empty:
                # Try to derive per90 from totals if explicit per90 missing
                name_col = next((c for c in fb.columns if str(c).lower() in ("player", "name")), None)
                if name_col is not None:
                    # Build a safe subset
                    numeric_candidates = [c for c in fb.columns if any(k in str(c).lower() for k in ["xg", "xa", "npxg", "xag"]) or str(c).strip()=="90s"]
                    fb_small = fb[[name_col] + numeric_candidates].copy()
                    fb_small = fb_small.rename(columns={name_col: "other_name"})
                    link = matcher.bulk_match(fb_small, "other_name")
                    joined = link.merge(fb_small, on="other_name", how="left")
                    # Compute per90 if possible
                    if "90s" in fb.columns or "nineties" in fb.columns:
                        n90 = pd.to_numeric(fb.get("90s", fb.get("nineties")), errors="coerce")
                        for cname in list(fb.columns):
                            lc = str(cname).lower()
                            if lc in ("xg", "xa") and n90 is not None is not False:
                                per = pd.to_numeric(fb[cname], errors="coerce") / n90.replace(0, pd.NA)
                                fb[f"{cname}_per90"] = per
                    # Aggregate by player_id
                    agg = joined.groupby("player_id").mean(numeric_only=True).reset_index()
                    df = df.merge(agg, on="player_id", how="left")
                    # Normalize column aliases
                    for col in list(df.columns):
                        lc = str(col).lower()
                        if (lc == "xg_per90" or "xg/90" in lc) and "xg_per90" not in df.columns:
                            df.rename(columns={col: "xg_per90"}, inplace=True)
                        if (lc == "xa_per90" or "xa/90" in lc) and "xa_per90" not in df.columns:
                            df.rename(columns={col: "xa_per90"}, inplace=True)
        except Exception:
            pass

        # Merge Understat expected stats as primary current-season source
        try:
            us = self.understat.get_league_players("EPL", year=int(self.fpl.current_gameweek() > 0 and 2025 or 2025))
            if not us.empty:
                # Name match to FPL names
                matcher = PlayerNameMatcher(df[["player_id", "name"]])
                link = matcher.bulk_match(us, "name")
                us_small = us[[c for c in ["name", "xg_per90", "xa_per90"] if c in us.columns]].rename(columns={"name": "other_name"})
                joined = link.merge(us_small, on="other_name", how="left")
                agg = joined.groupby("player_id").mean(numeric_only=True).reset_index()
                df = df.merge(agg, on="player_id", how="left", suffixes=("", "_us"))
                # Prefer Understat per90 if FBref missing
                for col in ["xg_per90", "xa_per90"]:
                    if col not in df.columns and f"{col}_us" in df.columns:
                        df.rename(columns={f"{col}_us": col}, inplace=True)
                    elif f"{col}_us" in df.columns:
                        df[col] = df[col].fillna(df[f"{col}_us"])  # backfill
                # Drop helper columns
                for c in ["xg_per90_us", "xa_per90_us"]:
                    if c in df.columns:
                        df.drop(columns=[c], inplace=True)
        except Exception:
            pass

        engineered = self.engineer.create_features(df, pd.DataFrame())
        return engineered

    def _team_opp_map(self) -> Tuple[dict, dict]:
        """Build mapping of team_id -> (opp_id, is_home) for current GW."""
        try:
            current_gw = self.fpl.current_gameweek()
            fixtures = pd.DataFrame(self.fpl.fixtures())
            cur = fixtures[fixtures.get("event") == current_gw]
            opp_map: dict[int, Tuple[int, bool]] = {}
            for _, r in cur.iterrows():
                th, ta = int(r.get("team_h")), int(r.get("team_a"))
                opp_map[th] = (ta, True)
                opp_map[ta] = (th, False)
            # Team strengths
            teams = pd.DataFrame(self.fpl.bootstrap_static().get("teams", []))
            t = teams.rename(columns={
                "id": "team_id",
                "strength_attack_home": "att_h",
                "strength_attack_away": "att_a",
                "strength_defence_home": "def_h",
                "strength_defence_away": "def_a",
            })
            t = t[[c for c in ["team_id", "att_h", "att_a", "def_h", "def_a"] if c in t.columns]]
            strength = {int(r.team_id): {"att_h": r.get("att_h"), "att_a": r.get("att_a"), "def_h": r.get("def_h"), "def_a": r.get("def_a")} for _, r in t.iterrows()}  # type: ignore
            return opp_map, strength
        except Exception:
            return {}, {}

    @staticmethod
    def _clean_sheet_probability(our_def: float | None, opp_att: float | None) -> float:
        """Heuristic CS probability from relative strengths. Lower opp_att and higher our_def -> higher CS.
        Returns 0..1.
        """
        try:
            if our_def is None or opp_att is None:
                return 0.25
            # Normalize strengths roughly around league average (~100)
            our = float(our_def) / 100.0
            opp = float(opp_att) / 100.0
            score = (our - opp)  # higher better
            # Map score to probability via logistic
            import math
            prob = 1.0 / (1.0 + math.exp(-3.0 * score))
            # Clamp to sensible range for FPL CS (~0.1..0.7)
            return float(max(0.1, min(0.7, prob)))
        except Exception:
            return 0.25

    @staticmethod
    def _appearance_points(start_prob: float) -> float:
        # 2 points if >=60; approximate with start probability
        return max(0.0, min(1.0, start_prob)) * 2.0

    @staticmethod
    def _goal_points(position: str) -> int:
        return 6 if position == "GKP" or position == "DEF" else (5 if position == "MID" else 4)

    @staticmethod
    def _clean_sheet_points(position: str) -> int:
        return 4 if position in ("GKP", "DEF") else (1 if position == "MID" else 0)

    def _component_expected_points(self, df: pd.DataFrame) -> pd.Series:
        # Requires columns: position, xg_per90, xa_per90, chance_next, team_id
        opp_map, strength = self._team_opp_map()

        def row_ep(r) -> float:
            try:
                raw = pd.to_numeric(r.get("chance_next"), errors="coerce")
                start_prob = float(90.0 if pd.isna(raw) else raw) / 100.0
            except Exception:
                start_prob = 0.9
            expected_minutes = 80.0 * start_prob
            xv = pd.to_numeric(r.get("xg_per90"), errors="coerce")
            av = pd.to_numeric(r.get("xa_per90"), errors="coerce")
            xg90 = 0.0 if pd.isna(xv) else float(xv)
            xa90 = 0.0 if pd.isna(av) else float(av)
            exp_goals = xg90 * (expected_minutes / 90.0)
            exp_assists = xa90 * (expected_minutes / 90.0)
            pos = str(r.get("position") or "")
            goal_pts = exp_goals * self._goal_points(pos)
            assist_pts = exp_assists * 3.0
            # Clean sheet probability based on our def vs opp att
            team_id = int(r.get("team_id")) if pd.notna(r.get("team_id")) else None
            cs_prob = 0.25
            if team_id in opp_map and team_id in strength:
                opp_id, is_home = opp_map[team_id]
                our = strength.get(team_id, {})
                opp = strength.get(opp_id, {})
                our_def = our.get("def_h" if is_home else "def_a")
                opp_att = opp.get("att_a" if is_home else "att_h")
                cs_prob = self._clean_sheet_probability(our_def, opp_att)
            cs_pts = cs_prob * self._clean_sheet_points(pos)
            # Appearance and simple bonus proxy
            appear = self._appearance_points(start_prob)
            bonus = min(1.5, 0.6 * (exp_goals + exp_assists))
            return float(appear + goal_pts + assist_pts + cs_pts + bonus)

        return df.apply(row_ep, axis=1)

    def train(self) -> Dict:
        # Prefer historical training if available
        X, y = self.prep.build_training_table()
        if not X.empty and not y.empty:
            return self.xgb.train(X, y)
        # Fallback to stub training
        df = self.prepare_training_data()
        result = self.trainer.train_model_2025_26(df)
        joblib.dump({"status": result.get("status", "stub")}, "models/2025_26/model_stub.pkl")
        return result

    def predict_current(self) -> pd.DataFrame:
        df = self.prepare_training_data()
        if df.empty:
            return df
        # ML prediction if available
        ml_pred: pd.Series | None = None
        try:
            num = df.select_dtypes(include=[float, int])
            ml_pred = self.xgb.predict(num)
        except Exception:
            ml_pred = None

        # Component-based EP
        ep_component = self._component_expected_points(df)
        ep_component = pd.to_numeric(ep_component, errors="coerce").fillna(0.0)

        # Blend
        if ml_pred is not None and len(ml_pred) == len(df):
            alpha = 0.6
            blended = (alpha * pd.to_numeric(ml_pred, errors="coerce").fillna(0.0).values) + ((1 - alpha) * ep_component.values)
        else:
            blended = ep_component.values
        # Clean NaN/inf
        import numpy as np
        blended = np.nan_to_num(blended, nan=0.0, posinf=0.0, neginf=0.0)

        base_cols = ["player_id", "name", "position", "team_id", "price", "team_short", "fpl_status", "chance_next", "xg_per90", "xa_per90"]
        keep_cols = [c for c in base_cols if c in df.columns]
        results = df[keep_cols].copy()
        # Ensure optional columns exist for downstream UI
        for opt in ["xg_per90", "xa_per90"]:
            if opt not in results.columns:
                results[opt] = pd.NA
        results["ep_component"] = ep_component.round(2)
        if ml_pred is not None and len(ml_pred) == len(df):
            results["ep_ml"] = pd.Series(ml_pred).round(2).values
        else:
            results["ep_ml"] = pd.NA
        results["predicted_points"] = pd.Series(blended).round(2).values
        preds = results.sort_values("predicted_points", ascending=False)
        preds.to_csv("data/processed/predictions_current.csv", index=False)
        return preds

