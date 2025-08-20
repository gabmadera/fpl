from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import numpy as np
import joblib

from .feature_engineer import FeatureEngineer
from .model_trainer import FPLModelTrainer, XGBoostTrainer
from .pipeline import DataPipeline
from .data_prep import DataPrep
from .fpl_client import FPLClient
from .name_matching import PlayerNameMatcher
from .fbref_scraper import FBRefScraper
from .understat_scraper import UnderstatScraper
from .alternative_data_sources import AlternativeDataSources


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
        self.alt_sources = AlternativeDataSources()

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

        # Enhanced FBref stats integration with better error handling
        try:
            matcher = PlayerNameMatcher(df[["player_id", "name"]])
            fb = self.fbref.get_player_expected_stats()
            
            if not fb.empty:
                name_col = next((c for c in fb.columns if str(c).lower() in ("player", "name")), None)
                if name_col is not None:
                    # Enhanced stat extraction
                    stat_cols = []
                    for col in fb.columns:
                        col_lower = str(col).lower()
                        if any(k in col_lower for k in ["xg", "xa", "npxg", "xag", "sca", "gca", "shot", "pass"]) or "90s" in col_lower:
                            stat_cols.append(col)
                    
                    if stat_cols:
                        fb_enhanced = fb[[name_col] + stat_cols].copy()
                        fb_enhanced = fb_enhanced.rename(columns={name_col: "other_name"})
                        
                        # Better name matching with fuzzy matching
                        link = matcher.bulk_match(fb_enhanced, "other_name")
                        joined = link.merge(fb_enhanced, on="other_name", how="left")
                        
                        # Enhanced per90 calculations
                        ninety_cols = [c for c in fb_enhanced.columns if "90s" in str(c).lower()]
                        if ninety_cols:
                            n90_col = ninety_cols[0]
                            n90 = pd.to_numeric(fb_enhanced[n90_col], errors="coerce")
                            
                            for col in fb_enhanced.columns:
                                if col != n90_col and col != "other_name":
                                    values = pd.to_numeric(fb_enhanced[col], errors="coerce")
                                    per90_values = values / n90.replace(0, pd.NA)
                                    
                                    # Create standardized column names
                                    col_clean = str(col).lower().replace(" ", "_")
                                    if "xg" in col_clean and "per90" not in col_clean:
                                        joined[f"{col_clean}_per90"] = per90_values
                                    elif "xa" in col_clean and "per90" not in col_clean:
                                        joined[f"{col_clean}_per90"] = per90_values
                        
                        # Aggregate with better handling of multiple matches
                        agg_dict = {}
                        numeric_cols = joined.select_dtypes(include=[np.number]).columns
                        
                        for col in numeric_cols:
                            if col != "player_id":
                                # Use weighted average if we have multiple data sources
                                agg_dict[col] = 'mean'
                        
                        if agg_dict:
                            agg = joined.groupby("player_id").agg(agg_dict).reset_index()
                            df = df.merge(agg, on="player_id", how="left")
                            
                            # Standardize column names
                            column_mapping = {}
                            for col in df.columns:
                                col_lower = str(col).lower()
                                if "xg" in col_lower and "per90" in col_lower and "xg_per90" not in df.columns:
                                    column_mapping[col] = "xg_per90"
                                elif "xa" in col_lower and "per90" in col_lower and "xa_per90" not in df.columns:
                                    column_mapping[col] = "xa_per90"
                            
                            if column_mapping:
                                df = df.rename(columns=column_mapping)
                                
        except Exception as e:
            print(f"FBref integration failed: {e}")

        # Enhanced Understat integration with current season focus
        try:
            current_year = 2025  # Current season
            us = self.understat.get_league_players("EPL", year=current_year)
            us_prev = self.understat.get_league_players("EPL", year=current_year - 1)
            
            if not us.empty:
                matcher = PlayerNameMatcher(df[["player_id", "name"]])
                link = matcher.bulk_match(us, "name")
                
                # Get comprehensive Understat stats (include minutes/time)
                us_stats = ["name", "xg_per90", "xa_per90", "npxg_per90", "shots_per90", "key_passes_per90", "time", "games"]
                available_stats = [col for col in us_stats if col in us.columns]
                
                if available_stats:
                    us_selected = us[available_stats].rename(columns={"name": "other_name"})
                    joined = link.merge(us_selected, on="other_name", how="left")
                    
                    # Enhanced aggregation with recency weighting
                    numeric_stats = [col for col in available_stats if col != "other_name"]
                    if numeric_stats:
                        # Convert to numeric first
                        for col in numeric_stats:
                            if col in joined.columns:
                                joined[col] = pd.to_numeric(joined[col], errors='coerce')
                        
                        agg_dict = {col: 'mean' for col in numeric_stats}
                        agg = joined.groupby("player_id").agg(agg_dict).reset_index()
                    else:
                        agg = pd.DataFrame()
                    
                    if not agg.empty:
                        df = df.merge(agg, on="player_id", how="left", suffixes=("", "_us"))
                    
                    # Intelligent stat prioritization: Understat (current) > FBref (recent)
                    stat_priority = [("xg_per90", "xg_per90_us"), ("xa_per90", "xa_per90_us")]
                    
                    for primary, understat in stat_priority:
                        if f"{understat}" in df.columns:
                            if primary not in df.columns:
                                df[primary] = df[understat]
                            else:
                                # Weighted blend: 70% Understat (more current), 30% FBref (more complete)
                                us_vals = pd.to_numeric(df[understat], errors="coerce")
                                fb_vals = pd.to_numeric(df[primary], errors="coerce")
                                
                                # Create blended values where both exist
                                both_exist = us_vals.notna() & fb_vals.notna()
                                df.loc[both_exist, primary] = (0.7 * us_vals[both_exist] + 0.3 * fb_vals[both_exist])
                                
                                # Fill missing values
                                df[primary] = df[primary].fillna(us_vals).fillna(fb_vals)
                            
                            # Clean up temporary columns
                            df = df.drop(columns=[understat])
                    
                    # Add derived metrics if we have the base stats
                    if "xg_per90" in df.columns and "xa_per90" in df.columns:
                        df["total_threat_per90"] = (pd.to_numeric(df["xg_per90"], errors="coerce").fillna(0) + 
                                                   pd.to_numeric(df["xa_per90"], errors="coerce").fillna(0))
                        
                        # Position-adjusted threat scores
                        pos_multipliers = {"FWD": 1.2, "MID": 1.0, "DEF": 0.8, "GKP": 0.5}
                        df["adjusted_threat"] = df["total_threat_per90"] * df["position"].map(pos_multipliers).fillna(1.0)
            # Blend previous season per90 if available to reduce early-season noise
            if not us_prev.empty:
                matcher_prev = PlayerNameMatcher(df[["player_id", "name"]])
                link_prev = matcher_prev.bulk_match(us_prev, "name")
                prev_cols = [c for c in ["name", "xg_per90", "xa_per90", "time"] if c in us_prev.columns]
                if prev_cols:
                    prev_sel = us_prev[prev_cols].rename(columns={"name": "other_name", "xg_per90": "xg_per90_prev", "xa_per90": "xa_per90_prev", "time": "time_prev"})
                    prev_join = link_prev.merge(prev_sel, on="other_name", how="left")
                    prev_agg = prev_join.groupby("player_id").agg({"xg_per90_prev": "mean", "xa_per90_prev": "mean", "time_prev": "mean"}).reset_index()
                    df = df.merge(prev_agg, on="player_id", how="left")
                    # If current per90 missing or low-minutes, blend with previous (60/40)
                    for stat in [("xg_per90", "xg_per90_prev"), ("xa_per90", "xa_per90_prev")]:
                        cur, prev = stat
                        if cur in df.columns and prev in df.columns:
                            cur_vals = pd.to_numeric(df[cur], errors="coerce")
                            prev_vals = pd.to_numeric(df[prev], errors="coerce")
                            # Weight by whether we have current minutes >= 270
                            cur_minutes = pd.to_numeric(df.get("time"), errors="coerce").fillna(0)
                            w_cur = (cur_minutes >= 270).astype(float) * 0.6 + (cur_minutes < 270).astype(float) * 0.3
                            blended = w_cur.fillna(0.3) * cur_vals.fillna(0) + (1 - w_cur.fillna(0.7)) * prev_vals.fillna(0)
                            df[cur] = blended.where(cur_vals.notna() | prev_vals.notna(), df[cur])
            
            # Apply minutes-aware shrinkage and positional clamps on xG/90 and xA/90
            df = self._apply_xg_xa_shrinkage(df)
                        
        except Exception as e:
            print(f"Understat integration failed: {e}")

        # Apply alternative data sources for missing xG/xA data  
        missing_data = df["xg_per90"].isna().sum() + df["xa_per90"].isna().sum()
        if missing_data > len(df):  # Most players missing data
            print(f"Primary sources incomplete ({missing_data}/{len(df)*2} missing). Using fallback sources...")
            df = self.alt_sources.get_combined_fallback_data(df)
        
        # Enhanced feature engineering with fixture data
        try:
            fixture_data = pd.DataFrame(self.fpl.fixtures())
        except:
            fixture_data = pd.DataFrame()
            
        engineered = self.engineer.create_features(df, fixture_data)
        
        # Add additional computed features
        if "total_points" in engineered.columns and "minutes" in engineered.columns:
            # Points per minute efficiency
            minutes = pd.to_numeric(engineered["minutes"], errors="coerce")
            points = pd.to_numeric(engineered["total_points"], errors="coerce")
            engineered["points_per_minute"] = points / minutes.replace(0, pd.NA)
            
        # Form-based adjustments
        if "form" in engineered.columns:
            form_numeric = pd.to_numeric(engineered["form"], errors="coerce")
            engineered["form_multiplier"] = 0.8 + (form_numeric / 10.0)  # Scale form to multiplier
            
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
    def _predict_minutes(row: pd.Series) -> float:
        """Simple xMins model using chance_next, fpl_status, and recent minutes if available."""
        try:
            chance = pd.to_numeric(row.get("chance_next"), errors="coerce")
            status = str(row.get("fpl_status", "a")).lower()
            # Base start probability from chance/status
            if status in ["i", "s", "u"]:
                start_prob = 0.05
            elif status == "d":
                start_prob = 0.5 if pd.isna(chance) else float(chance) / 100.0
            else:
                start_prob = 0.9 if pd.isna(chance) else float(chance) / 100.0
            # Recent minutes proxy from rolling features if present
            recent = pd.to_numeric(row.get("minutes_form_3"), errors="coerce")
            if pd.isna(recent):
                recent = pd.to_numeric(row.get("minutes"), errors="coerce")
            # Scale expected minutes: starters ~ 85, others lower
            exp_mins = 85.0 * start_prob
            if pd.notna(recent):
                exp_mins = max(20.0, min(90.0, 0.5 * exp_mins + 0.5 * min(90.0, recent)))
            return float(exp_mins)
        except Exception:
            return 65.0

    @staticmethod
    def _goal_points(position: str) -> int:
        return 6 if position == "GKP" or position == "DEF" else (5 if position == "MID" else 4)

    @staticmethod
    def _clean_sheet_points(position: str) -> int:
        return 4 if position in ("GKP", "DEF") else (1 if position == "MID" else 0)

    @staticmethod
    def _apply_xg_xa_shrinkage(df: pd.DataFrame) -> pd.DataFrame:
        """Shrink xG/90 and xA/90 by minutes to avoid small-sample explosions and clamp by position."""
        out = df.copy()
        # Minutes from Understat if available; else use FPL minutes
        minutes = pd.to_numeric(out.get("time"), errors="coerce")
        if minutes is None or minutes.isna().all():
            minutes = pd.to_numeric(out.get("minutes"), errors="coerce")
        minutes = minutes.fillna(0)
        # Priors by position (tune as backtests evolve)
        priors = {
            "FWD": {"xg": 0.35, "xa": 0.12, "xg_cap": 0.9, "xa_cap": 0.5},
            "MID": {"xg": 0.18, "xa": 0.18, "xg_cap": 0.7, "xa_cap": 0.6},
            "DEF": {"xg": 0.03, "xa": 0.08, "xg_cap": 0.25, "xa_cap": 0.3},
            "GKP": {"xg": 0.005, "xa": 0.01, "xg_cap": 0.05, "xa_cap": 0.06},
        }
        xg = pd.to_numeric(out.get("xg_per90"), errors="coerce").fillna(np.nan)
        xa = pd.to_numeric(out.get("xa_per90"), errors="coerce").fillna(np.nan)
        pos = out.get("position").fillna("")
        # Weight by minutes (up to 900 mins ~ 10 matches)
        w = (minutes / 900.0).clip(lower=0.0, upper=1.0)
        # Apply shrinkage and clamps row-wise
        xg_final = []
        xa_final = []
        for i in range(len(out)):
            p = priors.get(str(pos.iloc[i]), priors["MID"])
            xg_i = xg.iloc[i]
            xa_i = xa.iloc[i]
            wi = w.iloc[i]
            xg_blend = (wi * (0 if np.isnan(xg_i) else xg_i)) + ((1 - wi) * p["xg"]) if not np.isnan(xg_i) else p["xg"] * (1 - wi)
            xa_blend = (wi * (0 if np.isnan(xa_i) else xa_i)) + ((1 - wi) * p["xa"]) if not np.isnan(xa_i) else p["xa"] * (1 - wi)
            xg_final.append(float(max(0.0, min(p["xg_cap"], xg_blend))))
            xa_final.append(float(max(0.0, min(p["xa_cap"], xa_blend))))
        out["xg_per90"] = xg_final
        out["xa_per90"] = xa_final
        return out

    def _component_expected_points(self, df: pd.DataFrame) -> pd.Series:
        """Enhanced expected points calculation with better stat integration"""
        opp_map, strength = self._team_opp_map()

        def row_ep(r) -> float:
            # Enhanced availability calculation
            try:
                raw = pd.to_numeric(r.get("chance_next"), errors="coerce")
                fpl_status = str(r.get("fpl_status", "a"))
                
                if fpl_status in ["i", "s", "u"]:  # injured, suspended, unavailable
                    start_prob = 0.0 if pd.isna(raw) else float(raw) / 100.0
                elif fpl_status == "d":  # doubtful
                    start_prob = 0.5 if pd.isna(raw) else float(raw) / 100.0
                else:
                    start_prob = 0.9 if pd.isna(raw) else float(raw) / 100.0
            except Exception:
                start_prob = 0.9
            
            # Use xMins model
            expected_minutes = self._predict_minutes(r)
            
            # Enhanced xG/xA processing with fallback to FPL stats
            xv = pd.to_numeric(r.get("xg_per90"), errors="coerce")
            av = pd.to_numeric(r.get("xa_per90"), errors="coerce")
            
            # Use recent form if available
            recent_xg = pd.to_numeric(r.get("xg_form_5", xv), errors="coerce")
            recent_xa = pd.to_numeric(r.get("xa_form_5", av), errors="coerce")
            
            # Debug: check what xG/xA data we have  
            # if r.get("name") in ["Evanilson", "Salah", "Haaland"]:  # Debug sample players
            #     print(f"DEBUG {r.get('name', 'Unknown')}: start_prob={start_prob:.3f}, expected_minutes={expected_minutes:.1f}")
            #     print(f"  xv={xv}, recent_xg={recent_xg}, av={av}, recent_xa={recent_xa}")
            
            # If xG/xA data unavailable, use price-based estimates (pre-season situation)
            if pd.isna(recent_xg) and pd.isna(xv):
                position = str(r.get("position", ""))
                price = pd.to_numeric(r.get("price", 5.0), errors="coerce") or 5.0
                
                # Enhanced price-based xG estimation with recent FPL form
                last_season_form = pd.to_numeric(r.get("form", 0), errors="coerce") or 0
                form_multiplier = max(0.5, min(1.5, 1.0 + (last_season_form - 3.0) / 10.0))
                
                if position == "FWD":
                    xg90 = max(0.2, min(0.7, (price - 6.0) * 0.12)) * form_multiplier
                elif position == "MID":
                    xg90 = max(0.05, min(0.3, (price - 5.0) * 0.06)) * form_multiplier  
                elif position == "DEF":
                    xg90 = max(0.01, min(0.08, (price - 4.0) * 0.015)) * form_multiplier
                else:  # GKP
                    xg90 = 0.005
            else:
                xg90 = float(recent_xg or xv or 0.0)
            
            if pd.isna(recent_xa) and pd.isna(av):
                position = str(r.get("position", ""))
                price = pd.to_numeric(r.get("price", 5.0), errors="coerce") or 5.0
                
                # Enhanced price-based xA estimation with form
                last_season_form = pd.to_numeric(r.get("form", 0), errors="coerce") or 0
                form_multiplier = max(0.5, min(1.5, 1.0 + (last_season_form - 3.0) / 10.0))
                
                if position == "MID":
                    xa90 = max(0.10, min(0.4, (price - 5.0) * 0.08)) * form_multiplier
                elif position == "FWD":
                    xa90 = max(0.05, min(0.2, (price - 6.0) * 0.05)) * form_multiplier
                elif position == "DEF":
                    xa90 = max(0.02, min(0.1, (price - 4.0) * 0.02)) * form_multiplier
                else:  # GKP
                    xa90 = 0.01
            else:
                xa90 = float(recent_xa or av or 0.0)
            
            # Apply fixture difficulty adjustment
            fixture_mult = float(r.get("fixture_strength", 1.0))
            xg90 *= fixture_mult
            xa90 *= fixture_mult
            
            exp_goals = xg90 * (expected_minutes / 90.0)
            exp_assists = xa90 * (expected_minutes / 90.0)
            
            pos = str(r.get("position") or "")
            goal_pts = exp_goals * self._goal_points(pos)
            assist_pts = exp_assists * 3.0
            
            # Enhanced clean sheet calculation
            team_id = int(r.get("team_id")) if pd.notna(r.get("team_id")) else None
            cs_prob = 0.25
            
            if team_id in opp_map and team_id in strength:
                opp_id, is_home = opp_map[team_id]
                our = strength.get(team_id, {})
                opp = strength.get(opp_id, {})
                our_def = our.get("def_h" if is_home else "def_a")
                opp_att = opp.get("att_a" if is_home else "att_h")
                cs_prob = self._clean_sheet_probability(our_def, opp_att)
                
                # Home advantage for clean sheets
                if is_home:
                    cs_prob *= 1.1
            
            cs_pts = cs_prob * self._clean_sheet_points(pos)
            
            # Enhanced appearance points with form consideration
            appear = self._appearance_points(start_prob)
            
            # More sophisticated bonus calculation
            attacking_threat = exp_goals + exp_assists
            bonus_base = min(2.0, 0.8 * attacking_threat)
            
            # Add consistency bonus (ensure we have a valid value)
            consistency = pd.to_numeric(r.get("consistency"), errors="coerce")
            if pd.isna(consistency):
                consistency = 1.0
            bonus = bonus_base * float(consistency)
            
            # Add momentum adjustment  
            momentum = pd.to_numeric(r.get("momentum"), errors="coerce")
            if pd.isna(momentum):
                momentum = 0.0
            momentum_adj = max(-1.0, min(1.0, float(momentum) * 0.5))
            
            total_ep = appear + goal_pts + assist_pts + cs_pts + bonus + momentum_adj
            
            # Apply price-value adjustment (budget players get slight boost)
            price = float(r.get("price", 10.0))
            if price <= 5.0:
                total_ep *= 1.05  # 5% boost for budget picks
            
            # Debug output for key players
            # if r.get("name") in ["Evanilson", "Salah", "Haaland"]:
            #     print(f"  exp_goals={exp_goals:.3f}, exp_assists={exp_assists:.3f}")
            #     print(f"  appear={appear:.3f}, goal_pts={goal_pts:.3f}, assist_pts={assist_pts:.3f}")
            #     print(f"  cs_pts={cs_pts:.3f}, bonus={bonus:.3f}, total_ep={total_ep:.3f}")
            
            return max(0.0, float(total_ep))

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
        """Enhanced prediction with better model blending and validation"""
        df = self.prepare_training_data()
        if df.empty:
            return df
        
        # Enhanced ML prediction with error handling
        ml_pred: pd.Series | None = None
        ml_confidence = 0.5  # Default confidence
        
        try:
            num = df.select_dtypes(include=[float, int])
            # Remove any remaining problematic columns
            num = num.replace([np.inf, -np.inf], np.nan).fillna(0)
            ml_pred = self.xgb.predict(num)
            
            # Calculate prediction confidence based on feature completeness
            feature_completeness = (1 - num.isna().mean()).mean()
            ml_confidence = min(0.8, max(0.3, feature_completeness))
            
        except Exception as e:
            print(f"ML prediction failed: {e}")
            ml_pred = None

        # Enhanced component-based EP
        ep_component = self._component_expected_points(df)
        ep_component = pd.to_numeric(ep_component, errors="coerce").fillna(0.0)

        # Intelligent blending based on data quality and model confidence
        if ml_pred is not None and len(ml_pred) == len(df):
            ml_pred_clean = pd.to_numeric(ml_pred, errors="coerce").fillna(0.0)
            
            # Adaptive blending - use more ML when we have good data
            alpha = ml_confidence  # Use ML confidence as blending weight
            blended = (alpha * ml_pred_clean.values) + ((1 - alpha) * ep_component.values)
            
            # Apply sanity checks
            blended = np.where(blended < 0, ep_component.values, blended)
            blended = np.where(blended > 20, np.minimum(20, ep_component.values * 1.5), blended)
        else:
            blended = ep_component.values
            
        # Clean NaN/inf values (np already imported at top)
        blended = np.nan_to_num(blended, nan=0.0, posinf=15.0, neginf=0.0)

        # Enhanced result compilation with team names and fixture info
        base_cols = ["player_id", "name", "position", "team_id", "price", "team_short", 
                    "fpl_status", "chance_next", "xg_per90", "xa_per90", "element_type", "now_cost"]
        keep_cols = [c for c in base_cols if c in df.columns]
        results = df[keep_cols].copy()
        
        # Add team names from FPL API
        try:
            teams_data = pd.DataFrame(self.fpl.bootstrap_static().get("teams", []))
            if not teams_data.empty:
                team_map = teams_data.set_index('id')['name'].to_dict()
                results['team_name'] = results['team_id'].map(team_map).fillna('Unknown Team')
            else:
                results['team_name'] = 'Unknown Team'
        except:
            results['team_name'] = 'Unknown Team'
        
        # Add opponent and next 5 fixture difficulty
        try:
            fixtures_df = pd.DataFrame(self.fpl.fixtures())
            events = self.fpl.bootstrap_static().get("events", [])
            current_gw = next((e["id"] for e in events if e.get("is_current", False)), None)
            next_gw = next((e["id"] for e in events if e.get("is_next", False)), None)
            start_gw = next_gw or current_gw or 1
            end_gw = start_gw + 4  # next 5 gameweeks inclusive of start

            # Precompute mapping team_id -> ordered list of next fixtures with difficulty
            team_fixtures: dict[int, list[dict]] = {}
            if not fixtures_df.empty:
                fixtures_slice = fixtures_df[(fixtures_df.get("event") >= start_gw) & (fixtures_df.get("event") <= end_gw)]
                for _, fx in fixtures_slice.sort_values(["event", "kickoff_time"]).iterrows():
                    th = int(fx.get("team_h")); ta = int(fx.get("team_a"))
                    ev = int(fx.get("event")) if not pd.isna(fx.get("event")) else None
                    dh = int(fx.get("team_h_difficulty")) if not pd.isna(fx.get("team_h_difficulty")) else None
                    da = int(fx.get("team_a_difficulty")) if not pd.isna(fx.get("team_a_difficulty")) else None
                    # Home perspective
                    team_fixtures.setdefault(th, []).append({
                        "event": ev, "opponent": ta, "is_home": True, "fdr": dh
                    })
                    # Away perspective
                    team_fixtures.setdefault(ta, []).append({
                        "event": ev, "opponent": th, "is_home": False, "fdr": da
                    })

            # Map team_id -> next opponent/fdr and next5 difficulty metrics (also keep next5 opponent shorts)
            teams_meta = pd.DataFrame(self.fpl.bootstrap_static().get("teams", []))
            short_map = teams_meta.set_index('id')['short_name'].to_dict() if not teams_meta.empty else {}
            name_map = teams_meta.set_index('id')['name'].to_dict() if not teams_meta.empty else {}

            next_opps = []
            next_is_home = []
            next_fdrs = []
            next_opp_shorts = []
            next5_lists = []
            next5_opp_shorts = []
            next5_avgs = []
            for _, row in results.iterrows():
                tid = int(row.get("team_id")) if pd.notna(row.get("team_id")) else None
                arr = team_fixtures.get(tid, []) if tid is not None else []
                # Next fixture
                if arr:
                    nf = arr[0]
                    opp_id = nf.get("opponent")
                    next_opps.append(name_map.get(opp_id, short_map.get(opp_id, opp_id)))
                    next_opp_shorts.append(short_map.get(opp_id, None))
                    next_is_home.append('H' if nf.get("is_home") else 'A')
                    next_fdrs.append(int(nf.get("fdr")) if nf.get("fdr") is not None else None)
                else:
                    next_opps.append(None); next_opp_shorts.append(None); next_is_home.append(None); next_fdrs.append(None)
                # Next 5 FDR list and average
                fdr_list = [int(x.get("fdr")) for x in arr if x.get("fdr") is not None][:5]
                opp_short_list = [short_map.get(x.get("opponent")) for x in arr][:5]
                next5_lists.append(fdr_list)
                next5_opp_shorts.append(opp_short_list)
                next5_avgs.append(float(np.mean(fdr_list)) if fdr_list else None)

            results["next_opponent"] = next_opps
            results["next_opponent_short"] = next_opp_shorts
            results["next_is_home"] = next_is_home
            results["next_fdr"] = next_fdrs
            results["next5_fdr_list"] = next5_lists
            results["next5_opp_shorts"] = next5_opp_shorts
            results["next5_fdr_avg"] = next5_avgs
        except Exception as e:
            # If fixtures not available, leave fields empty
            results["next_opponent"] = None
            results["next_opponent_short"] = None
            results["next_is_home"] = None
            results["next_fdr"] = None
            results["next5_fdr_list"] = None
            results["next5_fdr_avg"] = None

        # Ensure price is properly converted
        if 'now_cost' in results.columns:
            results['now_cost'] = pd.to_numeric(results['now_cost'], errors='coerce') / 10.0
        elif 'price' not in results.columns:
            results['now_cost'] = 0.0
        
        # Add additional useful columns for UI
        extra_cols = ["selected_by_percent", "form", "points_per_game", "value_form", "value_season", "total_points"]
        for col in extra_cols:
            if col in df.columns:
                results[col] = df[col]
        
        # Ensure optional columns exist
        for opt in ["xg_per90", "xa_per90"]:
            if opt not in results.columns:
                results[opt] = pd.NA
                
        # Add prediction components
        results["ep_component"] = ep_component.round(2)
        
        if ml_pred is not None and len(ml_pred) == len(df):
            results["ep_ml"] = pd.Series(ml_pred).round(2).values
            results["ml_confidence"] = round(ml_confidence, 2)
        else:
            results["ep_ml"] = pd.NA
            results["ml_confidence"] = 0.0
            
        results["predicted_points"] = pd.Series(blended).round(2).values
        
        # Add value metrics
        results["value_score"] = (results["predicted_points"] / results["price"]).round(3)
        results["points_per_million"] = (results["predicted_points"] / results["price"] * 10).round(2)
        
        # Multi-GW horizon: compute simple next-3 EP using next5 FDR average as proxy multiplier
        if "next5_fdr_avg" in results.columns:
            fdr = pd.to_numeric(results["next5_fdr_avg"], errors="coerce")
            # Translate FDR avg (1–5) to an approximate multiplier (2=1.1, 3=1.0, 4=0.9)
            horizon_mult = 1.0 + (2.5 - fdr.fillna(3.0)) * 0.05
            results["predicted_points_next3"] = (results["predicted_points"] * horizon_mult).round(2)
        else:
            results["predicted_points_next3"] = results["predicted_points"]

        # Sort by predicted points and apply final filters
        preds = results.sort_values("predicted_points", ascending=False)
        
        # Keep all players for complete analysis (don't filter injured players)
        # Users can filter by availability in the UI if needed
        
        preds.to_csv("data/processed/predictions_current.csv", index=False)
        return preds

