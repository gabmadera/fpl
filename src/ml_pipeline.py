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
from .twitter_news import TwitterNewsClient
from .enhanced_prediction_model import EnhancedPredictionModel


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

        # Enhanced prediction model
        self.enhanced_model = EnhancedPredictionModel()
        self.understat = UnderstatScraper()
        self.alt_sources = AlternativeDataSources()
        # Optional Twitter news client (may be disabled if no token)
        try:
            self.twitter = TwitterNewsClient()
        except Exception:
            self.twitter = None

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
                    numeric_stats = [col for col in available_stats if col not in ["name", "other_name"]]
                    if numeric_stats:
                        # Convert to numeric first and filter out any remaining non-numeric columns
                        valid_numeric_stats = []
                        for col in numeric_stats:
                            if col in joined.columns:
                                joined[col] = pd.to_numeric(joined[col], errors='coerce')
                                # Only include if it successfully converts to numeric
                                if joined[col].dtype.kind in 'biufc':  # numeric types
                                    valid_numeric_stats.append(col)
                        
                        if valid_numeric_stats:
                            agg_dict = {col: 'mean' for col in valid_numeric_stats}
                            agg = joined.groupby("player_id").agg(agg_dict).reset_index()
                        else:
                            agg = pd.DataFrame()
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
            
            # Note: shrinkage will be applied unconditionally after this try/except
                        
        except Exception as e:
            print(f"Understat integration failed: {e}")

        # Always apply minutes-aware shrinkage and positional clamps on xG/90 and xA/90,
        # even if Understat integration failed and we fell back to alternative sources.
        df = self._apply_xg_xa_shrinkage(df)

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
        
        # Try to enhance fixture multipliers using free odds totals (if keys provided)
        try:
            odds_mult = self._fetch_odds_multipliers()
            if odds_mult:
                engineered["odds_fixture_mult"] = engineered["team_id"].map(odds_mult)
                # Where we have odds, prefer them; else keep existing fixture_strength
                engineered["fixture_strength"] = engineered["odds_fixture_mult"].fillna(engineered.get("fixture_strength", 1.0))
        except Exception as _:
            pass

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

    def _fetch_odds_multipliers(self) -> dict[int, float]:
        """Fetch free odds totals and convert to simple multipliers per team.
        Uses The Odds API or API-Football if available. Returns team_id -> multiplier.
        """
        try:
            raw = self.alt_sources.get_free_odds_team_totals()
            if not isinstance(raw, dict) or raw.get("status") != "ok":
                return {}
            # Build FPL team mapping
            teams = pd.DataFrame(self.fpl.bootstrap_static().get("teams", []))
            if teams.empty:
                return {}
            def norm(s: str) -> str:
                return str(s or "").lower().replace(" fc", "").replace(".", "").strip()
            id_to_name = {int(r.id): r.name for _, r in teams.iterrows()}
            id_to_short = {int(r.id): r.short_name for _, r in teams.iterrows()}
            name_to_id = {norm(v): k for k, v in id_to_name.items()}
            short_to_id = {norm(v): k for k, v in id_to_short.items()}

            def team_id_from(name: str) -> int | None:
                n = norm(name)
                if n in name_to_id:
                    return name_to_id[n]
                if n in short_to_id:
                    return short_to_id[n]
                return None

            totals = {}
            if raw.get("source") == "oddsapi":
                # raw['raw'] is a list of events
                for ev in raw.get("raw", []):
                    home = ev.get("home_team"); away = ev.get("away_team")
                    markets = ev.get("bookmakers", []) or ev.get("markets", [])
                    total_line = None
                    # Different books list: support both top-level markets or nested bookmakers
                    for m in ev.get("markets", []) or []:
                        if m.get("key") == "totals" and m.get("outcomes"):
                            total_line = m["outcomes"][0].get("point")
                            break
                    if total_line is None:
                        for bk in markets:
                            for m in bk.get("markets", []) or []:
                                if m.get("key") == "totals" and m.get("outcomes"):
                                    total_line = m["outcomes"][0].get("point"); break
                            if total_line is not None: break
                    if total_line is None:
                        continue
                    try:
                        total_line = float(total_line)
                    except Exception:
                        continue
                    mult = max(0.85, min(1.15, 1.0 + (total_line - 2.5) * 0.07))
                    hid = team_id_from(home); aid = team_id_from(away)
                    if hid is not None: totals[hid] = mult
                    if aid is not None: totals[aid] = mult
            elif raw.get("source") == "api-football":
                # Schema varies; fall back to empty until mapped properly
                pass
            return totals
        except Exception:
            return {}

    def _team_opp_map(self) -> Tuple[dict, dict]:
        """Build mapping of team_id -> (opp_id, is_home) for next active GW."""
        try:
            # Use next active gameweek for fresh fixture data
            target_gw = self.fpl.next_active_gameweek()
            fixtures = pd.DataFrame(self.fpl.fixtures())
            cur = fixtures[fixtures.get("event") == target_gw]
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
        """Improved CS probability from relative strengths. Lower opp_att and higher our_def -> higher CS.
        Uses realistic FPL clean sheet rates based on team strength differential.
        Returns 0..1.
        """
        try:
            if our_def is None or opp_att is None:
                return 0.25  # League average CS rate ~25%

            # Normalize strengths around league average (~100)
            our_def_norm = float(our_def) / 100.0
            opp_att_norm = float(opp_att) / 100.0

            # Calculate strength differential (positive = better chance of CS)
            strength_diff = our_def_norm - opp_att_norm

            # Map to CS probability using a more realistic curve
            # Based on historical FPL data: top defenses get ~40-50% CS vs weak attacks
            # Average matchups get ~20-30%, strong attacks vs weak defenses get ~10-15%
            import math

            # Use gentler logistic curve with coefficient of 1.5 instead of 3.0
            # This creates more realistic probability distribution
            prob = 0.25 + 0.25 * math.tanh(1.5 * strength_diff)

            # Clamp to realistic FPL clean sheet range
            return float(max(0.10, min(0.60, prob)))
        except Exception:
            return 0.25

    @staticmethod
    def _appearance_points(start_prob: float) -> float:
        # 2 points if >=60; approximate with start probability
        return max(0.0, min(1.0, start_prob)) * 2.0

    def _predict_minutes(self, row: pd.Series) -> float:
        """Enhanced xMins model with sophisticated rotation risk analysis."""
        try:
            player_id = row.get("id", 0)
            team_id = row.get("team_id", 0)

            # Step 1: Base probability from FPL status and chance
            base_prob = self._get_base_start_probability(row)

            # Step 2: Fixture congestion analysis
            congestion_factor = self._analyze_fixture_congestion(team_id)

            # Step 3: Historical rotation patterns
            rotation_risk = self._calculate_rotation_risk(player_id, team_id, row)

            # Step 4: Recent playing time trend
            minutes_trend = self._analyze_minutes_trend(row)

            # Step 5: Position and role analysis
            position_factor = self._get_position_rotation_factor(row)

            # Step 6: Manager tendencies
            manager_factor = self._get_manager_rotation_tendency(team_id)

            # Step 7: Combine all factors
            final_start_prob = base_prob * congestion_factor * (1 - rotation_risk) * minutes_trend * position_factor * manager_factor

            # Ensure realistic bounds
            final_start_prob = max(0.05, min(0.95, final_start_prob))

            # Convert to expected minutes
            expected_minutes = self._convert_probability_to_minutes(final_start_prob, row)

            return float(expected_minutes)

        except Exception as e:
            self.logger.error(f"Error in enhanced minutes prediction: {e}")
            return 65.0  # Safe fallback

    def _get_base_start_probability(self, row: pd.Series) -> float:
        """Get base start probability from FPL status and chance"""
        chance = pd.to_numeric(row.get("chance_next"), errors="coerce")
        status = str(row.get("fpl_status", "a")).lower()

        # FPL status mapping
        if status in ["i", "s", "u"]:  # Injured, suspended, unavailable
            return 0.05
        elif status == "d":  # Doubtful
            return 0.4 if pd.isna(chance) else float(chance) / 100.0 * 0.8
        else:  # Available
            return 0.88 if pd.isna(chance) else float(chance) / 100.0

    def _analyze_fixture_congestion(self, team_id: int) -> float:
        """Analyze fixture congestion for rotation risk"""
        try:
            # Get team's fixtures in next 7 days
            fixtures = self.fpl.fixtures()
            current_time = pd.Timestamp.now()

            upcoming_fixtures = []
            for fixture in fixtures:
                if fixture.get('team_h') == team_id or fixture.get('team_a') == team_id:
                    kickoff_time = pd.to_datetime(fixture.get('kickoff_time', ''), errors='coerce')
                    if pd.notna(kickoff_time) and kickoff_time > current_time:
                        days_until = (kickoff_time - current_time).days
                        if days_until <= 7:
                            upcoming_fixtures.append(days_until)

            # Rotation factor based on fixture density
            if len(upcoming_fixtures) >= 3:  # 3+ games in 7 days
                return 0.7  # High rotation risk
            elif len(upcoming_fixtures) == 2:  # 2 games in 7 days
                return 0.85  # Moderate rotation risk
            else:
                return 1.0  # No congestion

        except Exception:
            return 1.0  # No adjustment if data unavailable

    def _calculate_rotation_risk(self, player_id: int, team_id: int, row: pd.Series) -> float:
        """Calculate rotation risk based on historical patterns"""
        try:
            # Age factor (older players more likely to be rotated)
            age = row.get('age', 27)
            age_factor = 0.1 if age >= 32 else (0.05 if age >= 29 else 0.0)

            # Price factor (cheap players more rotation risk)
            price = row.get('now_cost', 50) / 10
            price_factor = 0.15 if price < 5.0 else (0.05 if price < 7.0 else 0.0)

            # Minutes consistency (check recent variance)
            recent_minutes = [
                row.get('minutes_form_1', 0),
                row.get('minutes_form_2', 0),
                row.get('minutes_form_3', 0)
            ]
            recent_minutes = [m for m in recent_minutes if m > 0]

            if len(recent_minutes) >= 2:
                minutes_variance = np.var(recent_minutes)
                variance_factor = min(0.2, minutes_variance / 1000)  # High variance = rotation risk
            else:
                variance_factor = 0.0

            # Total rotation risk
            total_risk = age_factor + price_factor + variance_factor
            return min(0.4, total_risk)  # Cap at 40% risk

        except Exception:
            return 0.1  # Default modest risk

    def _analyze_minutes_trend(self, row: pd.Series) -> float:
        """Analyze recent minutes trend"""
        try:
            # Get last 3 gameweeks minutes
            minutes_data = [
                row.get('minutes_form_1', 0),
                row.get('minutes_form_2', 0),
                row.get('minutes_form_3', 0)
            ]

            # Remove zeros and calculate trend
            valid_minutes = [m for m in minutes_data if m > 0]
            if len(valid_minutes) < 2:
                return 1.0  # No trend data

            # Calculate trend (is player getting more or fewer minutes?)
            recent_avg = np.mean(valid_minutes[:2]) if len(valid_minutes) >= 2 else valid_minutes[0]
            older_avg = valid_minutes[-1] if len(valid_minutes) >= 3 else recent_avg

            if recent_avg > older_avg + 10:  # Getting more minutes
                return 1.1
            elif recent_avg < older_avg - 15:  # Getting fewer minutes
                return 0.85
            else:
                return 1.0  # Stable

        except Exception:
            return 1.0

    def _get_position_rotation_factor(self, row: pd.Series) -> float:
        """Get position-specific rotation factors"""
        position_id = row.get('element_type', 3)

        # Position rotation tendencies
        position_factors = {
            1: 1.0,   # GKP - usually fixed
            2: 0.92,  # DEF - some rotation
            3: 0.88,  # MID - more rotation
            4: 0.90   # FWD - moderate rotation
        }

        return position_factors.get(position_id, 0.9)

    def _get_manager_rotation_tendency(self, team_id: int) -> float:
        """Get manager-specific rotation tendencies"""
        # Manager rotation tendencies (based on general knowledge)
        # In real implementation, this would be learned from historical data
        rotation_managers = {
            # Teams known for heavy rotation (lower factor)
            1: 0.85,  # Arsenal (Arteta rotates)
            3: 0.80,  # Brighton (De Zerbi tactical rotation)
            6: 0.88,  # Chelsea (rotation based on form)
            7: 0.75,  # Crystal Palace (Vieira rotation)
            11: 0.85, # Liverpool (Klopp rotation)
            13: 0.80, # Man City (Pep heavy rotation)
            17: 0.85, # Tottenham (Conte/Postecoglou rotation)
        }

        return rotation_managers.get(team_id, 0.95)  # Most teams have modest rotation

    def _convert_probability_to_minutes(self, start_prob: float, row: pd.Series) -> float:
        """Convert start probability to expected minutes"""
        # Base minutes for different probability ranges
        if start_prob >= 0.8:  # Likely starter
            base_minutes = 82
        elif start_prob >= 0.6:  # Probable starter
            base_minutes = 75
        elif start_prob >= 0.4:  # Possible starter
            base_minutes = 60
        elif start_prob >= 0.2:  # Bench option
            base_minutes = 25
        else:  # Unlikely to play
            base_minutes = 10

        # Adjust based on recent playing patterns
        recent = pd.to_numeric(row.get("minutes_form_3"), errors="coerce")
        if pd.notna(recent) and recent > 0:
            # Blend with recent average (60% model, 40% recent)
            final_minutes = 0.6 * base_minutes + 0.4 * recent
        else:
            final_minutes = base_minutes

        # Apply start probability scaling
        expected_minutes = final_minutes * start_prob

        # Ensure realistic bounds
        return max(5.0, min(90.0, expected_minutes))

    def _apply_twitter_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adjust availability/start probability and augment roles using Twitter signals if available."""
        if not self.twitter:
            return df
        try:
            tweets = self.twitter.fetch_news()
            enhanced = self.twitter.extract_enhanced_signals(tweets)
            avail = enhanced.get('availability', {})
            set_pieces = enhanced.get('set_pieces', {})
            pos_hints = enhanced.get('position', {})
            recommend = enhanced.get('recommend', {})
            if not (avail or set_pieces or pos_hints or recommend):
                return df
            out = df.copy()
            # Map by player last name if possible
            out['name_lower'] = out.get('name', '').astype(str).str.lower()
            out['last_name'] = out['name_lower'].str.split().str[-1]
            # Adjust chance_next by small increments based on signals
            if 'chance_next' not in out.columns:
                out['chance_next'] = 90.0
            base = pd.to_numeric(out['chance_next'], errors='coerce').fillna(90.0)
            delta = out['last_name'].map(avail).fillna(0.0) * 100.0  # convert 0.1 -> 10%
            out['chance_next'] = (base + delta).clip(lower=0.0, upper=100.0)

            # Set-piece role nudges: small boosts to xa/xg priors
            if 'xa_per90' in out.columns:
                xa = pd.to_numeric(out['xa_per90'], errors='coerce')
                corners_boost = out['last_name'].map({k: 0.03 for k, v in set_pieces.items() if v.get('corners')}).fillna(0.0)
                fk_boost = out['last_name'].map({k: 0.02 for k, v in set_pieces.items() if v.get('fks')}).fillna(0.0)
                out['xa_per90'] = (xa.fillna(0.0) + corners_boost + fk_boost).clip(upper=0.6)
            if 'xg_per90' in out.columns:
                xg = pd.to_numeric(out['xg_per90'], errors='coerce')
                pens_boost = out['last_name'].map({k: 0.04 for k, v in set_pieces.items() if v.get('pens')}).fillna(0.0)
                out['xg_per90'] = (xg.fillna(0.0) + pens_boost).clip(upper=0.9)

            # Position hints: can nudge position for priors (only if plausible)
            mask = out['last_name'].isin(list(pos_hints.keys()))
            out.loc[mask, 'position_hint'] = out.loc[mask, 'last_name'].map(pos_hints)

            # Recommendations: small EP nudge via a temporary feature used in blending
            out['news_recommendation_boost'] = out['last_name'].map(recommend).fillna(0.0)
            return out.drop(columns=['name_lower', 'last_name'])
        except Exception:
            return df

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
                    # Increased scaling for forwards - they score more than modeled
                    xg90 = max(0.3, min(1.0, (price - 6.0) * 0.18)) * form_multiplier  # Increased from 0.2-0.7 to 0.3-1.0
                elif position == "MID":
                    # Premium midfielders get significant xG
                    xg90 = max(0.08, min(0.5, (price - 5.0) * 0.09)) * form_multiplier  # Increased from 0.05-0.3 to 0.08-0.5
                elif position == "DEF":
                    # Attacking defenders can score
                    xg90 = max(0.02, min(0.12, (price - 4.0) * 0.025)) * form_multiplier  # Increased from 0.01-0.08 to 0.02-0.12
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
                    # Premium midfielders create many assists
                    xa90 = max(0.15, min(0.6, (price - 5.0) * 0.12)) * form_multiplier  # Increased from 0.10-0.4 to 0.15-0.6
                elif position == "FWD":
                    # Modern forwards get assists too
                    xa90 = max(0.08, min(0.3, (price - 6.0) * 0.08)) * form_multiplier  # Increased from 0.05-0.2 to 0.08-0.3
                elif position == "DEF":
                    # Wing-backs and attacking defenders
                    xa90 = max(0.03, min(0.15, (price - 4.0) * 0.03)) * form_multiplier  # Increased from 0.02-0.1 to 0.03-0.15
                else:  # GKP
                    xa90 = 0.01
            else:
                xa90 = float(recent_xa or av or 0.0)
            
            # Apply fixture difficulty adjustment (generic)
            fixture_mult = float(r.get("fixture_strength", 1.0))
            xg90 *= fixture_mult
            xa90 *= fixture_mult

            # Opponent-aware scaling: temper attack by opponent defence for the specific matchup
            # Uses FPL team strengths: lower opp attack/defence values mean stronger team.
            try:
                team_id_local = int(r.get("team_id")) if pd.notna(r.get("team_id")) else None
            except Exception:
                team_id_local = None
            if team_id_local in opp_map and team_id_local in strength:
                opp_id, is_home = opp_map[team_id_local]
                our = strength.get(team_id_local, {})
                opp = strength.get(opp_id, {})
                # Defensive strength scale ~100; lower is stronger defence
                opp_def = opp.get("def_a" if is_home else "def_h")
                our_att = our.get("att_h" if is_home else "att_a")
                try:
                    opp_def = float(opp_def) if opp_def is not None else 100.0
                    our_att = float(our_att) if our_att is not None else 100.0
                    # Normalize around 100. Strong opp defence (<100) reduces our attack; weak defence (>100) boosts it.
                    # Keep the effect modest to avoid whiplash.
                    attack_scale = (our_att / 100.0) * (100.0 / max(1.0, opp_def))
                    # Clamp to a reasonable band (slightly wider to have visible effect)
                    attack_scale = max(0.80, min(1.20, attack_scale))
                    xg90 *= attack_scale
                    xa90 *= attack_scale
                except Exception:
                    pass
            
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
                base_cs = self._clean_sheet_probability(our_def, opp_att)
                # Apply modest home advantage consistent with new CS ranges
                cs_prob = base_cs * (1.08 if is_home else 1.0)
                # Clamp to realistic range (already handled in _clean_sheet_probability, but ensure consistency)
                cs_prob = float(max(0.10, min(0.60, cs_prob)))
            
            cs_pts = cs_prob * self._clean_sheet_points(pos)
            
            # Enhanced appearance points with form consideration
            appear = self._appearance_points(start_prob)
            
            # Enhanced bonus calculation - FPL bonus system is significant
            attacking_threat = exp_goals + exp_assists

            # Bonus points are more frequent than modeled - increase base calculation
            bonus_base = min(3.0, 1.2 * attacking_threat)  # Increased from 0.8 to 1.2, max from 2.0 to 3.0

            # Price-based bonus adjustment - expensive players get bonus more often
            price = float(r.get("price", 5.0))
            price_bonus_mult = 1.0 + max(0.0, (price - 8.0) * 0.1)  # Premium players get 10% more bonus per £1m above 8.0

            # Add consistency bonus (ensure we have a valid value)
            consistency = pd.to_numeric(r.get("consistency"), errors="coerce")
            if pd.isna(consistency):
                consistency = 1.0
            bonus = bonus_base * float(consistency) * price_bonus_mult
            
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
        # Validate fixture data freshness before predictions
        validation = self.fpl.validate_fixture_freshness()
        if validation["status"] == "error":
            print(f"WARNING: Fixture data validation failed: {validation.get('error', 'Unknown error')}")
            for issue in validation.get("issues", []):
                print(f"  - {issue}")
        elif validation["status"] == "warning":
            print(f"WARNING: Fixture data quality issues detected:")
            for issue in validation.get("issues", []):
                print(f"  - {issue}")

        print(f"Using gameweek {validation.get('next_active_gw', 'unknown')} for fixture data ({validation.get('upcoming_fixtures_count', 0)} upcoming fixtures)")

        df = self.prepare_training_data()
        if df.empty:
            return df
        # Apply Twitter news availability adjustments if possible
        df = self._apply_twitter_signals(df)
        
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

        # Intelligent blending based on data quality, model confidence, and matchup risk
        if ml_pred is not None and len(ml_pred) == len(df):
            ml_pred_clean = pd.to_numeric(ml_pred, errors="coerce").fillna(0.0)

            # Compute per-row risk adjustment to downweight ML in risky contexts (rotation, tough matchup)
            opp_map, strength = self._team_opp_map()
            risk = np.ones(len(df), dtype=float)
            for i, r in df.reset_index(drop=True).iterrows():
                try:
                    pos = str(r.get("position", ""))
                    team_id = int(r.get("team_id")) if pd.notna(r.get("team_id")) else None
                    if team_id in opp_map and team_id in strength:
                        opp_id, is_home = opp_map[team_id]
                        our = strength.get(team_id, {})
                        opp = strength.get(opp_id, {})
                        # Opponent attack/defense indices
                        opp_att = float(opp.get("att_a" if is_home else "att_h", 100))
                        opp_def = float(opp.get("def_h" if is_home else "def_a", 100))
                        # Downweight defenders vs strong attacking opponents
                        if pos in ("DEF", "GKP"):
                            if opp_att > 110:
                                risk[i] *= 0.6
                            elif opp_att > 105:
                                risk[i] *= 0.75
                        # Downweight attackers vs strong defenses
                        elif pos in ("FWD", "MID"):
                            if opp_def < 95:
                                risk[i] *= 0.8
                    # Rotation risk: very low recent minutes
                    recent = pd.to_numeric(r.get("minutes_form_3"), errors="coerce")
                    if pd.notna(recent) and recent < 45:
                        risk[i] *= 0.85
                except Exception:
                    continue

            # Bound risk and derive per-row alpha
            risk = np.clip(risk, 0.5, 1.0)
            # Recommendation boosts can slightly increase reliance on component EP (more transparent)
            if 'news_recommendation_boost' in df.columns:
                rec_boost_vals = pd.to_numeric(df['news_recommendation_boost'], errors='coerce').fillna(0.0).values
            else:
                rec_boost_vals = np.zeros(len(df), dtype=float)
            # Premium players are more predictable - give more weight to ML predictions
            player_prices = pd.to_numeric(df.get('price', df.get('now_cost', 5)), errors='coerce').fillna(5)
            premium_boost = np.where(player_prices >= 10.0, 0.1, 0.0)  # +10% ML weight for premium players
            
            alpha_vec = np.clip((ml_confidence * risk) - (0.1 * rec_boost_vals) + premium_boost, 0.25, 0.85)

            blended = (alpha_vec * ml_pred_clean.values) + ((1 - alpha_vec) * ep_component.values)

            # Apply sanity checks (but allow premium players to exceed 20 points)
            blended = np.where(blended < 0, ep_component.values, blended)
            
            # Position-based scoring caps - FPL players can score much more than 20 points
            player_prices = pd.to_numeric(df.get('price', df.get('now_cost', 5)), errors='coerce').fillna(5)
            positions = df.get('position', 'MID').fillna('MID')

            # Realistic FPL scoring caps by position and price
            base_caps = np.where(positions == 'GKP', 15,
                        np.where(positions == 'DEF', 20,
                        np.where(positions == 'MID', 25,
                        np.where(positions == 'FWD', 30, 20))))  # Higher caps for attackers

            # Premium player multiplier
            premium_multiplier = np.where(player_prices >= 10.0, 1.3,
                                np.where(player_prices >= 8.0, 1.2, 1.0))

            max_allowed = base_caps * premium_multiplier

            # Apply realistic caps - good players can have big games
            blended = np.where(blended > max_allowed, np.minimum(max_allowed, ep_component.values * 1.5), blended)
        else:
            blended = ep_component.values
            
        # Clean NaN/inf values (np already imported at top)
        blended = np.nan_to_num(blended, nan=0.0, posinf=30.0, neginf=0.0)

        # Add prediction validation logging for debugging
        self._validate_predictions(blended, df)

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
            # Use next active gameweek for fresh fixture display
            start_gw = self.fpl.next_active_gameweek()
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

    def predict_gameweek(self, use_enhanced_model: bool = True) -> pd.DataFrame:
        """
        Generate gameweek predictions using enhanced model or fallback to basic model
        """
        try:
            if use_enhanced_model:
                # Get current gameweek
                bs = self.fpl.bootstrap_static()
                if not bs:
                    raise Exception("Could not fetch bootstrap data")

                events = bs.get("events", [])
                current_gw = next((e["id"] for e in events if e.get("is_current", False)), 1)

                # Use enhanced prediction model
                predictions_df = self.enhanced_model.generate_enhanced_predictions(current_gw)

                if not predictions_df.empty:
                    # Convert enhanced predictions to expected format
                    predictions_df = self._format_enhanced_predictions(predictions_df)
                    print(f"Enhanced model generated predictions for {len(predictions_df)} players")
                    return predictions_df

            # Fallback to basic model if enhanced model fails or is disabled
            print("Falling back to basic prediction model")
            return self.predict_players()

        except Exception as e:
            print(f"Prediction failed: {e}")
            # Return empty DataFrame as ultimate fallback
            return pd.DataFrame()

    def _validate_predictions(self, predictions: np.ndarray, df: pd.DataFrame) -> None:
        """Validate predictions and log anomalies for debugging"""
        try:
            positions = df.get('position', 'MID').fillna('MID')
            prices = pd.to_numeric(df.get('price', 5), errors='coerce').fillna(5)
            names = df.get('name', 'Unknown').fillna('Unknown')

            # Position-based validation ranges
            for i, (pred, pos, price, name) in enumerate(zip(predictions, positions, prices, names)):
                # Define realistic ranges
                if pos == 'GKP' and (pred < 1 or pred > 20):
                    print(f"Warning: GKP {name} prediction {pred:.1f} outside range (1-20)")
                elif pos == 'DEF' and (pred < 1 or pred > 25):
                    print(f"Warning: DEF {name} prediction {pred:.1f} outside range (1-25)")
                elif pos == 'MID' and (pred < 2 or pred > 30):
                    print(f"Warning: MID {name} prediction {pred:.1f} outside range (2-30)")
                elif pos == 'FWD' and (pred < 2 or pred > 35):
                    print(f"Warning: FWD {name} prediction {pred:.1f} outside range (2-35)")

                # Price vs prediction sanity check
                if price >= 12.0 and pred < 6.0:
                    print(f"Warning: Premium player {name} (£{price}m) has low prediction {pred:.1f}")
                elif price <= 5.0 and pred > 15.0:
                    print(f"Warning: Budget player {name} (£{price}m) has high prediction {pred:.1f}")

            # Team summary stats
            avg_pred = np.mean(predictions)
            print(f"Prediction validation: {len(predictions)} players, avg={avg_pred:.2f}, min={np.min(predictions):.2f}, max={np.max(predictions):.2f}")

        except Exception as e:
            print(f"Validation failed: {e}")

    def predict_players(self) -> pd.DataFrame:
        """Basic player prediction fallback method"""
        try:
            # Use the existing predict_current method as fallback
            return self.predict_current()
        except Exception as e:
            print(f"Basic prediction fallback failed: {e}")
            # Return minimal empty dataframe with required columns
            return pd.DataFrame(columns=['id', 'name', 'position', 'team', 'predicted_points', 'prediction_confidence'])

    def _format_enhanced_predictions(self, enhanced_df: pd.DataFrame) -> pd.DataFrame:
        """
        Format enhanced predictions to match expected interface
        """
        try:
            # Map enhanced prediction columns to expected format
            formatted_df = enhanced_df.copy()

            # Rename columns to match expected interface
            column_mapping = {
                'id': 'player_id',  # TeamSelector expects player_id
                'enhanced_prediction': 'predicted_points',
                'prediction_confidence': 'confidence',
                'availability_score': 'availability',
                'venue_adjusted_difficulty': 'fixture_difficulty',
                'points_per_million': 'value_rating'
            }

            for old_col, new_col in column_mapping.items():
                if old_col in formatted_df.columns:
                    formatted_df[new_col] = formatted_df[old_col]

            # Ensure required columns exist
            required_columns = ['player_id', 'name', 'position', 'team', 'price', 'predicted_points']
            for col in required_columns:
                if col not in formatted_df.columns:
                    if col == 'predicted_points':
                        formatted_df[col] = formatted_df.get('enhanced_prediction', 2.0)
                    else:
                        formatted_df[col] = 'Unknown'

            # Add additional useful columns
            if 'points_per_million' not in formatted_df.columns:
                formatted_df['points_per_million'] = (formatted_df['predicted_points'] / formatted_df['price']).round(2)

            # Add form and team context
            if 'form' in formatted_df.columns:
                formatted_df['recent_form'] = formatted_df['form']

            # Add captaincy potential (simple heuristic)
            formatted_df['captaincy_potential'] = (
                formatted_df['predicted_points'] *
                formatted_df.get('confidence', 0.5) *
                (1 + formatted_df.get('momentum_score', 1.0) * 0.2)
            ).round(2)

            # Sort by predicted points
            formatted_df = formatted_df.sort_values('predicted_points', ascending=False)

            print(f"Enhanced model prediction summary:")
            print(f"- Total players: {len(formatted_df)}")
            print(f"- Top prediction: {formatted_df['predicted_points'].max():.1f} pts")
            print(f"- Average prediction: {formatted_df['predicted_points'].mean():.1f} pts")
            print(f"- Positions covered: {formatted_df['position'].nunique()}")

            return formatted_df

        except Exception as e:
            print(f"Error formatting enhanced predictions: {e}")
            return enhanced_df  # Return original if formatting fails

