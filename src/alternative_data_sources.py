from __future__ import annotations

import requests
import pandas as pd
import time
from typing import Dict, List, Optional
from .fpl_client import FPLClient
from .config import config


class AlternativeDataSources:
    """Alternative data sources for pre-season scenarios when primary sources fail"""
    
    def __init__(self):
        self.fpl = FPLClient()
        self.headers = {"User-Agent": "Mozilla/5.0 (compatible; FPL-AI/1.0)"}
    
    def get_fpl_historical_performance(self) -> pd.DataFrame:
        """Get historical FPL performance data from previous seasons"""
        try:
            # Get current season player data
            bootstrap = self.fpl.bootstrap_static()
            players = pd.DataFrame(bootstrap.get("elements", []))
            
            if players.empty:
                return pd.DataFrame()
            
            # Extract useful historical metrics from FPL API
            useful_cols = [
                "id", "web_name", "element_type", "team", "now_cost",
                "total_points", "points_per_game", "form", "value_form", "value_season",
                "goals_scored", "assists", "clean_sheets", "goals_conceded",
                "saves", "bonus", "bps", "influence", "creativity", "threat", "ict_index"
            ]
            
            available_cols = [col for col in useful_cols if col in players.columns]
            df = players[available_cols].copy()
            
            # Rename to standard format
            df = df.rename(columns={
                "id": "player_id",
                "web_name": "name", 
                "element_type": "position_id",
                "team": "team_id",
                "now_cost": "price"
            })
            
            # Convert price from pence to pounds
            df["price"] = pd.to_numeric(df["price"], errors="coerce") / 10.0
            
            # Map position IDs
            df["position"] = df["position_id"].map({1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"})
            
            # Estimate xG/xA from FPL metrics (creativity/threat are proxies)
            if "creativity" in df.columns and "threat" in df.columns:
                creativity = pd.to_numeric(df["creativity"], errors="coerce").fillna(0)
                threat = pd.to_numeric(df["threat"], errors="coerce").fillna(0)
                
                # Normalize ICT stats (typically 0-200+ range) to reasonable xG/xA values
                df["estimated_xa_per90"] = (creativity / 100.0).clip(upper=0.8)  # Max ~0.8 xA/90
                df["estimated_xg_per90"] = (threat / 100.0).clip(upper=1.2)     # Max ~1.2 xG/90
                
                # Position adjustments
                position_xa_mult = {"GKP": 0.1, "DEF": 0.4, "MID": 1.0, "FWD": 0.6}
                position_xg_mult = {"GKP": 0.1, "DEF": 0.3, "MID": 0.7, "FWD": 1.0}
                
                df["estimated_xa_per90"] *= df["position"].map(position_xa_mult).fillna(1.0)
                df["estimated_xg_per90"] *= df["position"].map(position_xg_mult).fillna(1.0)
            
            return df
            
        except Exception as e:
            print(f"FPL historical data failed: {e}")
            return pd.DataFrame()
    
    def get_price_based_estimates(self, players_df: pd.DataFrame) -> pd.DataFrame:
        """Generate intelligent price-based estimates using team strength and player pricing"""
        df = players_df.copy()
        
        try:
            # Get team strength data from FPL API
            bootstrap = self.fpl.bootstrap_static()
            teams = pd.DataFrame(bootstrap.get("teams", []))
            
            team_strength = {}
            if not teams.empty:
                for _, team in teams.iterrows():
                    team_id = team["id"]
                    # Average attack and defense strengths
                    att_strength = (team.get("strength_attack_home", 100) + team.get("strength_attack_away", 100)) / 2
                    def_strength = (team.get("strength_defence_home", 100) + team.get("strength_defence_away", 100)) / 2
                    team_strength[team_id] = {"attack": att_strength, "defense": def_strength}
            
            # Enhanced price-based modeling
            for idx, player in df.iterrows():
                position = player.get("position", "MID")
                price = pd.to_numeric(player.get("price", 5.0), errors="coerce") or 5.0
                team_id = player.get("team_id")
                
                # Base estimates by position and price tier
                if position == "FWD":
                    base_xg = 0.2 + (price - 6.0) * 0.08  # £6m+ forwards get scaling xG
                    base_xa = 0.05 + (price - 6.0) * 0.03
                elif position == "MID":
                    base_xg = 0.05 + (price - 5.0) * 0.04
                    base_xa = 0.1 + (price - 5.0) * 0.06  # Midfielders are assist-heavy
                elif position == "DEF":
                    base_xg = 0.01 + max(0, price - 4.5) * 0.015
                    base_xa = 0.02 + max(0, price - 4.5) * 0.02
                else:  # GKP
                    base_xg = 0.005
                    base_xa = 0.01
                
                # Team strength multiplier
                team_mult = 1.0
                if team_id in team_strength:
                    attack_strength = team_strength[team_id]["attack"]
                    team_mult = attack_strength / 100.0  # Normalize around 1.0
                
                # Apply team strength
                final_xg = base_xg * team_mult
                final_xa = base_xa * team_mult
                
                # Reasonable bounds
                final_xg = max(0.001, min(1.5, final_xg))
                final_xa = max(0.001, min(1.0, final_xa))
                
                df.loc[idx, "xg_per90"] = round(final_xg, 3)
                df.loc[idx, "xa_per90"] = round(final_xa, 3)
            
            return df
            
        except Exception as e:
            print(f"Price-based estimates failed: {e}")
            return df
    
    def get_combined_fallback_data(self, base_df: pd.DataFrame) -> pd.DataFrame:
        """Combine all fallback data sources for robust pre-season predictions"""
        df = base_df.copy()
        
        # Try FPL historical first
        fpl_data = self.get_fpl_historical_performance()
        if not fpl_data.empty:
            # Merge FPL creativity/threat estimates
            merge_cols = ["player_id", "estimated_xg_per90", "estimated_xa_per90"]
            available_merge_cols = [col for col in merge_cols if col in fpl_data.columns]
            if len(available_merge_cols) > 1:
                df = df.merge(fpl_data[available_merge_cols], on="player_id", how="left")
                
                # Use FPL estimates where external data is missing
                if "estimated_xg_per90" in df.columns:
                    df["xg_per90"] = df["xg_per90"].fillna(df["estimated_xg_per90"])
                if "estimated_xa_per90" in df.columns:
                    df["xa_per90"] = df["xa_per90"].fillna(df["estimated_xa_per90"])
        
        # Price-based fallback for remaining missing values
        df = self.get_price_based_estimates(df)
        
        # Final validation - ensure no player has 0 threat
        for idx, player in df.iterrows():
            if pd.isna(player.get("xg_per90")) or player.get("xg_per90", 0) <= 0:
                position = player.get("position", "MID")
                price = float(player.get("price", 5.0))
                
                # Minimal baseline by position
                if position == "FWD":
                    df.loc[idx, "xg_per90"] = max(0.1, price * 0.05)
                elif position == "MID":
                    df.loc[idx, "xg_per90"] = max(0.05, price * 0.02)
                elif position == "DEF":
                    df.loc[idx, "xg_per90"] = max(0.01, price * 0.005)
                else:  # GKP
                    df.loc[idx, "xg_per90"] = 0.005
            
            if pd.isna(player.get("xa_per90")) or player.get("xa_per90", 0) <= 0:
                position = player.get("position", "MID")  
                price = float(player.get("price", 5.0))
                
                # Minimal baseline by position
                if position == "MID":
                    df.loc[idx, "xa_per90"] = max(0.08, price * 0.04)
                elif position == "FWD":
                    df.loc[idx, "xa_per90"] = max(0.04, price * 0.02)
                elif position == "DEF":
                    df.loc[idx, "xa_per90"] = max(0.02, price * 0.01)
                else:  # GKP
                    df.loc[idx, "xa_per90"] = 0.01
        
        print(f"Alternative sources: Enhanced {len(df)} players with fallback xG/xA data")
        return df

    def get_free_odds_team_totals(self) -> Dict:
        """Fetch basic odds snapshot from free APIs; returns raw JSON for now."""
        # The Odds API (requires ODDSAPI_KEY)
        if config.ODDSAPI_KEY:
            try:
                r = requests.get(
                    "https://api.the-odds-api.com/v4/sports/soccer_epl/odds",
                    params={"regions": "uk,eu", "markets": "h2h,totals", "apiKey": config.ODDSAPI_KEY, "oddsFormat": "decimal"},
                    timeout=20,
                )
                if r.ok:
                    return {"status": "ok", "source": "oddsapi", "raw": r.json()}
            except Exception:
                pass
        # API-Football via RapidAPI (requires API_FOOTBALL_KEY)
        if config.API_FOOTBALL_KEY:
            try:
                headers = {"x-rapidapi-host": "api-football-v1.p.rapidapi.com", "x-rapidapi-key": config.API_FOOTBALL_KEY}
                r = requests.get("https://api-football-v1.p.rapidapi.com/v3/odds", headers=headers, params={"league": 39, "season": 2025}, timeout=20)
                if r.ok:
                    return {"status": "ok", "source": "api-football", "raw": r.json()}
            except Exception:
                pass
        return {"status": "empty"}