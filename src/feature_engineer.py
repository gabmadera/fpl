from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional
from .position_specific_features import PositionSpecificFeatureEngineer
from .enhanced_fixture_analyzer import EnhancedFixtureAnalyzer


class FeatureEngineer:
    def __init__(self) -> None:
        self.lookback_windows = [3, 5, 8, 12, 20]
        self.form_windows = [3, 5, 10]
        self.difficulty_weights = {'easy': 1.2, 'medium': 1.0, 'hard': 0.8}
        self.position_engineer = PositionSpecificFeatureEngineer()
        self.enhanced_fixture_analyzer = EnhancedFixtureAnalyzer()

    def create_features(self, player_data: pd.DataFrame, fixture_data: pd.DataFrame) -> pd.DataFrame:
        df = player_data.copy()
        if "predicted_points" not in df.columns:
            df["predicted_points"] = 0.0
        
        # Enhanced price processing
        df = self._process_price_features(df)
        
        # Advanced form features
        df = self._create_form_features(df)
        
        # Enhanced position-specific features
        df = self._create_position_features(df)
        df = self.position_engineer.engineer_position_features(df)
        
        # Enhanced fixture analysis
        if not fixture_data.empty and "team_id" in df.columns:
            df = self._create_fixture_features(df, fixture_data)
            df = self._create_enhanced_fixture_features(df)
        else:
            df["opp_difficulty"] = 3
            df["fixture_strength"] = 1.0
        
        # Value features
        df = self._create_value_features(df)
        
        # Momentum and consistency features
        df = self._create_momentum_features(df)
        
        return df

    def _process_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced price processing with better error handling"""
        if "price" not in df.columns:
            if "value" in df.columns:
                df["price"] = pd.to_numeric(df["value"], errors="coerce") / 10.0
            else:
                df["price"] = 10.0
        
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        if df["price"].isna().all():
            df["price"] = 10.0
        else:
            median_price = df["price"].median()
            df["price"] = df["price"].fillna(median_price)

        # Enhanced price-based features
        try:
            uniques = max(1, int(df["price"].nunique()))
            df["price_band"] = pd.qcut(df["price"], q=min(5, uniques), labels=False, duplicates="drop")
        except Exception:
            df["price_band"] = 0
        
        df["is_premium"] = df["price"] >= df["price"].quantile(0.85)
        df["is_budget"] = df["price"] <= df["price"].quantile(0.15)
        df["price_tier"] = pd.cut(df["price"], bins=5, labels=False)
        
        return df
    
    def _create_form_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling form features"""
        form_cols = ['total_points', 'minutes', 'goals_scored', 'assists', 'clean_sheets', 'bps']
        
        for col in form_cols:
            if col in df.columns:
                for window in self.form_windows:
                    df[f"{col}_form_{window}"] = df.groupby('player_id')[col].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
                    df[f"{col}_trend_{window}"] = df.groupby('player_id')[col].rolling(window, min_periods=1).apply(self._calculate_trend).reset_index(level=0, drop=True)
        
        # Points per million value
        if 'total_points' in df.columns:
            df['points_per_million'] = df['total_points'] / df['price']
        
        return df
    
    def _create_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced position-specific features"""
        # Position encoding
        position_map = {'GKP': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}
        df['position_encoded'] = df['position'].map(position_map).fillna(0)
        
        # Position-specific probabilities
        df["dc_point_probability"] = np.where(df["position"] == "DEF", 0.25, 
                                             np.where(df["position"] == "GKP", 0.20, 0.05))
        df["goal_probability"] = np.where(df["position"] == "FWD", 0.4, 
                                         np.where(df["position"] == "MID", 0.2, 0.1))
        df["assist_probability"] = np.where(df["position"] == "MID", 0.3, 
                                           np.where(df["position"] == "FWD", 0.2, 0.1))
        
        # Expected returns by position
        if 'xg_per90' in df.columns and 'xa_per90' in df.columns:
            goal_points = {'GKP': 6, 'DEF': 6, 'MID': 5, 'FWD': 4}
            df['expected_goal_points'] = df['xg_per90'] * df['position'].map(goal_points).fillna(4)
            df['expected_assist_points'] = df['xa_per90'] * 3
            df['expected_total'] = df['expected_goal_points'] + df['expected_assist_points']
        
        return df
    
    def _create_fixture_features(self, df: pd.DataFrame, fixture_data: pd.DataFrame) -> pd.DataFrame:
        """Enhanced fixture difficulty analysis"""
        opp_difficulty = self._compute_next_fixture_difficulty(fixture_data)
        df = df.merge(opp_difficulty, on="team_id", how="left")
        
        median_diff = df["opp_difficulty"].median() if not df["opp_difficulty"].isna().all() else 3
        df["opp_difficulty"] = df["opp_difficulty"].fillna(median_diff)
        
        # Fixture strength multiplier
        df['fixture_strength'] = np.where(df['opp_difficulty'] <= 2, 1.2,
                                         np.where(df['opp_difficulty'] >= 4, 0.8, 1.0))
        
        # Home/away analysis if available
        if 'was_home' in df.columns:
            df['home_advantage'] = np.where(df['was_home'], 1.1, 0.9)
        else:
            df['home_advantage'] = 1.0
        
        return df
    
    def _create_enhanced_fixture_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add position-specific enhanced fixture features"""
        try:
            # Get enhanced fixture analysis
            enhanced_fixtures = self.enhanced_fixture_analyzer.analyze_enhanced_fixtures(gameweeks=5)
            
            if 'error' in enhanced_fixtures:
                return df
            
            # Add position-specific fixture difficulties
            for position in ['GKP', 'DEF', 'MID', 'FWD']:
                position_mask = df['position'] == position
                if position_mask.sum() == 0:
                    continue
                
                # Default values
                df.loc[position_mask, f'{position.lower()}_fixture_difficulty'] = 3
                df.loc[position_mask, f'{position.lower()}_clean_sheet_prob'] = 0.3
                df.loc[position_mask, f'{position.lower()}_expected_goals'] = 1.5
            
            # Try to map team difficulties to players
            if 'team_id' in df.columns:
                for gw_data in enhanced_fixtures.values():
                    if isinstance(gw_data, dict):
                        for team_id, team_fixtures in gw_data.items():
                            if isinstance(team_fixtures, dict) and 'difficulties' in team_fixtures:
                                team_mask = df['team_id'] == team_id
                                
                                for position, difficulty in team_fixtures['difficulties'].items():
                                    pos_mask = team_mask & (df['position'] == position)
                                    if pos_mask.sum() > 0:
                                        df.loc[pos_mask, f'{position.lower()}_fixture_difficulty'] = difficulty
                                        
                                        if 'clean_sheet_probability' in team_fixtures:
                                            df.loc[pos_mask, f'{position.lower()}_clean_sheet_prob'] = team_fixtures['clean_sheet_probability']
                                        
                                        if 'expected_goals' in team_fixtures:
                                            df.loc[pos_mask, f'{position.lower()}_expected_goals'] = team_fixtures['expected_goals']
            
            return df
            
        except Exception as e:
            # Fallback: return original df if enhanced fixture analysis fails
            return df
    
    def _create_value_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create value-based features for optimization"""
        if 'predicted_points' in df.columns:
            df['value_score'] = df['predicted_points'] / df['price']
            df['roi'] = (df['predicted_points'] - 2) / df['price']  # Assuming 2 pts baseline
        
        # Ownership and captaincy potential
        if 'selected_by_percent' in df.columns:
            ownership = pd.to_numeric(df['selected_by_percent'], errors='coerce').fillna(10.0)
            df['differential_potential'] = np.where(ownership < 5, 1.2, 1.0)
        
        return df
    
    def _create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create momentum and consistency features"""
        if 'total_points' in df.columns:
            # Recent vs historical performance
            df['recent_form'] = df.groupby('player_id')['total_points'].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
            df['season_avg'] = df.groupby('player_id')['total_points'].expanding(min_periods=1).mean().reset_index(level=0, drop=True)
            
            # Momentum indicator
            df['momentum'] = df['recent_form'] - df['season_avg']
            
            # Consistency score (inverse of coefficient of variation)
            rolling_std = df.groupby('player_id')['total_points'].rolling(5, min_periods=2).std().reset_index(level=0, drop=True)
            rolling_mean = df.groupby('player_id')['total_points'].rolling(5, min_periods=2).mean().reset_index(level=0, drop=True)
            df['consistency'] = 1 / (1 + (rolling_std / rolling_mean.replace(0, 1)))
        
        return df
    
    def _calculate_trend(self, series: pd.Series) -> float:
        """Calculate trend using simple linear regression slope"""
        if len(series) < 2:
            return 0.0
        x = np.arange(len(series))
        try:
            slope = np.polyfit(x, series.values, 1)[0]
            return slope
        except:
            return 0.0
    
    def _compute_next_fixture_difficulty(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        """Enhanced fixture difficulty calculation"""
        upcoming = fixtures.copy()
        if "finished" in upcoming.columns:
            upcoming = upcoming[~upcoming["finished"].astype(bool)]
        
        # Take next 5 fixtures for better analysis
        upcoming = upcoming.head(100)  # Reasonable limit
        
        cols = ["team_h", "team_a", "team_h_difficulty", "team_a_difficulty"]
        for c in cols:
            if c not in upcoming.columns:
                upcoming[c] = np.nan
        
        home_part = upcoming[["team_h", "team_a_difficulty"]].rename(
            columns={"team_h": "team_id", "team_a_difficulty": "opp_difficulty"})
        away_part = upcoming[["team_a", "team_h_difficulty"]].rename(
            columns={"team_a": "team_id", "team_h_difficulty": "opp_difficulty"})
        
        combined = pd.concat([home_part, away_part], axis=0, ignore_index=True)
        
        # Clean numeric data
        combined['opp_difficulty'] = pd.to_numeric(combined['opp_difficulty'], errors='coerce').fillna(3)
        combined['team_id'] = pd.to_numeric(combined['team_id'], errors='coerce')
        combined = combined.dropna(subset=['team_id'])
        
        # Weight recent fixtures more heavily
        combined['weight'] = np.exp(-0.1 * combined.index)
        
        # Use aggregation function to avoid deprecation warning
        def weighted_avg(group):
            if len(group) == 0:
                return 3.0
            try:
                return np.average(group['opp_difficulty'], weights=group['weight'])
            except:
                return group['opp_difficulty'].mean()
        
        grouped = combined.groupby("team_id").apply(weighted_avg, include_groups=False).reset_index()
        grouped.columns = ['team_id', 'opp_difficulty']
        
        return grouped

