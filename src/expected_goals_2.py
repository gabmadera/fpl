from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass


@dataclass
class ShotQuality:
    """Represents shot quality metrics"""
    xg_value: float
    shot_angle: float
    distance_from_goal: float
    body_part: str  # foot, head, other
    assist_type: str  # through_ball, cross, corner, etc.
    situation: str  # open_play, set_piece, penalty, etc.
    pressure: float  # defensive pressure level
    
    
class ExpectedGoals2:
    """Advanced Expected Goals 2.0 with shot quality analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Shot quality multipliers
        self.body_part_multipliers = {
            'foot': 1.0,
            'head': 0.8,  # Headers generally less accurate
            'other': 0.6  # Other body parts
        }
        
        self.assist_type_multipliers = {
            'through_ball': 1.3,  # High quality chances
            'cross': 0.9,         # Crosses often contested
            'corner': 0.7,        # Crowded box
            'free_kick': 0.8,     # Set piece delivery
            'cutback': 1.2,       # High quality assists
            'pass': 1.0,          # Regular assists
            'none': 0.8           # No assist (individual effort)
        }
        
        self.situation_multipliers = {
            'open_play': 1.0,
            'corner': 0.7,
            'free_kick': 0.8,
            'penalty': 1.0,  # Already high xG
            'counter_attack': 1.4,  # High quality chances
            'set_piece': 0.8
        }
        
        # Defensive action impact
        self.defensive_multipliers = {
            'blocks': 0.6,        # Shots blocked
            'tackles': 0.5,       # Tackle attempts
            'interceptions': 0.4, # Intercepted passes
            'clearances': 0.3     # Defensive clearances
        }
    
    def calculate_advanced_xg(self, shot_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced xG with shot quality analysis"""
        try:
            if shot_data.empty:
                return shot_data
            
            # Start with base xG values
            advanced_data = shot_data.copy()
            
            # Apply shot quality adjustments
            advanced_data['base_xg'] = advanced_data.get('xG', 0.0)
            advanced_data['quality_adjusted_xg'] = advanced_data['base_xg']
            
            # Apply body part adjustments
            if 'body_part' in advanced_data.columns:
                for body_part, multiplier in self.body_part_multipliers.items():
                    mask = advanced_data['body_part'] == body_part
                    advanced_data.loc[mask, 'quality_adjusted_xg'] *= multiplier
            
            # Apply assist type adjustments  
            if 'assist_type' in advanced_data.columns:
                for assist_type, multiplier in self.assist_type_multipliers.items():
                    mask = advanced_data['assist_type'] == assist_type
                    advanced_data.loc[mask, 'quality_adjusted_xg'] *= multiplier
            
            # Apply situation adjustments
            if 'situation' in advanced_data.columns:
                for situation, multiplier in self.situation_multipliers.items():
                    mask = advanced_data['situation'] == situation
                    advanced_data.loc[mask, 'quality_adjusted_xg'] *= multiplier
            
            # Apply defensive pressure adjustments
            if 'defensive_pressure' in advanced_data.columns:
                pressure_multiplier = 1 - (advanced_data['defensive_pressure'] * 0.3)
                pressure_multiplier = np.clip(pressure_multiplier, 0.3, 1.0)
                advanced_data['quality_adjusted_xg'] *= pressure_multiplier
            
            # Calculate final xG2.0 values
            advanced_data['xg_2'] = np.clip(advanced_data['quality_adjusted_xg'], 0.01, 0.95)
            
            return advanced_data
            
        except Exception as e:
            self.logger.error(f"Advanced xG calculation failed: {e}")
            return shot_data
    
    def calculate_player_xg2_metrics(self, player_data: pd.DataFrame, shots_data: pd.DataFrame = None) -> pd.DataFrame:
        """Calculate xG 2.0 metrics for players"""
        try:
            if player_data.empty:
                return player_data
            
            enhanced_data = player_data.copy()
            
            # If we have detailed shot data, use it
            if shots_data is not None and not shots_data.empty:
                player_xg2_metrics = self._calculate_detailed_xg2_metrics(shots_data)
                enhanced_data = enhanced_data.merge(
                    player_xg2_metrics, 
                    left_on='player_id', 
                    right_on='player_id', 
                    how='left'
                )
            else:
                # Use enhanced estimation based on available data
                enhanced_data = self._estimate_xg2_from_basic_data(enhanced_data)
            
            return enhanced_data
            
        except Exception as e:
            self.logger.error(f"Player xG2 metrics calculation failed: {e}")
            return player_data
    
    def _calculate_detailed_xg2_metrics(self, shots_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate detailed xG2.0 metrics from shot-level data"""
        try:
            # Apply quality adjustments
            enhanced_shots = self.calculate_advanced_xg(shots_data)
            
            # Aggregate by player
            player_metrics = enhanced_shots.groupby('player_id').agg({
                'base_xg': ['sum', 'mean', 'count'],
                'xg_2': ['sum', 'mean'],
                'quality_adjusted_xg': 'sum',
                'goals': 'sum' if 'goals' in enhanced_shots.columns else lambda x: 0
            }).round(3)
            
            # Flatten column names
            player_metrics.columns = ['_'.join(col).strip() if col[1] else col[0] for col in player_metrics.columns]
            
            # Calculate efficiency metrics
            player_metrics['xg2_per_shot'] = player_metrics['xg_2_mean']
            player_metrics['xg2_total'] = player_metrics['xg_2_sum']
            player_metrics['shot_quality_index'] = (
                player_metrics['xg_2_sum'] / player_metrics['base_xg_sum']
            ).fillna(1.0)
            
            # Goals vs xG2.0 performance
            if 'goals_sum' in player_metrics.columns:
                player_metrics['xg2_overperformance'] = (
                    player_metrics['goals_sum'] - player_metrics['xg2_total']
                )
                player_metrics['xg2_conversion_rate'] = (
                    player_metrics['goals_sum'] / player_metrics['xg2_total']
                ).fillna(0)
            
            return player_metrics.reset_index()
            
        except Exception as e:
            self.logger.error(f"Detailed xG2 metrics calculation failed: {e}")
            return pd.DataFrame()
    
    def _estimate_xg2_from_basic_data(self, player_data: pd.DataFrame) -> pd.DataFrame:
        """Estimate xG2.0 from basic player data when shot details unavailable"""
        try:
            enhanced_data = player_data.copy()
            
            # Base xG values
            base_xg = enhanced_data.get('expected_goals', enhanced_data.get('xG', 0)).fillna(0)
            
            # Estimate shot quality based on available metrics
            shot_quality_multiplier = 1.0
            
            # Players with high assists likely create/receive better chances
            if 'assists' in enhanced_data.columns:
                assists = pd.to_numeric(enhanced_data['assists'], errors='coerce').fillna(0)
                shot_quality_multiplier *= (1 + (assists / 10) * 0.1)  # Max 10% bonus
            
            # Players with high key passes likely get better chances
            if 'key_passes' in enhanced_data.columns:
                key_passes = pd.to_numeric(enhanced_data['key_passes'], errors='coerce').fillna(0)
                shot_quality_multiplier *= (1 + (key_passes / 20) * 0.1)
            
            # Position-based adjustments
            if 'position' in enhanced_data.columns:
                position_multipliers = {
                    'FWD': 1.1,  # Forwards get better chances
                    'MID': 1.0,  # Baseline
                    'DEF': 0.9,  # Defenders get fewer quality chances
                    'GKP': 0.8   # Goalkeepers rarely shoot
                }
                
                for pos, multiplier in position_multipliers.items():
                    mask = enhanced_data['position'] == pos
                    shot_quality_multiplier = np.where(mask, multiplier, shot_quality_multiplier)
            
            # Team strength factor (better teams create better chances)
            if 'team_strength' in enhanced_data.columns:
                team_strength = pd.to_numeric(enhanced_data['team_strength'], errors='coerce').fillna(3.0)
                strength_multiplier = 0.8 + (team_strength / 5.0) * 0.4  # 0.8 to 1.2 range
                shot_quality_multiplier *= strength_multiplier
            
            # Apply shot quality adjustment
            enhanced_data['xg_2'] = base_xg * shot_quality_multiplier
            enhanced_data['shot_quality_index'] = shot_quality_multiplier
            enhanced_data['xg2_per_90'] = enhanced_data['xg_2'] / enhanced_data.get('minutes', 90) * 90
            
            # Clip to reasonable bounds
            enhanced_data['xg_2'] = np.clip(enhanced_data['xg_2'], 0, 1.5)
            
            return enhanced_data
            
        except Exception as e:
            self.logger.error(f"xG2 estimation failed: {e}")
            return player_data
    
    def calculate_defensive_xg_metrics(self, defensive_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate defensive xG2.0 metrics"""
        try:
            if defensive_data.empty:
                return defensive_data
            
            enhanced_data = defensive_data.copy()
            
            # Defensive actions impact on opponent xG
            defensive_actions = ['blocks', 'tackles', 'interceptions', 'clearances']
            
            xg_prevented = 0
            for action in defensive_actions:
                if action in enhanced_data.columns:
                    action_count = pd.to_numeric(enhanced_data[action], errors='coerce').fillna(0)
                    multiplier = self.defensive_multipliers.get(action, 0.1)
                    xg_prevented += action_count * multiplier
            
            enhanced_data['xg_prevented'] = xg_prevented
            enhanced_data['defensive_xg_impact'] = xg_prevented / enhanced_data.get('minutes', 90) * 90
            
            return enhanced_data
            
        except Exception as e:
            self.logger.error(f"Defensive xG calculation failed: {e}")
            return defensive_data
    
    def get_xg2_insights(self, player_data: pd.DataFrame) -> Dict[str, any]:
        """Generate insights from xG2.0 analysis"""
        try:
            insights = {}
            
            if 'xg_2' not in player_data.columns:
                return {"error": "No xG2.0 data available"}
            
            # Top xG2.0 performers
            top_xg2 = player_data.nlargest(10, 'xg_2')[['name', 'position', 'xg_2', 'shot_quality_index']]
            insights['top_xg2_players'] = top_xg2.to_dict('records')
            
            # Best shot quality (xG2/shot ratio)
            if 'xg2_per_shot' in player_data.columns:
                best_quality = player_data.nlargest(10, 'xg2_per_shot')[['name', 'position', 'xg2_per_shot']]
                insights['best_shot_quality'] = best_quality.to_dict('records')
            
            # Most undervalued (high xG2.0, low ownership)
            if 'selected_by_percent' in player_data.columns:
                player_data['xg2_value'] = player_data['xg_2'] / np.log1p(player_data['selected_by_percent'])
                undervalued = player_data.nlargest(10, 'xg2_value')[['name', 'position', 'xg_2', 'selected_by_percent']]
                insights['undervalued_xg2'] = undervalued.to_dict('records')
            
            # Position-based xG2.0 averages
            position_xg2 = player_data.groupby('position')['xg_2'].agg(['mean', 'std', 'count']).round(3)
            insights['position_xg2_stats'] = position_xg2.to_dict('index')
            
            return insights
            
        except Exception as e:
            self.logger.error(f"xG2 insights generation failed: {e}")
            return {"error": str(e)}
    
    def integrate_with_predictions(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Integrate xG2.0 metrics with FPL predictions"""
        try:
            enhanced_predictions = predictions_df.copy()
            
            # Calculate xG2.0 metrics
            enhanced_predictions = self.calculate_player_xg2_metrics(enhanced_predictions)
            
            # Adjust predictions based on xG2.0
            if 'xg_2' in enhanced_predictions.columns:
                # xG2.0 boost for attacking players
                xg2_boost = enhanced_predictions['xg_2'] * 6  # Rough conversion to points
                
                # Apply position-specific multipliers
                position_multipliers = {'FWD': 1.0, 'MID': 0.8, 'DEF': 0.6, 'GKP': 0.1}
                for pos, multiplier in position_multipliers.items():
                    mask = enhanced_predictions['position'] == pos
                    enhanced_predictions.loc[mask, 'xg2_boost'] = xg2_boost[mask] * multiplier
                
                # Add xG2.0 component to predictions
                if 'predicted_points' in enhanced_predictions.columns:
                    enhanced_predictions['base_prediction'] = enhanced_predictions['predicted_points']
                    enhanced_predictions['predicted_points'] += enhanced_predictions.get('xg2_boost', 0)
                    
                    # Clip to reasonable bounds
                    enhanced_predictions['predicted_points'] = np.clip(
                        enhanced_predictions['predicted_points'], 0, 20
                    )
            
            return enhanced_predictions
            
        except Exception as e:
            self.logger.error(f"xG2 integration failed: {e}")
            return predictions_df