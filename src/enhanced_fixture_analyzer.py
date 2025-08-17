from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from .fpl_client import FPLClient


class EnhancedFixtureAnalyzer:
    """Advanced fixture analysis with position-specific difficulty and context"""
    
    def __init__(self):
        self.fpl = FPLClient()
        self.logger = logging.getLogger(__name__)
        
        # Enhanced team strength by position
        self.team_defensive_strength = {}  # How good at preventing goals
        self.team_attacking_strength = {}  # How good at scoring goals
        self.team_clean_sheet_probability = {}
        self.team_goals_conceded_rate = {}
        
        # Position-specific multipliers
        self.position_multipliers = {
            'GKP': {'clean_sheet_weight': 0.7, 'save_weight': 0.3},
            'DEF': {'clean_sheet_weight': 0.5, 'attacking_weight': 0.3, 'defensive_weight': 0.2},
            'MID': {'attacking_weight': 0.6, 'possession_weight': 0.2, 'defensive_weight': 0.2},
            'FWD': {'attacking_weight': 0.8, 'big_chance_weight': 0.2}
        }
        
        # Recent form weighting (exponential decay)
        self.form_weights = [0.4, 0.3, 0.2, 0.1]  # Last 4 games
        
    def analyze_enhanced_fixtures(self, gameweeks: int = 5) -> Dict:
        """Enhanced fixture analysis with position-specific difficulty"""
        try:
            bootstrap = self.fpl.bootstrap_static()
            fixtures = self.fpl.fixtures()
            teams = pd.DataFrame(bootstrap.get("teams", []))
            
            if teams.empty or not fixtures:
                return {"error": "No fixture data available"}
            
            # Calculate enhanced team strengths
            self._calculate_enhanced_team_strength(teams)
            
            # Process fixtures with enhanced analysis
            fixtures_df = pd.DataFrame(fixtures)
            current_gw = self._get_current_gameweek(bootstrap)
            
            # Filter for target gameweeks
            target_gameweeks = list(range(current_gw, min(current_gw + gameweeks, 39)))
            upcoming_fixtures = fixtures_df[fixtures_df['event'].isin(target_gameweeks)].copy()
            
            if upcoming_fixtures.empty:
                return {"error": "No upcoming fixtures found"}
            
            # Enhanced difficulty matrix with position-specific analysis
            enhanced_matrix = {}
            
            for _, fixture in upcoming_fixtures.iterrows():
                gw = fixture['event']
                home_team = fixture['team_h']
                away_team = fixture['team_a']
                
                if gw not in enhanced_matrix:
                    enhanced_matrix[gw] = {}
                
                # Calculate position-specific difficulties
                home_difficulties = self._calculate_position_specific_difficulty(
                    home_team, away_team, is_home=True
                )
                away_difficulties = self._calculate_position_specific_difficulty(
                    away_team, home_team, is_home=False
                )
                
                enhanced_matrix[gw][home_team] = {
                    'opponent': away_team,
                    'opponent_name': self._get_team_name(away_team, teams),
                    'is_home': True,
                    'difficulties': home_difficulties,
                    'overall_difficulty': home_difficulties['overall'],
                    'clean_sheet_probability': self._calculate_clean_sheet_probability(home_team, away_team, True),
                    'expected_goals': self._calculate_expected_goals(home_team, away_team, True),
                    'kickoff_time': fixture.get('kickoff_time'),
                    'fixture_id': fixture['id']
                }
                
                enhanced_matrix[gw][away_team] = {
                    'opponent': home_team,
                    'opponent_name': self._get_team_name(home_team, teams),
                    'is_home': False,
                    'difficulties': away_difficulties,
                    'overall_difficulty': away_difficulties['overall'],
                    'clean_sheet_probability': self._calculate_clean_sheet_probability(away_team, home_team, False),
                    'expected_goals': self._calculate_expected_goals(away_team, home_team, False),
                    'kickoff_time': fixture.get('kickoff_time'),
                    'fixture_id': fixture['id']
                }
            
            return {
                'current_gameweek': current_gw,
                'gameweeks_analyzed': target_gameweeks,
                'enhanced_matrix': enhanced_matrix,
                'team_names': {team['id']: team['name'] for _, team in teams.iterrows()},
                'team_short_names': {team['id']: team['short_name'] for _, team in teams.iterrows()},
                'position_legend': {
                    'GKP': 'Clean sheet probability focus',
                    'DEF': 'Defensive solidity + attacking threat',
                    'MID': 'Balanced attacking/defensive analysis',
                    'FWD': 'Attacking opportunity focus'
                }
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced fixture analysis failed: {e}")
            return {"error": str(e)}
    
    def _calculate_enhanced_team_strength(self, teams: pd.DataFrame):
        """Calculate position-specific team strengths"""
        try:
            for _, team in teams.iterrows():
                team_id = team['id']
                
                # Base strength from FPL API
                attack_strength = team.get('strength_attack_home', 3) + team.get('strength_attack_away', 3)
                defense_strength = team.get('strength_defence_home', 3) + team.get('strength_defence_away', 3)
                overall_strength = team.get('strength_overall_home', 3) + team.get('strength_overall_away', 3)
                
                # Normalize if on FPL's 1000+ scale
                if attack_strength > 100:
                    attack_strength = 1 + (attack_strength - 2000) / 200  # 2000 = home + away base
                    defense_strength = 1 + (defense_strength - 2000) / 200
                    overall_strength = 1 + (overall_strength - 2000) / 200
                
                # Store position-specific strengths
                self.team_attacking_strength[team_id] = max(1, min(5, attack_strength / 2))
                self.team_defensive_strength[team_id] = max(1, min(5, defense_strength / 2))
                
                # Calculate derived metrics
                self.team_clean_sheet_probability[team_id] = min(0.8, max(0.1, 
                    (5 - self.team_defensive_strength[team_id]) / 4 * 0.5 + 0.1
                ))
                
                self.team_goals_conceded_rate[team_id] = max(0.5, min(3.0,
                    self.team_defensive_strength[team_id] * 0.6
                ))
            
            print(f"Enhanced team analysis complete - {len(self.team_attacking_strength)} teams")
            
        except Exception as e:
            self.logger.error(f"Enhanced team strength calculation failed: {e}")
            # Fallback to default values
            for _, team in teams.iterrows():
                team_id = team['id']
                self.team_attacking_strength[team_id] = 3.0
                self.team_defensive_strength[team_id] = 3.0
                self.team_clean_sheet_probability[team_id] = 0.3
                self.team_goals_conceded_rate[team_id] = 1.5
    
    def _calculate_position_specific_difficulty(self, team_id: int, opponent_id: int, is_home: bool) -> Dict[str, int]:
        """Calculate difficulty specific to each position"""
        try:
            opponent_def_strength = self.team_defensive_strength.get(opponent_id, 3.0)
            opponent_att_strength = self.team_attacking_strength.get(opponent_id, 3.0)
            team_att_strength = self.team_attacking_strength.get(team_id, 3.0)
            
            # Home advantage adjustments
            home_bonus = 0.3 if is_home else -0.3
            
            difficulties = {}
            
            # Goalkeeper difficulty (focus on clean sheet probability)
            gkp_base = opponent_att_strength - home_bonus
            difficulties['GKP'] = self._convert_to_difficulty_scale(gkp_base)
            
            # Defender difficulty (balance between attacking and defensive)
            def_defensive = opponent_att_strength - home_bonus  # Harder if opponent attacks well
            def_attacking = opponent_def_strength + home_bonus  # Harder if opponent defends well
            def_base = (def_defensive * 0.6 + def_attacking * 0.4)
            difficulties['DEF'] = self._convert_to_difficulty_scale(def_base)
            
            # Midfielder difficulty (balanced analysis)
            mid_base = ((opponent_def_strength + opponent_att_strength) / 2) + home_bonus * 0.5
            difficulties['MID'] = self._convert_to_difficulty_scale(mid_base)
            
            # Forward difficulty (focus on scoring opportunities)
            fwd_base = opponent_def_strength + home_bonus
            difficulties['FWD'] = self._convert_to_difficulty_scale(fwd_base)
            
            # Overall difficulty (weighted average)
            difficulties['overall'] = int(np.mean([
                difficulties['GKP'] * 0.15,
                difficulties['DEF'] * 0.25, 
                difficulties['MID'] * 0.35,
                difficulties['FWD'] * 0.25
            ]))
            
            return difficulties
            
        except Exception:
            # Fallback to moderate difficulty
            return {'GKP': 3, 'DEF': 3, 'MID': 3, 'FWD': 3, 'overall': 3}
    
    def _convert_to_difficulty_scale(self, strength_value: float) -> int:
        """Convert strength value to 1-5 difficulty scale"""
        if strength_value <= 2.0:
            return 1  # Very Easy
        elif strength_value <= 2.5:
            return 2  # Easy  
        elif strength_value <= 3.5:
            return 3  # Moderate
        elif strength_value <= 4.0:
            return 4  # Hard
        else:
            return 5  # Very Hard
    
    def _calculate_clean_sheet_probability(self, team_id: int, opponent_id: int, is_home: bool) -> float:
        """Calculate probability of clean sheet for defenders/goalkeepers"""
        try:
            team_def_strength = self.team_defensive_strength.get(team_id, 3.0)
            opponent_att_strength = self.team_attacking_strength.get(opponent_id, 3.0)
            
            # Base probability calculation
            strength_diff = team_def_strength - opponent_att_strength
            base_prob = 0.3 + (strength_diff / 10)  # Base 30% + adjustment
            
            # Home advantage
            if is_home:
                base_prob += 0.05
            else:
                base_prob -= 0.05
            
            return max(0.05, min(0.75, base_prob))
            
        except Exception:
            return 0.25  # Default 25% clean sheet probability
    
    def _calculate_expected_goals(self, team_id: int, opponent_id: int, is_home: bool) -> float:
        """Calculate expected goals for attacking players"""
        try:
            team_att_strength = self.team_attacking_strength.get(team_id, 3.0)
            opponent_def_strength = self.team_defensive_strength.get(opponent_id, 3.0)
            
            # Base expected goals
            strength_diff = team_att_strength - opponent_def_strength
            base_xg = 1.3 + (strength_diff / 5)  # Base 1.3 goals + adjustment
            
            # Home advantage
            if is_home:
                base_xg += 0.2
            else:
                base_xg -= 0.1
            
            return max(0.3, min(4.0, base_xg))
            
        except Exception:
            return 1.4  # Default expected goals
    
    def get_position_specific_multipliers(self, position: str, fixture_data: Dict) -> Dict[str, float]:
        """Get prediction multipliers based on position and fixture"""
        try:
            if position not in self.position_multipliers:
                return {'prediction_multiplier': 1.0}
            
            pos_config = self.position_multipliers[position]
            fixture_difficulty = fixture_data.get('difficulties', {}).get(position, 3)
            clean_sheet_prob = fixture_data.get('clean_sheet_probability', 0.25)
            expected_goals = fixture_data.get('expected_goals', 1.4)
            
            multipliers = {}
            
            if position == 'GKP':
                # Goalkeepers benefit from high clean sheet probability
                cs_multiplier = 0.5 + (clean_sheet_prob * 1.5)  # 0.5-1.6 range
                save_multiplier = 1.0 + ((5 - fixture_difficulty) / 10)  # More saves in harder games
                multipliers['prediction_multiplier'] = (cs_multiplier + save_multiplier) / 2
                
            elif position == 'DEF':
                # Defenders benefit from clean sheets but also attacking opportunities
                cs_multiplier = 0.7 + (clean_sheet_prob * 1.0)
                att_multiplier = 0.8 + (expected_goals / 4)  # Attacking returns
                multipliers['prediction_multiplier'] = (cs_multiplier * 0.6 + att_multiplier * 0.4)
                
            elif position == 'MID':
                # Midfielders balanced between attacking and defensive actions
                att_multiplier = 0.8 + (expected_goals / 3)
                def_multiplier = 0.9 + (clean_sheet_prob * 0.5)
                multipliers['prediction_multiplier'] = (att_multiplier * 0.7 + def_multiplier * 0.3)
                
            elif position == 'FWD':
                # Forwards primarily benefit from goal scoring opportunities
                att_multiplier = 0.6 + (expected_goals / 2)
                multipliers['prediction_multiplier'] = att_multiplier
            
            # Ensure multiplier is in reasonable range
            multipliers['prediction_multiplier'] = max(0.4, min(2.0, multipliers.get('prediction_multiplier', 1.0)))
            
            return multipliers
            
        except Exception as e:
            self.logger.error(f"Failed to calculate position multipliers: {e}")
            return {'prediction_multiplier': 1.0}
    
    def _get_current_gameweek(self, bootstrap: Dict) -> int:
        """Get current gameweek from bootstrap data"""
        events = bootstrap.get("events", [])
        for event in events:
            if event.get("is_current", False):
                return event.get("id", 1)
        return 1
    
    def _get_team_name(self, team_id: int, teams: pd.DataFrame) -> str:
        """Get team name from team ID"""
        team = teams[teams['id'] == team_id]
        if not team.empty:
            return team.iloc[0]['short_name']
        return f"Team {team_id}"