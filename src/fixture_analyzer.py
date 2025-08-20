from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from .fpl_client import FPLClient


class FixtureAnalyzer:
    """Advanced fixture analysis for FPL predictions and user interface"""
    
    def __init__(self):
        self.fpl = FPLClient()
        self.logger = logging.getLogger(__name__)
        
        # Team strength ratings (will be calculated dynamically)
        self.team_strength = {}
        self.home_advantage = 0.3  # Average home advantage in points
        
    def get_fixture_difficulty_matrix(self, gameweeks: int = 5) -> Dict:
        """Get fixture difficulty matrix for next N gameweeks"""
        try:
            bootstrap = self.fpl.bootstrap_static()
            fixtures = self.fpl.fixtures()
            teams = pd.DataFrame(bootstrap.get("teams", []))
            
            if teams.empty or not fixtures:
                return {"error": "No fixture data available"}
            
            # Calculate team strength ratings
            self._calculate_team_strength(teams)
            
            # Process fixtures
            fixtures_df = pd.DataFrame(fixtures)
            current_gw = self._get_current_gameweek(bootstrap)
            
            # Filter for next N gameweeks
            target_gameweeks = list(range(current_gw, min(current_gw + gameweeks, 39)))
            upcoming_fixtures = fixtures_df[fixtures_df['event'].isin(target_gameweeks)].copy()
            
            if upcoming_fixtures.empty:
                return {"error": "No upcoming fixtures found"}
            
            # Create difficulty matrix
            difficulty_matrix = {}
            
            for _, fixture in upcoming_fixtures.iterrows():
                gw = fixture['event']
                home_team = fixture['team_h']
                away_team = fixture['team_a']
                
                if gw not in difficulty_matrix:
                    difficulty_matrix[gw] = {}
                
                # Calculate difficulty scores (1-5, where 5 is hardest)
                home_difficulty = self._calculate_fixture_difficulty(home_team, away_team, is_home=True)
                away_difficulty = self._calculate_fixture_difficulty(away_team, home_team, is_home=False)
                
                difficulty_matrix[gw][home_team] = {
                    'opponent': away_team,
                    'opponent_name': self._get_team_name(away_team, teams),
                    'is_home': True,
                    'difficulty': home_difficulty,
                    'kickoff_time': fixture.get('kickoff_time'),
                    'fixture_id': fixture['id']
                }
                
                difficulty_matrix[gw][away_team] = {
                    'opponent': home_team,
                    'opponent_name': self._get_team_name(home_team, teams),
                    'is_home': False,
                    'difficulty': away_difficulty,
                    'kickoff_time': fixture.get('kickoff_time'),
                    'fixture_id': fixture['id']
                }
            
            return {
                'current_gameweek': current_gw,
                'gameweeks_analyzed': target_gameweeks,
                'difficulty_matrix': difficulty_matrix,
                'team_names': {team['id']: team['name'] for _, team in teams.iterrows()},
                'team_short_names': {team['id']: team['short_name'] for _, team in teams.iterrows()},
                'legend': {
                    1: 'Very Easy',
                    2: 'Easy', 
                    3: 'Moderate',
                    4: 'Hard',
                    5: 'Very Hard'
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get fixture difficulty matrix: {e}")
            return {"error": str(e)}
    
    def get_player_fixture_run(self, player_id: int, gameweeks: int = 5) -> Dict:
        """Get fixture run for a specific player"""
        try:
            bootstrap = self.fpl.bootstrap_static()
            elements = pd.DataFrame(bootstrap.get("elements", []))
            
            if elements.empty:
                return {"error": "No player data available"}
            
            player = elements[elements['id'] == player_id]
            if player.empty:
                return {"error": "Player not found"}
            
            team_id = player.iloc[0]['team']
            player_name = player.iloc[0]['web_name']
            
            # Get team's fixture difficulty
            matrix = self.get_fixture_difficulty_matrix(gameweeks)
            if 'error' in matrix:
                return matrix
            
            player_fixtures = []
            total_difficulty = 0
            
            for gw in matrix['gameweeks_analyzed']:
                if team_id in matrix['difficulty_matrix'].get(gw, {}):
                    fixture = matrix['difficulty_matrix'][gw][team_id]
                    player_fixtures.append({
                        'gameweek': gw,
                        'opponent': fixture['opponent_name'],
                        'is_home': fixture['is_home'],
                        'difficulty': fixture['difficulty'],
                        'kickoff_time': fixture['kickoff_time']
                    })
                    total_difficulty += fixture['difficulty']
            
            avg_difficulty = total_difficulty / len(player_fixtures) if player_fixtures else 0
            
            return {
                'player_id': player_id,
                'player_name': player_name,
                'team_id': team_id,
                'fixtures': player_fixtures,
                'average_difficulty': round(avg_difficulty, 2),
                'fixture_rating': self._get_fixture_rating(avg_difficulty),
                'total_home_games': sum(1 for f in player_fixtures if f['is_home']),
                'total_away_games': sum(1 for f in player_fixtures if not f['is_home'])
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_fixture_based_multipliers(self, gameweek: int = None) -> Dict[int, float]:
        """Get fixture-based multipliers for ML predictions"""
        try:
            matrix = self.get_fixture_difficulty_matrix(1 if gameweek else 5)
            if 'error' in matrix:
                return {}
            
            multipliers = {}
            target_gw = gameweek or matrix['current_gameweek']
            
            if target_gw in matrix['difficulty_matrix']:
                for team_id, fixture_info in matrix['difficulty_matrix'][target_gw].items():
                    difficulty = fixture_info['difficulty']
                    is_home = fixture_info['is_home']
                    
                    # Convert difficulty to multiplier
                    base_multiplier = {
                        1: 1.25,  # Very easy opponent = +25% points
                        2: 1.10,  # Easy opponent = +10% points
                        3: 1.00,  # Moderate opponent = baseline
                        4: 0.85,  # Hard opponent = -15% points
                        5: 0.70   # Very hard opponent = -30% points
                    }.get(difficulty, 1.0)
                    
                    # Apply home advantage
                    if is_home:
                        base_multiplier *= 1.1  # +10% for playing at home
                    else:
                        base_multiplier *= 0.95  # -5% for playing away
                    
                    multipliers[team_id] = round(base_multiplier, 3)
            
            return multipliers
            
        except Exception as e:
            self.logger.error(f"Failed to get fixture multipliers: {e}")
            return {}
    
    def _calculate_team_strength(self, teams: pd.DataFrame):
        """Calculate dynamic team strength ratings"""
        try:
            # Use FPL's own strength ratings as base
            for _, team in teams.iterrows():
                team_id = team['id']
                
                # Get strength values - FPL API sometimes uses different scales
                home_strength = team.get('strength_overall_home', team.get('strength', 3))
                away_strength = team.get('strength_overall_away', team.get('strength', 3))
                
                # Convert to numeric and handle different scales
                home_strength = pd.to_numeric(home_strength, errors='coerce') or 3
                away_strength = pd.to_numeric(away_strength, errors='coerce') or 3
                
                # If values are on a large scale (1000+), normalize to 1-5
                if home_strength > 100:
                    # Assume it's on 1000-1500 scale, convert to 1-5
                    home_strength = 1 + (home_strength - 1000) / 100
                    away_strength = 1 + (away_strength - 1000) / 100
                    
                    # Clip to 1-5 range
                    home_strength = max(1, min(5, home_strength))
                    away_strength = max(1, min(5, away_strength))
                
                # Average strength (should now be on 1-5 scale)
                overall_strength = (home_strength + away_strength) / 2
                self.team_strength[team_id] = overall_strength
                
            print(f"Calculated team strengths (1-5 scale): {list(self.team_strength.values())[:5]}...")
                
        except Exception as e:
            # Fallback to default ratings if FPL data unavailable
            self.logger.warning(f"Using default team strength ratings: {e}")
            # Assign default ratings (can be updated with historical data)
            default_strength = 3.0
            for _, team in teams.iterrows():
                self.team_strength[team['id']] = default_strength
    
    def _calculate_fixture_difficulty(self, team_id: int, opponent_id: int, is_home: bool) -> int:
        """Calculate fixture difficulty on 1-5 scale"""
        try:
            team_strength = self.team_strength.get(team_id, 3.0)
            opponent_strength = self.team_strength.get(opponent_id, 3.0)
            
            # Higher opponent strength = harder fixture for your team
            base_difficulty = opponent_strength
            
            # Adjust for home/away
            if is_home:
                base_difficulty -= 0.5  # Home games are easier
            else:
                base_difficulty += 0.3  # Away games are harder
            
            # Convert to 1-5 scale
            if base_difficulty <= 2.0:
                return 1  # Very Easy
            elif base_difficulty <= 2.5:
                return 2  # Easy
            elif base_difficulty <= 3.5:
                return 3  # Moderate
            elif base_difficulty <= 4.0:
                return 4  # Hard
            else:
                return 5  # Very Hard
                
        except Exception:
            return 3  # Default moderate difficulty
    
    def _get_current_gameweek(self, bootstrap: Dict) -> int:
        """Get current gameweek from bootstrap data"""
        events = bootstrap.get("events", [])
        for event in events:
            if event.get("is_current", False):
                return event.get("id", 1)
        return 1  # Default to GW1 if no current event
    
    def _get_team_name(self, team_id: int, teams: pd.DataFrame) -> str:
        """Get team name from team ID"""
        team = teams[teams['id'] == team_id]
        if not team.empty:
            return team.iloc[0]['short_name']
        return f"Team {team_id}"
    
    def _get_fixture_rating(self, avg_difficulty: float) -> str:
        """Convert average difficulty to rating"""
        if avg_difficulty <= 2.0:
            return "Excellent"
        elif avg_difficulty <= 2.5:
            return "Good"
        elif avg_difficulty <= 3.5:
            return "Average"
        elif avg_difficulty <= 4.0:
            return "Difficult"
        else:
            return "Very Difficult"