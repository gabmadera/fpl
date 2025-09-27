"""
Simple prediction model for cloud deployment bootstrap
Used when ML models aren't available or as fallback
"""

import json
from typing import Dict, List, Any
from pathlib import Path


class SimplePredictor:
    """Simple rule-based predictor for bootstrap/fallback"""

    def __init__(self):
        self.position_base_points = {
            'GKP': 4.0,
            'DEF': 5.0,
            'MID': 6.0,
            'FWD': 7.0
        }

        self.team_multipliers = {
            'MCI': 1.3, 'LIV': 1.25, 'ARS': 1.2, 'TOT': 1.15, 'CHE': 1.1,
            'MUN': 1.1, 'NEW': 1.05, 'AVL': 1.0, 'BHA': 0.95, 'FUL': 0.9,
            'CRY': 0.9, 'BOU': 0.85, 'EVE': 0.85, 'WHU': 0.85, 'WOL': 0.8,
            'BRE': 0.8, 'NFO': 0.75, 'SHU': 0.7, 'LUT': 0.7, 'BUR': 0.65
        }

    def predict_player_points(self, player_data: Dict[str, Any]) -> float:
        """Predict points for a single player"""
        position = player_data.get('position', 'MID')
        team = player_data.get('team_short', 'MID')
        price = player_data.get('price', 5.0)

        # Base points by position
        base_points = self.position_base_points.get(position, 5.0)

        # Team multiplier
        team_mult = self.team_multipliers.get(team, 0.9)

        # Price factor (expensive players tend to score more)
        price_factor = min(price / 10.0, 1.5)

        # Simple calculation
        predicted = base_points * team_mult * price_factor

        # Add some randomness to avoid identical scores
        import random
        predicted += random.uniform(-0.5, 0.5)

        return max(0.0, round(predicted, 1))

    def predict_team_selection(self, players: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Predict optimal team selection"""
        if not players:
            return self._generate_fallback_team()

        # Sort players by predicted points per price
        for player in players:
            player['predicted_points'] = self.predict_player_points(player)
            player['value'] = player['predicted_points'] / max(player.get('price', 4.0), 4.0)

        players_sorted = sorted(players, key=lambda x: x['value'], reverse=True)

        # Simple team selection by position
        team = {
            'GKP': players_sorted[:2] if len([p for p in players_sorted if p.get('position') == 'GKP']) >= 2 else [],
            'DEF': players_sorted[:5] if len([p for p in players_sorted if p.get('position') == 'DEF']) >= 5 else [],
            'MID': players_sorted[:5] if len([p for p in players_sorted if p.get('position') == 'MID']) >= 5 else [],
            'FWD': players_sorted[:3] if len([p for p in players_sorted if p.get('position') == 'FWD']) >= 3 else []
        }

        # Select best 11
        starters = []
        starters.extend(team['GKP'][:1])  # 1 GKP
        starters.extend(team['DEF'][:4])  # 4 DEF
        starters.extend(team['MID'][:4])  # 4 MID
        starters.extend(team['FWD'][:2])  # 2 FWD

        # Bench
        bench = []
        bench.extend(team['GKP'][1:2])   # 1 GKP
        bench.extend(team['DEF'][4:5])   # 1 DEF
        bench.extend(team['MID'][4:5])   # 1 MID
        bench.extend(team['FWD'][2:3])   # 1 FWD

        total_predicted = sum(p['predicted_points'] for p in starters)

        return {
            'starters': starters[:11],
            'bench': bench[:4],
            'formation': '4-4-2',
            'total_predicted_points': round(total_predicted, 1),
            'captain_id': starters[0]['id'] if starters else 1,
            'vice_captain_id': starters[1]['id'] if len(starters) > 1 else 1,
            'total_cost': sum(p.get('price', 5.0) for p in starters + bench)
        }

    def _generate_fallback_team(self) -> Dict[str, Any]:
        """Generate a fallback team when no data is available"""
        return {
            'starters': [],
            'bench': [],
            'formation': '4-4-2',
            'total_predicted_points': 65.0,
            'captain_id': 1,
            'vice_captain_id': 2,
            'total_cost': 100.0,
            'status': 'fallback',
            'message': 'Using fallback team - FPL API unavailable'
        }