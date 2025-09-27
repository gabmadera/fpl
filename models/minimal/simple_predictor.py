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
            'BRE': 0.8, 'NFO': 0.75, 'SOU': 0.7, 'IPS': 0.7, 'LEI': 0.75
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

        # Sort players by predicted points per price within each position
        for player in players:
            player['predicted_points'] = self.predict_player_points(player)
            player['value'] = player['predicted_points'] / max(player.get('price', 4.0), 4.0)

        # Separate by position and sort each position by value
        gkps = sorted([p for p in players if p.get('position') == 'GKP'], key=lambda x: x['value'], reverse=True)
        defs = sorted([p for p in players if p.get('position') == 'DEF'], key=lambda x: x['value'], reverse=True)
        mids = sorted([p for p in players if p.get('position') == 'MID'], key=lambda x: x['value'], reverse=True)
        fwds = sorted([p for p in players if p.get('position') == 'FWD'], key=lambda x: x['value'], reverse=True)

        # Select best 11 (formation 4-4-2)
        starters = []
        starters.extend(gkps[:1])  # 1 GKP
        starters.extend(defs[:4])  # 4 DEF
        starters.extend(mids[:4])  # 4 MID
        starters.extend(fwds[:2])  # 2 FWD

        # Bench (4 players)
        bench = []
        bench.extend(gkps[1:2])    # 1 GKP backup
        bench.extend(defs[4:5])    # 1 DEF backup
        bench.extend(mids[4:5])    # 1 MID backup
        bench.extend(fwds[2:3])    # 1 FWD backup

        # Remove empty entries
        starters = [p for p in starters if p]
        bench = [p for p in bench if p]

        total_predicted = sum(p['predicted_points'] for p in starters)

        # Find highest scoring player for captain
        captain_id = max(starters, key=lambda x: x['predicted_points'])['id'] if starters else 1
        vice_captain = sorted(starters, key=lambda x: x['predicted_points'], reverse=True)
        vice_captain_id = vice_captain[1]['id'] if len(vice_captain) > 1 else captain_id

        return {
            'starters': starters,
            'bench': bench,
            'formation': '4-4-2',
            'total_predicted_points': round(total_predicted, 1),
            'captain_id': captain_id,
            'vice_captain_id': vice_captain_id,
            'total_cost': sum(p.get('price', 5.0) for p in starters + bench)
        }

    def _generate_fallback_team(self) -> Dict[str, Any]:
        """Generate a fallback team when no data is available"""
        # Create realistic fallback players based on current 2024-25 season
        fallback_players = [
            # Goalkeepers
            {'id': 1, 'name': 'Jordan Pickford', 'position': 'GKP', 'team': 8, 'team_name': 'Everton', 'price': 5.0, 'total_points': 25, 'form': 5.0, 'team_short': 'EVE'},
            {'id': 2, 'name': 'Alphonse Areola', 'position': 'GKP', 'team': 19, 'team_name': 'West Ham', 'price': 4.5, 'total_points': 20, 'form': 4.0, 'team_short': 'WHU'},

            # Defenders
            {'id': 3, 'name': 'Gabriel Magalhães', 'position': 'DEF', 'team': 1, 'team_name': 'Arsenal', 'price': 6.0, 'total_points': 35, 'form': 7.0, 'team_short': 'ARS'},
            {'id': 4, 'name': 'Virgil van Dijk', 'position': 'DEF', 'team': 12, 'team_name': 'Liverpool', 'price': 6.5, 'total_points': 40, 'form': 8.0, 'team_short': 'LIV'},
            {'id': 5, 'name': 'Josko Gvardiol', 'position': 'DEF', 'team': 13, 'team_name': 'Man City', 'price': 5.5, 'total_points': 30, 'form': 6.0, 'team_short': 'MCI'},
            {'id': 6, 'name': 'Pedro Porro', 'position': 'DEF', 'team': 18, 'team_name': 'Tottenham', 'price': 5.5, 'total_points': 28, 'form': 5.5, 'team_short': 'TOT'},
            {'id': 7, 'name': 'Lewis Dunk', 'position': 'DEF', 'team': 5, 'team_name': 'Brighton', 'price': 4.5, 'total_points': 22, 'form': 4.5, 'team_short': 'BHA'},

            # Midfielders
            {'id': 8, 'name': 'Mohamed Salah', 'position': 'MID', 'team': 12, 'team_name': 'Liverpool', 'price': 12.5, 'total_points': 60, 'form': 12.0, 'team_short': 'LIV'},
            {'id': 9, 'name': 'Cole Palmer', 'position': 'MID', 'team': 6, 'team_name': 'Chelsea', 'price': 10.5, 'total_points': 50, 'form': 10.0, 'team_short': 'CHE'},
            {'id': 10, 'name': 'Bukayo Saka', 'position': 'MID', 'team': 1, 'team_name': 'Arsenal', 'price': 10.0, 'total_points': 45, 'form': 9.0, 'team_short': 'ARS'},
            {'id': 11, 'name': 'Morgan Gibbs-White', 'position': 'MID', 'team': 16, 'team_name': "Nott'm Forest", 'price': 6.5, 'total_points': 35, 'form': 7.0, 'team_short': 'NFO'},
            {'id': 12, 'name': 'Antoine Semenyo', 'position': 'MID', 'team': 3, 'team_name': 'Bournemouth', 'price': 6.0, 'total_points': 30, 'form': 6.0, 'team_short': 'BOU'},

            # Forwards
            {'id': 13, 'name': 'Erling Haaland', 'position': 'FWD', 'team': 13, 'team_name': 'Man City', 'price': 15.0, 'total_points': 80, 'form': 16.0, 'team_short': 'MCI'},
            {'id': 14, 'name': 'Alexander Isak', 'position': 'FWD', 'team': 15, 'team_name': 'Newcastle', 'price': 8.5, 'total_points': 45, 'form': 9.0, 'team_short': 'NEW'},
            {'id': 15, 'name': 'Yoane Wissa', 'position': 'FWD', 'team': 4, 'team_name': 'Brentford', 'price': 6.0, 'total_points': 25, 'form': 5.0, 'team_short': 'BRE'},
        ]

        # Add prediction data to each player
        for player in fallback_players:
            player['predicted_points'] = self.predict_player_points(player)
            player['confidence'] = 0.8
            player['availability'] = 1.0
            player['recent_form'] = player['form']
            player['value_rating'] = player['total_points'] / player['price']
            player['captaincy_potential'] = player['predicted_points'] * 1.2
            player['fpl_status'] = 'a'
            player['chance_next'] = 100.0
            player['team_id'] = player['team']
            player['opponent'] = 'Various'
            player['venue'] = 'H'

        # Select optimal team
        team_selection = self.predict_team_selection(fallback_players)

        # If team selection fails, create a basic fallback
        if not team_selection.get('starters'):
            return {
                'starters': fallback_players[:11],
                'bench': fallback_players[11:15],
                'formation': '4-4-2',
                'total_predicted_points': 75.0,
                'captain_id': 13,  # Haaland
                'vice_captain_id': 8,  # Salah
                'total_cost': 100.0,
                'status': 'fallback',
                'message': 'Using fallback team - FPL API unavailable'
            }

        return team_selection