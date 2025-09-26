from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta
from pathlib import Path

from .fpl_client import FPLClient
from .cache_manager import CacheManager


class EnhancedPredictionModel:
    """
    Enhanced FPL prediction model with comprehensive factor analysis
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.fpl = FPLClient()
        self.cache_manager = CacheManager()

        # Enhanced weighting system
        self.factor_weights = {
            # Core performance factors
            'recent_form': 0.25,           # Last 5 gameweeks performance
            'season_average': 0.15,        # Overall season performance
            'position_strength': 0.12,     # Position-specific analysis

            # Fixture and opponent factors
            'fixture_difficulty': 0.18,    # Opponent strength and venue
            'opponent_defensive_form': 0.08, # Recent defensive performance
            'venue_advantage': 0.05,       # Home/away impact

            # Value and strategic factors
            'price_efficiency': 0.08,      # Points per million value
            'ownership_contrarian': 0.03,  # Low ownership gems
            'captaincy_potential': 0.04,   # High ceiling players

            # Special circumstances
            'injury_rotation_risk': -0.12, # Availability concerns
            'penalty_set_pieces': 0.06,   # Bonus point opportunities
            'team_attacking_form': 0.10,  # Team's recent attacking performance
            'momentum_trend': 0.08         # Performance trajectory
        }

        # Position-specific adjustments
        self.position_multipliers = {
            'GKP': {'clean_sheet_weight': 1.5, 'save_bonus': 1.2},
            'DEF': {'clean_sheet_weight': 1.3, 'attacking_bonus': 1.1},
            'MID': {'assist_weight': 1.2, 'goal_bonus': 1.3},
            'FWD': {'goal_weight': 1.4, 'penalty_weight': 1.3}
        }

    def generate_enhanced_predictions(self, current_gw: int) -> pd.DataFrame:
        """Generate predictions using enhanced factor analysis"""

        try:
            # Get base player data
            players_df = self._get_enhanced_player_data(current_gw)

            if players_df.empty:
                self.logger.error("No player data available for enhanced predictions")
                return pd.DataFrame()

            # Apply enhanced factor analysis
            players_df = self._apply_enhanced_factors(players_df, current_gw)

            # Calculate final predictions
            players_df = self._calculate_enhanced_predictions(players_df)

            # Apply position-specific adjustments
            players_df = self._apply_position_adjustments(players_df)

            # Add confidence scores
            players_df = self._add_confidence_scores(players_df)

            self.logger.info(f"Generated enhanced predictions for {len(players_df)} players")
            return players_df

        except Exception as e:
            self.logger.error(f"Enhanced prediction generation failed: {e}")
            return pd.DataFrame()

    def _get_enhanced_player_data(self, current_gw: int) -> pd.DataFrame:
        """Get comprehensive player data with enhanced metrics"""

        try:
            # Get bootstrap data
            bs = self.fpl.bootstrap_static()
            if not bs:
                return pd.DataFrame()

            players = bs.get('elements', [])
            teams = {t['id']: t for t in bs.get('teams', [])}
            events = bs.get('events', [])

            enhanced_data = []

            for player in players:
                # Basic player info
                team_info = teams.get(player['team'], {})

                player_data = {
                    'id': player['id'],
                    'name': f"{player.get('first_name', '')} {player.get('second_name', '')}".strip(),
                    'position': self._get_position_name(player['element_type']),
                    'team': player['team'],
                    'team_name': team_info.get('name', 'Unknown'),
                    'price': player['now_cost'] / 10.0,

                    # Core performance metrics
                    'total_points': player['total_points'],
                    'points_per_game': player.get('points_per_game', 0),
                    'form': float(player.get('form', 0)),
                    'selected_by_percent': float(player.get('selected_by_percent', 0)),

                    # Enhanced performance metrics
                    'goals_scored': player.get('goals_scored', 0),
                    'assists': player.get('assists', 0),
                    'clean_sheets': player.get('clean_sheets', 0),
                    'goals_conceded': player.get('goals_conceded', 0),
                    'own_goals': player.get('own_goals', 0),
                    'penalties_saved': player.get('penalties_saved', 0),
                    'penalties_missed': player.get('penalties_missed', 0),
                    'yellow_cards': player.get('yellow_cards', 0),
                    'red_cards': player.get('red_cards', 0),
                    'saves': player.get('saves', 0),
                    'bonus': player.get('bonus', 0),
                    'bps': player.get('bps', 0),

                    # Availability and risk factors
                    'minutes': player.get('minutes', 0),
                    'chance_of_playing_this_round': player.get('chance_of_playing_this_round'),
                    'chance_of_playing_next_round': player.get('chance_of_playing_next_round'),
                    'status': player.get('status', 'a'),
                    'injury_news': player.get('news', ''),

                    # Value metrics
                    'value_form': player.get('value_form', 0),
                    'value_season': player.get('value_season', 0),
                    'cost_change_start': player.get('cost_change_start', 0),
                    'cost_change_event': player.get('cost_change_event', 0),

                    # Team context
                    'team_strength_overall_home': team_info.get('strength_overall_home', 3),
                    'team_strength_overall_away': team_info.get('strength_overall_away', 3),
                    'team_strength_attack_home': team_info.get('strength_attack_home', 3),
                    'team_strength_attack_away': team_info.get('strength_attack_away', 3),
                    'team_strength_defence_home': team_info.get('strength_defence_home', 3),
                    'team_strength_defence_away': team_info.get('strength_defence_away', 3),
                }

                enhanced_data.append(player_data)

            df = pd.DataFrame(enhanced_data)

            # Add derived metrics
            df = self._add_derived_metrics(df, current_gw)

            return df

        except Exception as e:
            self.logger.error(f"Failed to get enhanced player data: {e}")
            return pd.DataFrame()

    def _add_derived_metrics(self, df: pd.DataFrame, current_gw: int) -> pd.DataFrame:
        """Add derived metrics for enhanced analysis"""

        # Points per million efficiency
        df['points_per_million'] = pd.to_numeric(df['total_points'], errors='coerce').fillna(0) / pd.to_numeric(df['price'], errors='coerce').fillna(4.0)

        # Form vs season average
        df['form_vs_average'] = pd.to_numeric(df['form'], errors='coerce').fillna(0) - pd.to_numeric(df['points_per_game'], errors='coerce').fillna(0)

        # Availability score (0-1)
        df['availability_score'] = pd.to_numeric(df['chance_of_playing_this_round'], errors='coerce').fillna(100) / 100.0
        df.loc[df['status'] != 'a', 'availability_score'] *= 0.5  # Penalize injured/suspended

        # Consistency score (inverse of variance in recent performances)
        df['consistency_score'] = 1 / (1 + abs(pd.to_numeric(df['form'], errors='coerce').fillna(0) - pd.to_numeric(df['points_per_game'], errors='coerce').fillna(0)))

        # Value momentum (recent price changes)
        df['price_momentum'] = pd.to_numeric(df['cost_change_event'], errors='coerce').fillna(0) + (pd.to_numeric(df['cost_change_start'], errors='coerce').fillna(0) * 0.1)

        # Penalty taker bonus (heuristic based on penalties not missed)
        df['penalty_taker_bonus'] = np.where(
            (df['penalties_missed'] == 0) & (df['position'].isin(['MID', 'FWD'])),
            0.5, 0
        )

        # Set piece taker bonus (heuristic based on assists and position)
        df['set_piece_bonus'] = np.where(
            (df['assists'] >= 2) & (df['position'] == 'MID'),
            0.3, 0
        )

        return df

    def _apply_enhanced_factors(self, df: pd.DataFrame, current_gw: int) -> pd.DataFrame:
        """Apply enhanced factor analysis"""

        # Get fixture difficulty for next gameweek
        df = self._add_fixture_analysis(df, current_gw + 1)

        # Add opponent form analysis
        df = self._add_opponent_analysis(df, current_gw + 1)

        # Add team attacking/defensive form
        df = self._add_team_form_analysis(df)

        # Add momentum analysis
        df = self._add_momentum_analysis(df)

        # Add contrarian value analysis
        df = self._add_contrarian_analysis(df)

        return df

    def _add_fixture_analysis(self, df: pd.DataFrame, next_gw: int) -> pd.DataFrame:
        """Add comprehensive fixture difficulty analysis"""

        try:
            # Get fixtures for next gameweek
            fixtures = self.fpl.fixtures()
            if not fixtures:
                df['fixture_difficulty'] = 3  # Neutral default
                df['is_home'] = True
                return df

            next_fixtures = [f for f in fixtures if f.get('event') == next_gw]

            fixture_map = {}
            venue_map = {}

            for fixture in next_fixtures:
                home_team = fixture.get('team_h')
                away_team = fixture.get('team_a')

                # Home team difficulty (facing away team)
                fixture_map[home_team] = fixture.get('team_a_difficulty', 3)
                venue_map[home_team] = True

                # Away team difficulty (facing home team)
                fixture_map[away_team] = fixture.get('team_h_difficulty', 3)
                venue_map[away_team] = False

            df['fixture_difficulty'] = pd.to_numeric(df['team'].map(fixture_map).fillna(3), errors='coerce').fillna(3)
            df['is_home'] = df['team'].map(venue_map).fillna(True)

            # Adjust difficulty based on venue
            df['venue_adjusted_difficulty'] = df.apply(
                lambda row: float(row['fixture_difficulty']) - 0.3 if row['is_home'] else float(row['fixture_difficulty']) + 0.3,
                axis=1
            )

        except Exception as e:
            self.logger.error(f"Fixture analysis failed: {e}")
            df['fixture_difficulty'] = 3
            df['is_home'] = True
            df['venue_adjusted_difficulty'] = 3

        return df

    def _add_opponent_analysis(self, df: pd.DataFrame, next_gw: int) -> pd.DataFrame:
        """Add opponent defensive/attacking form analysis"""

        # This would require historical team performance data
        # For now, use team strength as proxy
        df['opponent_defensive_strength'] = 5 - df['fixture_difficulty']  # Inverse relationship
        df['opponent_attacking_threat'] = df['fixture_difficulty']

        return df

    def _add_team_form_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add team's recent attacking/defensive form"""

        # Use team strength ratings as base, enhanced with recent performance
        df['team_attacking_form'] = df.apply(
            lambda row: (row['team_strength_attack_home'] if row['is_home'] else row['team_strength_attack_away']) + (row['form'] * 0.2),
            axis=1
        )

        df['team_defensive_form'] = df.apply(
            lambda row: (row['team_strength_defence_home'] if row['is_home'] else row['team_strength_defence_away']) + (row['form'] * 0.1),
            axis=1
        )

        return df

    def _add_momentum_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add performance momentum/trend analysis"""

        # Momentum based on form vs season average
        df['momentum_score'] = np.where(
            df['form_vs_average'] > 1, 1.2,  # Hot streak
            np.where(df['form_vs_average'] < -1, 0.8, 1.0)  # Cold streak or neutral
        )

        # Boost for players in good value form
        value_form_numeric = pd.to_numeric(df['value_form'], errors='coerce').fillna(0)
        df['momentum_score'] *= (1 + (value_form_numeric * 0.1))

        return df

    def _add_contrarian_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add contrarian value analysis for low ownership gems"""

        # Contrarian bonus for good players with low ownership
        df['contrarian_bonus'] = np.where(
            (df['selected_by_percent'] < 5) & (df['points_per_million'] > 2),
            1.15,  # 15% bonus for low-owned, good value players
            1.0
        )

        return df

    def _calculate_enhanced_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate final predictions using weighted factors"""

        # Normalize key factors to 0-1 scale
        factors_to_normalize = ['form', 'points_per_game', 'points_per_million', 'availability_score',
                               'consistency_score', 'team_attacking_form', 'momentum_score']

        for factor in factors_to_normalize:
            if factor in df.columns:
                factor_numeric = pd.to_numeric(df[factor], errors='coerce').fillna(0)
                max_val = factor_numeric.max()
                if max_val > 0:
                    df[f'{factor}_norm'] = factor_numeric / max_val
                else:
                    df[f'{factor}_norm'] = 0

        # Base prediction from weighted factors - ensure all values are numeric
        form_norm = pd.to_numeric(df.get('form_norm', 0), errors='coerce').fillna(0)
        ppg_norm = pd.to_numeric(df.get('points_per_game_norm', 0), errors='coerce').fillna(0)
        venue_diff = pd.to_numeric(df['venue_adjusted_difficulty'], errors='coerce').fillna(3)
        ppm_norm = pd.to_numeric(df.get('points_per_million_norm', 0), errors='coerce').fillna(0)
        avail_norm = pd.to_numeric(df.get('availability_score_norm', 0), errors='coerce').fillna(0)
        penalty_bonus = pd.to_numeric(df.get('penalty_taker_bonus', 0), errors='coerce').fillna(0)
        set_bonus = pd.to_numeric(df.get('set_piece_bonus', 0), errors='coerce').fillna(0)
        attack_norm = pd.to_numeric(df.get('team_attacking_form_norm', 0), errors='coerce').fillna(0)
        momentum_norm = pd.to_numeric(df.get('momentum_score_norm', 0), errors='coerce').fillna(0)

        df['base_prediction'] = (
            form_norm * self.factor_weights['recent_form'] +
            ppg_norm * self.factor_weights['season_average'] +
            (5 - venue_diff) / 5 * self.factor_weights['fixture_difficulty'] +
            ppm_norm * self.factor_weights['price_efficiency'] +
            avail_norm * (-self.factor_weights['injury_rotation_risk']) +
            (penalty_bonus + set_bonus) * self.factor_weights['penalty_set_pieces'] +
            attack_norm * self.factor_weights['team_attacking_form'] +
            momentum_norm * self.factor_weights['momentum_trend']
        ) * 10  # Scale to points

        # Apply contrarian bonus
        df['enhanced_prediction'] = df['base_prediction'] * df.get('contrarian_bonus', 1)

        # Ensure minimum sensible predictions
        df['enhanced_prediction'] = np.maximum(df['enhanced_prediction'], 1.0)

        return df

    def _apply_position_adjustments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply position-specific adjustments"""

        for position, multipliers in self.position_multipliers.items():
            pos_mask = df['position'] == position

            if position == 'GKP':
                # Goalkeepers: boost based on clean sheet potential
                clean_sheet_boost = (5 - df['venue_adjusted_difficulty']) / 5 * multipliers['clean_sheet_weight']
                df.loc[pos_mask, 'enhanced_prediction'] *= (1 + clean_sheet_boost * 0.3)

            elif position == 'DEF':
                # Defenders: clean sheets + attacking threat
                clean_sheet_boost = (5 - df['venue_adjusted_difficulty']) / 5 * multipliers['clean_sheet_weight']
                assists_numeric = pd.to_numeric(df['assists'], errors='coerce').fillna(0)
                max_assists = assists_numeric.max() if assists_numeric.max() > 0 else 1
                attacking_boost = assists_numeric / max_assists * multipliers['attacking_bonus']
                df.loc[pos_mask, 'enhanced_prediction'] *= (1 + (clean_sheet_boost + attacking_boost) * 0.2)

            elif position == 'MID':
                # Midfielders: assists and goals
                assists_numeric = pd.to_numeric(df.loc[pos_mask, 'assists'], errors='coerce').fillna(0)
                df.loc[pos_mask, 'enhanced_prediction'] *= (1 + assists_numeric * 0.1)

            elif position == 'FWD':
                # Forwards: goal scoring focus
                goals_numeric = pd.to_numeric(df.loc[pos_mask, 'goals_scored'], errors='coerce').fillna(0)
                df.loc[pos_mask, 'enhanced_prediction'] *= (1 + goals_numeric * 0.15)

        return df

    def _add_confidence_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add prediction confidence scores"""

        # Confidence based on data quality and consistency
        df['prediction_confidence'] = (
            df.get('consistency_score', 0.5) * 0.4 +
            df.get('availability_score', 0.5) * 0.3 +
            (df['minutes'] / df['minutes'].max()) * 0.3
        )

        # Boost confidence for reliable players
        df.loc[df['minutes'] > 1000, 'prediction_confidence'] *= 1.2
        df.loc[df['status'] != 'a', 'prediction_confidence'] *= 0.6

        return df

    def _get_position_name(self, element_type: int) -> str:
        """Convert element type to position name"""
        position_map = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        return position_map.get(element_type, 'Unknown')

    def get_factor_importance(self) -> Dict[str, float]:
        """Return current factor weights for transparency"""
        return self.factor_weights.copy()

    def update_factor_weights(self, weight_adjustments: Dict[str, float]) -> None:
        """Update factor weights based on feedback"""
        for factor, adjustment in weight_adjustments.items():
            if factor in self.factor_weights:
                self.factor_weights[factor] += adjustment
                self.logger.info(f"Updated {factor} weight by {adjustment:+.3f}")

        # Normalize weights to ensure they sum to approximately 1.0
        total_weight = sum(abs(w) for w in self.factor_weights.values())
        if total_weight > 0:
            for factor in self.factor_weights:
                self.factor_weights[factor] /= total_weight