from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging


class PositionSpecificFeatureEngineer:
    """Advanced feature engineering tailored to each position"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Position-specific feature weights and importance
        self.position_features = {
            'GKP': {
                'primary': ['saves', 'clean_sheets', 'bonus_from_saves', 'penalties_saved'],
                'secondary': ['goals_conceded', 'yellow_cards', 'minutes'],
                'derived': ['save_percentage', 'cs_probability', 'penalty_save_rate']
            },
            'DEF': {
                'primary': ['clean_sheets', 'goals_scored', 'assists', 'bonus', 'aerial_duels_won'],
                'secondary': ['tackles', 'interceptions', 'clearances', 'blocks', 'key_passes'],
                'derived': ['attacking_threat', 'defensive_solidity', 'set_piece_threat']
            },
            'MID': {
                'primary': ['goals_scored', 'assists', 'bonus', 'key_passes', 'chances_created'],
                'secondary': ['passes_completed', 'tackles', 'interceptions', 'shots'],
                'derived': ['creativity_index', 'box_to_box_rating', 'final_third_entries']
            },
            'FWD': {
                'primary': ['goals_scored', 'assists', 'bonus', 'shots_on_target', 'big_chances'],
                'secondary': ['shots', 'penalties_won', 'touches_in_box', 'key_passes'],
                'derived': ['clinical_rating', 'penalty_threat', 'big_chance_conversion']
            }
        }
        
        # Form calculation weights (recent games weighted more heavily)
        self.form_weights = np.array([0.4, 0.3, 0.2, 0.1])  # Last 4 games
        
    def engineer_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create position-specific enhanced features"""
        try:
            if df.empty:
                return df
            
            enhanced_df = df.copy()
            
            # Add basic position-specific features for each position
            for position in ['GKP', 'DEF', 'MID', 'FWD']:
                position_mask = enhanced_df['position'] == position
                if position_mask.sum() == 0:
                    continue
                
                position_data = enhanced_df[position_mask].copy()
                
                # Generate position-specific features
                if position == 'GKP':
                    enhanced_df.loc[position_mask, :] = self._engineer_goalkeeper_features(position_data)
                elif position == 'DEF':
                    enhanced_df.loc[position_mask, :] = self._engineer_defender_features(position_data)
                elif position == 'MID':
                    enhanced_df.loc[position_mask, :] = self._engineer_midfielder_features(position_data)
                elif position == 'FWD':
                    enhanced_df.loc[position_mask, :] = self._engineer_forward_features(position_data)
            
            # Add cross-position comparative features
            enhanced_df = self._add_comparative_features(enhanced_df)
            
            # Add role-based sub-position features
            enhanced_df = self._add_role_classification(enhanced_df)
            
            # Add team context features
            enhanced_df = self._add_team_context_features(enhanced_df)
            
            self.logger.info(f"Position-specific feature engineering complete: {len(enhanced_df)} players")
            return enhanced_df
            
        except Exception as e:
            self.logger.error(f"Position-specific feature engineering failed: {e}")
            return df
    
    def _engineer_goalkeeper_features(self, gkp_df: pd.DataFrame) -> pd.DataFrame:
        """Engineer goalkeeper-specific features"""
        df = gkp_df.copy()
        
        try:
            # Save-related features
            df['saves_per_game'] = df.get('saves', 0) / np.maximum(df.get('games_played', 1), 1)
            df['save_percentage'] = df.get('saves', 0) / np.maximum(df.get('shots_faced', 1), 1)
            df['saves_bonus_rate'] = df.get('bonus_from_saves', 0) / np.maximum(df.get('saves', 1), 1)
            
            # Clean sheet features
            df['clean_sheet_rate'] = df.get('clean_sheets', 0) / np.maximum(df.get('games_played', 1), 1)
            df['goals_conceded_rate'] = df.get('goals_conceded', 0) / np.maximum(df.get('games_played', 1), 1)
            
            # Penalty features
            df['penalty_save_rate'] = df.get('penalties_saved', 0) / np.maximum(df.get('penalties_faced', 1), 1)
            
            # Composite goalkeeper rating
            save_rating = np.clip(df['saves_per_game'] / 5, 0, 1)  # Normalize to 0-1
            cs_rating = np.clip(df['clean_sheet_rate'], 0, 1)
            bonus_rating = np.clip(df.get('bonus', 0) / 10, 0, 1)
            
            df['gkp_composite_rating'] = (save_rating * 0.4 + cs_rating * 0.4 + bonus_rating * 0.2)
            
            # Recent form (weighted)
            if 'recent_points' in df.columns:
                df['gkp_weighted_form'] = self._calculate_weighted_form(df, 'recent_points')
            
        except Exception as e:
            self.logger.error(f"Goalkeeper feature engineering failed: {e}")
        
        return df
    
    def _engineer_defender_features(self, def_df: pd.DataFrame) -> pd.DataFrame:
        """Engineer defender-specific features"""
        df = def_df.copy()
        
        try:
            # Defensive solidity features
            df['clean_sheet_rate'] = df.get('clean_sheets', 0) / np.maximum(df.get('games_played', 1), 1)
            df['defensive_actions'] = (
                df.get('tackles', 0) + df.get('interceptions', 0) + 
                df.get('clearances', 0) + df.get('blocks', 0)
            )
            df['def_actions_per_game'] = df['defensive_actions'] / np.maximum(df.get('games_played', 1), 1)
            
            # Attacking threat for defenders
            df['attacking_returns'] = df.get('goals_scored', 0) + df.get('assists', 0)
            df['att_returns_rate'] = df['attacking_returns'] / np.maximum(df.get('games_played', 1), 1)
            df['shots_per_game'] = df.get('shots', 0) / np.maximum(df.get('games_played', 1), 1)
            
            # Aerial ability
            df['aerial_win_rate'] = df.get('aerial_duels_won', 0) / np.maximum(df.get('aerial_duels', 1), 1)
            
            # Set piece threat (proxy using headed goals + free kick attempts)
            df['set_piece_threat'] = (
                df.get('headed_goals', 0) + df.get('free_kick_attempts', 0) * 0.1
            )
            
            # Composite defender rating
            defensive_rating = np.clip(df['clean_sheet_rate'] + df['def_actions_per_game'] / 10, 0, 1)
            attacking_rating = np.clip(df['att_returns_rate'] * 5, 0, 1)
            
            df['def_composite_rating'] = (defensive_rating * 0.6 + attacking_rating * 0.4)
            
            # Bonus point efficiency
            df['bonus_efficiency'] = df.get('bonus', 0) / np.maximum(df.get('games_played', 1), 1)
            
        except Exception as e:
            self.logger.error(f"Defender feature engineering failed: {e}")
        
        return df
    
    def _engineer_midfielder_features(self, mid_df: pd.DataFrame) -> pd.DataFrame:
        """Engineer midfielder-specific features"""
        df = mid_df.copy()
        
        try:
            # Creativity and playmaking
            df['creativity_index'] = (
                df.get('key_passes', 0) * 0.4 + 
                df.get('assists', 0) * 0.4 + 
                df.get('chances_created', 0) * 0.2
            )
            df['creativity_per_game'] = df['creativity_index'] / np.maximum(df.get('games_played', 1), 1)
            
            # Goal threat
            df['goal_threat'] = (
                df.get('shots_on_target', 0) * 0.5 + 
                df.get('shots_in_box', 0) * 0.3 + 
                df.get('big_chances', 0) * 0.2
            )
            df['goal_threat_per_game'] = df['goal_threat'] / np.maximum(df.get('games_played', 1), 1)
            
            # Box-to-box rating (attacking + defensive contributions)
            attacking_contrib = df.get('goals_scored', 0) + df.get('assists', 0)
            defensive_contrib = df.get('tackles', 0) + df.get('interceptions', 0)
            df['box_to_box_rating'] = (attacking_contrib + defensive_contrib * 0.5) / np.maximum(df.get('games_played', 1), 1)
            
            # Pass completion and tempo
            df['pass_completion_rate'] = df.get('passes_completed', 0) / np.maximum(df.get('passes_attempted', 1), 1)
            df['progressive_passes_rate'] = df.get('progressive_passes', 0) / np.maximum(df.get('passes_completed', 1), 1)
            
            # Final third impact
            df['final_third_entries'] = df.get('touches_att_pen_area', 0) + df.get('passes_final_third', 0)
            
            # Composite midfielder rating
            creative_rating = np.clip(df['creativity_per_game'] / 3, 0, 1)
            goal_rating = np.clip(df['goal_threat_per_game'] / 2, 0, 1)
            overall_contrib = np.clip(df['box_to_box_rating'] / 2, 0, 1)
            
            df['mid_composite_rating'] = (creative_rating * 0.4 + goal_rating * 0.4 + overall_contrib * 0.2)
            
        except Exception as e:
            self.logger.error(f"Midfielder feature engineering failed: {e}")
        
        return df
    
    def _engineer_forward_features(self, fwd_df: pd.DataFrame) -> pd.DataFrame:
        """Engineer forward-specific features"""
        df = fwd_df.copy()
        
        try:
            # Clinical finishing
            df['shot_conversion_rate'] = df.get('goals_scored', 0) / np.maximum(df.get('shots', 1), 1)
            df['big_chance_conversion'] = df.get('goals_scored', 0) / np.maximum(df.get('big_chances', 1), 1)
            df['shots_on_target_rate'] = df.get('shots_on_target', 0) / np.maximum(df.get('shots', 1), 1)
            
            # Box presence and positioning
            df['box_touches_per_game'] = df.get('touches_in_box', 0) / np.maximum(df.get('games_played', 1), 1)
            df['shots_per_game'] = df.get('shots', 0) / np.maximum(df.get('games_played', 1), 1)
            
            # Penalty involvement
            df['penalties_won_rate'] = df.get('penalties_won', 0) / np.maximum(df.get('games_played', 1), 1)
            df['penalty_taker_prob'] = df.get('penalties_taken', 0) / np.maximum(df.get('team_penalties', 1), 1)
            
            # Assist contribution
            df['assist_rate'] = df.get('assists', 0) / np.maximum(df.get('games_played', 1), 1)
            df['key_pass_rate'] = df.get('key_passes', 0) / np.maximum(df.get('games_played', 1), 1)
            
            # Composite forward rating
            clinical_rating = np.clip((df['shot_conversion_rate'] + df['big_chance_conversion']) / 2, 0, 1)
            positioning_rating = np.clip(df['box_touches_per_game'] / 5, 0, 1)
            overall_threat = np.clip((df.get('goals_scored', 0) + df.get('assists', 0)) / np.maximum(df.get('games_played', 1), 1), 0, 1)
            
            df['fwd_composite_rating'] = (clinical_rating * 0.4 + positioning_rating * 0.3 + overall_threat * 0.3)
            
            # Expected goals efficiency
            if 'expected_goals' in df.columns:
                df['xg_overperformance'] = (df.get('goals_scored', 0) - df.get('expected_goals', 0)) / np.maximum(df.get('games_played', 1), 1)
            
        except Exception as e:
            self.logger.error(f"Forward feature engineering failed: {e}")
        
        return df
    
    def _add_comparative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features comparing players within their position"""
        try:
            for position in ['GKP', 'DEF', 'MID', 'FWD']:
                position_mask = df['position'] == position
                if position_mask.sum() <= 1:
                    continue
                
                position_data = df[position_mask]
                
                # Percentile rankings within position
                key_stats = ['goals_scored', 'assists', 'bonus', 'minutes']
                for stat in key_stats:
                    if stat in df.columns:
                        percentiles = position_data[stat].rank(pct=True)
                        df.loc[position_mask, f'{stat}_position_percentile'] = percentiles
            
            return df
            
        except Exception as e:
            self.logger.error(f"Comparative feature engineering failed: {e}")
            return df
    
    def _add_role_classification(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify players into sub-roles within their position"""
        try:
            df['player_role'] = df['position']  # Default to main position
            
            # Defender sub-roles
            def_mask = df['position'] == 'DEF'
            if def_mask.sum() > 0:
                # Wing-backs (high attacking returns)
                wb_mask = (df['att_returns_rate'] > df[def_mask]['att_returns_rate'].quantile(0.7)) & def_mask
                df.loc[wb_mask, 'player_role'] = 'DEF_WB'
                
                # Center-backs (low attacking returns, high defensive actions)
                cb_mask = (df['att_returns_rate'] <= df[def_mask]['att_returns_rate'].quantile(0.3)) & def_mask
                df.loc[cb_mask, 'player_role'] = 'DEF_CB'
            
            # Midfielder sub-roles  
            mid_mask = df['position'] == 'MID'
            if mid_mask.sum() > 0:
                mid_data = df[mid_mask]
                
                # Attacking midfielders (high creativity + goals)
                am_mask = (
                    (df['creativity_per_game'] > mid_data['creativity_per_game'].quantile(0.6)) &
                    (df['goal_threat_per_game'] > mid_data['goal_threat_per_game'].quantile(0.6)) & 
                    mid_mask
                )
                df.loc[am_mask, 'player_role'] = 'MID_AM'
                
                # Defensive midfielders (high defensive actions, low attacking)
                dm_mask = (
                    (df['box_to_box_rating'] < mid_data['box_to_box_rating'].quantile(0.4)) & 
                    mid_mask
                )
                df.loc[dm_mask, 'player_role'] = 'MID_DM'
            
            # Forward sub-roles
            fwd_mask = df['position'] == 'FWD' 
            if fwd_mask.sum() > 0:
                # Wide forwards vs Central strikers based on assists/key passes
                wf_mask = (df['assist_rate'] > df[fwd_mask]['assist_rate'].quantile(0.6)) & fwd_mask
                df.loc[wf_mask, 'player_role'] = 'FWD_W'
                
                cf_mask = (df['box_touches_per_game'] > df[fwd_mask]['box_touches_per_game'].quantile(0.6)) & fwd_mask
                df.loc[cf_mask, 'player_role'] = 'FWD_C'
            
            return df
            
        except Exception as e:
            self.logger.error(f"Role classification failed: {e}")
            return df
    
    def _add_team_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add team-level context features"""
        try:
            # Team attacking style (goals scored, possession-based vs direct)
            team_stats = df.groupby('team_id').agg({
                'goals_scored': 'sum',
                'assists': 'sum', 
                'shots': 'sum',
                'minutes': 'sum'
            }).reset_index()
            
            team_stats['team_attacking_rate'] = team_stats['goals_scored'] / np.maximum(team_stats['minutes'] / 90, 1)
            team_stats['team_creativity_rate'] = team_stats['assists'] / np.maximum(team_stats['minutes'] / 90, 1)
            
            # Merge back to main dataframe
            df = df.merge(
                team_stats[['team_id', 'team_attacking_rate', 'team_creativity_rate']], 
                on='team_id', 
                how='left'
            )
            
            # Player's share of team output
            df['goal_share'] = df['goals_scored'] / np.maximum(df['team_attacking_rate'] * df['minutes'] / 90, 1)
            df['assist_share'] = df['assists'] / np.maximum(df['team_creativity_rate'] * df['minutes'] / 90, 1)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Team context features failed: {e}")
            return df
    
    def _calculate_weighted_form(self, df: pd.DataFrame, points_col: str) -> pd.Series:
        """Calculate exponentially weighted recent form"""
        try:
            # Assume recent_points is a list of last 4 games
            weighted_form = []
            
            for idx, row in df.iterrows():
                if points_col in row and isinstance(row[points_col], (list, tuple)):
                    recent_points = row[points_col][-4:]  # Last 4 games
                    
                    # Pad with zeros if less than 4 games
                    while len(recent_points) < 4:
                        recent_points = [0] + recent_points
                    
                    # Calculate weighted average
                    weighted_avg = np.sum(np.array(recent_points) * self.form_weights)
                    weighted_form.append(weighted_avg)
                else:
                    # Fallback to total points if no recent data
                    weighted_form.append(row.get('total_points', 0) / np.maximum(row.get('games_played', 1), 1))
            
            return pd.Series(weighted_form, index=df.index)
            
        except Exception as e:
            self.logger.error(f"Weighted form calculation failed: {e}")
            return pd.Series([0] * len(df), index=df.index)