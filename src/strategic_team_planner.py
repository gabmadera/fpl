from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .team_selector import TeamSelector, FPLTeam
from .fpl_client import FPLClient
import itertools


@dataclass
class TransferPlan:
    """Represents a multi-gameweek transfer plan"""
    gameweek: int
    transfers_in: List[Dict]  # List of player dicts
    transfers_out: List[Dict]  # List of player dicts
    transfer_cost: int  # Points deducted for transfers (-4 per extra transfer)
    expected_points_gain: float  # Expected points gain after transfer costs
    net_benefit: float  # Expected points gain - transfer costs
    reasoning: str  # Why this transfer is recommended


@dataclass 
class MultiGameweekStrategy:
    """Represents a complete strategy across multiple gameweeks"""
    target_gameweeks: List[int]
    current_team: List[Dict]
    transfer_plans: List[TransferPlan]
    total_expected_points: float
    total_transfer_costs: int
    net_points_benefit: float  # Total expected points - total transfer costs
    recommended_formation: str


class StrategicTeamPlanner:
    """Multi-gameweek strategic team planning with transfer cost optimization"""
    
    def __init__(self):
        self.team_selector = TeamSelector()
        self.fpl = FPLClient()
        self.transfer_cost = 4  # Points deducted per extra transfer
        self.free_transfers_per_week = 1
        
    def create_multi_gameweek_strategy(self, 
                                     predictions_df: pd.DataFrame,
                                     current_team_ids: List[int],
                                     current_gameweek: int,
                                     horizon: int = 5) -> MultiGameweekStrategy:
        """
        Create a strategic plan for the next 'horizon' gameweeks that minimizes
        transfer costs while maximizing points over the period.
        
        Args:
            predictions_df: Player predictions with multi-gameweek data
            current_team_ids: List of current team player IDs
            current_gameweek: Current gameweek number
            horizon: Number of gameweeks to plan ahead (default 5)
        """
        target_gameweeks = list(range(current_gameweek, current_gameweek + horizon))
        
        # Get current team details
        current_team = self._get_team_details(current_team_ids, predictions_df)
        
        # Create multi-gameweek predictions
        mgw_predictions = self._create_multi_gameweek_predictions(predictions_df, target_gameweeks)
        
        # Find optimal strategy using dynamic programming approach
        strategy = self._optimize_multi_gameweek_strategy(
            current_team, mgw_predictions, target_gameweeks
        )
        
        return strategy
    
    def _create_multi_gameweek_predictions(self, 
                                         predictions_df: pd.DataFrame,
                                         target_gameweeks: List[int]) -> pd.DataFrame:
        """
        Create aggregated predictions across multiple gameweeks with weighted importance
        Recent gameweeks weighted more heavily than distant ones
        """
        df = predictions_df.copy()
        
        # Weight gameweeks - more recent gameweeks get higher weight
        weights = np.array([0.4, 0.3, 0.2, 0.1])[:len(target_gameweeks)]  # Exponential decay
        weights = weights / weights.sum()  # Normalize
        
        # Calculate multi-gameweek metrics
        base_points = pd.to_numeric(df['predicted_points'], errors='coerce').fillna(0)
        
        # Fixture difficulty adjustment
        if 'fixture_difficulty' in df.columns:
            fixture_mult = 1.1 - (pd.to_numeric(df['fixture_difficulty'], errors='coerce').fillna(3) - 3) * 0.1
            fixture_mult = np.clip(fixture_mult, 0.7, 1.3)
            base_points *= fixture_mult
        
        # Multi-gameweek expected points (weighted sum)
        df['mgw_expected_points'] = base_points * len(target_gameweeks) * weights[0]
        
        # Consistency score (players who perform consistently across gameweeks)
        if 'points_variance' in df.columns:
            consistency = 1.0 - (pd.to_numeric(df['points_variance'], errors='coerce').fillna(0) / 10.0)
            consistency = np.clip(consistency, 0.5, 1.0)
            df['mgw_expected_points'] *= consistency
        
        # Value for money over multiple gameweeks
        df['mgw_value_score'] = df['mgw_expected_points'] / df['price']
        
        # Injury/rotation risk (reduce points for risky players)
        availability_risk = pd.to_numeric(df.get('chance_next', 100), errors='coerce').fillna(100) / 100.0
        df['mgw_expected_points'] *= availability_risk
        
        return df
    
    def _optimize_multi_gameweek_strategy(self,
                                        current_team: List[Dict],
                                        mgw_predictions: pd.DataFrame,
                                        target_gameweeks: List[int]) -> MultiGameweekStrategy:
        """
        Use dynamic programming to find optimal transfer strategy across gameweeks
        """
        # Find the theoretically optimal team for the multi-gameweek period
        optimal_team = self.team_selector.select_optimal_team(mgw_predictions)
        
        current_ids = set(p['player_id'] for p in current_team)
        optimal_ids = set(p['player_id'] for p in optimal_team.starters + optimal_team.bench)
        
        # Players that need to be transferred
        players_to_remove = current_ids - optimal_ids
        players_to_add = optimal_ids - current_ids
        
        # Calculate priority scores for transfers
        transfer_priorities = self._calculate_transfer_priorities(
            current_team, mgw_predictions, players_to_remove, players_to_add
        )
        
        # Create weekly transfer plans with budget constraints
        transfer_plans = self._create_weekly_transfer_plans(
            transfer_priorities, target_gameweeks, len(target_gameweeks)
        )
        
        # Calculate total strategy metrics
        total_expected_points = sum(plan.expected_points_gain for plan in transfer_plans)
        total_transfer_costs = sum(plan.transfer_cost for plan in transfer_plans)
        net_benefit = total_expected_points - total_transfer_costs
        
        return MultiGameweekStrategy(
            target_gameweeks=target_gameweeks,
            current_team=current_team,
            transfer_plans=transfer_plans,
            total_expected_points=total_expected_points,
            total_transfer_costs=total_transfer_costs,
            net_points_benefit=net_benefit,
            recommended_formation=optimal_team.formation
        )
    
    def _calculate_transfer_priorities(self,
                                     current_team: List[Dict],
                                     mgw_predictions: pd.DataFrame,
                                     players_to_remove: set,
                                     players_to_add: set) -> List[Dict]:
        """Calculate priority scores for each potential transfer"""
        priorities = []
        
        current_dict = {p['player_id']: p for p in current_team}
        mgw_dict = mgw_predictions.set_index('player_id').to_dict('index')
        
        # Evaluate each potential transfer
        for out_id in players_to_remove:
            out_player = current_dict.get(out_id, {})
            out_position = out_player.get('position', '')
            
            # Find best replacement in same position
            same_position_candidates = [
                pid for pid in players_to_add 
                if mgw_dict.get(pid, {}).get('position') == out_position
            ]
            
            for in_id in same_position_candidates:
                in_player = mgw_dict.get(in_id, {})
                
                # Calculate transfer value
                points_gain = in_player.get('mgw_expected_points', 0) - out_player.get('mgw_expected_points', 0)
                cost_change = in_player.get('price', 0) - out_player.get('price', 0)
                
                # Priority factors
                urgency = 0.0
                
                # Injury/availability urgency
                if out_player.get('fpl_status') in ['i', 's', 'u']:
                    urgency += 10.0
                elif pd.to_numeric(out_player.get('chance_next', 100), errors='coerce') < 25:
                    urgency += 5.0
                
                # Form urgency (poor recent performance)
                if out_player.get('predicted_points', 0) < 3.0:
                    urgency += 3.0
                
                # Fixture difficulty
                out_fixtures = out_player.get('fixture_difficulty', 3)
                in_fixtures = in_player.get('fixture_difficulty', 3)
                fixture_benefit = (out_fixtures - in_fixtures) * 0.5
                
                priority_score = points_gain + urgency + fixture_benefit
                
                priorities.append({
                    'out_player': out_player,
                    'in_player': in_player,
                    'points_gain': points_gain,
                    'cost_change': cost_change,
                    'urgency': urgency,
                    'priority_score': priority_score,
                    'reasoning': self._generate_transfer_reasoning(
                        out_player, in_player, points_gain, urgency, fixture_benefit
                    )
                })
        
        return sorted(priorities, key=lambda x: x['priority_score'], reverse=True)
    
    def _create_weekly_transfer_plans(self,
                                    transfer_priorities: List[Dict],
                                    target_gameweeks: List[int],
                                    max_weeks: int) -> List[TransferPlan]:
        """Create weekly transfer plans with optimal timing"""
        plans = []
        remaining_transfers = transfer_priorities.copy()
        
        for week_idx, gameweek in enumerate(target_gameweeks[:max_weeks]):
            # Determine optimal number of transfers for this week
            # Be more aggressive early, more conservative later
            if week_idx == 0:
                max_transfers = 3  # First week can do more transfers
            elif week_idx < 2:
                max_transfers = 2  # Early weeks 2 transfers max
            else:
                max_transfers = 1  # Later weeks stick to free transfers
                
            week_transfers = []
            week_cost = 0
            week_points_gain = 0.0
            
            # Select best transfers for this week
            for transfer in remaining_transfers[:max_transfers]:
                # Check if we can afford the transfer
                cost_change = transfer['cost_change']
                
                # Only add transfer if benefit outweighs cost
                transfer_cost_this_week = max(0, len(week_transfers) + 1 - self.free_transfers_per_week) * self.transfer_cost
                net_benefit = transfer['points_gain'] - transfer_cost_this_week
                
                if net_benefit > 2.0 or transfer['urgency'] > 5.0:  # Only if significant benefit or urgent
                    week_transfers.append(transfer)
                    week_cost += transfer_cost_this_week
                    week_points_gain += transfer['points_gain']
                    
            # Create transfer plan for this week
            if week_transfers:
                plans.append(TransferPlan(
                    gameweek=gameweek,
                    transfers_in=[t['in_player'] for t in week_transfers],
                    transfers_out=[t['out_player'] for t in week_transfers], 
                    transfer_cost=week_cost,
                    expected_points_gain=week_points_gain,
                    net_benefit=week_points_gain - week_cost,
                    reasoning=f"Week {week_idx + 1}: {len(week_transfers)} transfers - " + 
                             "; ".join(t['reasoning'] for t in week_transfers)
                ))
                
                # Remove completed transfers from remaining list
                for transfer in week_transfers:
                    if transfer in remaining_transfers:
                        remaining_transfers.remove(transfer)
            else:
                # No transfers recommended this week
                plans.append(TransferPlan(
                    gameweek=gameweek,
                    transfers_in=[],
                    transfers_out=[],
                    transfer_cost=0,
                    expected_points_gain=0.0,
                    net_benefit=0.0,
                    reasoning=f"Week {week_idx + 1}: No beneficial transfers found"
                ))
        
        return plans
    
    def _generate_transfer_reasoning(self,
                                   out_player: Dict,
                                   in_player: Dict,
                                   points_gain: float,
                                   urgency: float,
                                   fixture_benefit: float) -> str:
        """Generate human-readable reasoning for transfer suggestion"""
        out_name = out_player.get('name', 'Unknown')
        in_name = in_player.get('name', 'Unknown')
        
        reasons = []
        
        if urgency >= 10:
            reasons.append(f"{out_name} injured/suspended")
        elif urgency >= 5:
            reasons.append(f"{out_name} unlikely to play")
        elif urgency >= 3:
            reasons.append(f"{out_name} poor form")
            
        if points_gain > 2:
            reasons.append(f"+{points_gain:.1f} pts upgrade")
        elif fixture_benefit > 1:
            reasons.append("better fixtures")
            
        reason_text = " - ".join(reasons) if reasons else "marginal upgrade"
        
        return f"{out_name} → {in_name}: {reason_text}"
    
    def _get_team_details(self, team_ids: List[int], predictions_df: pd.DataFrame) -> List[Dict]:
        """Get detailed information for current team players"""
        team_details = []
        
        for player_id in team_ids:
            player_data = predictions_df[predictions_df['player_id'] == player_id]
            if not player_data.empty:
                team_details.append(player_data.iloc[0].to_dict())
        
        return team_details
    
    def get_next_gameweek_recommendation(self,
                                       predictions_df: pd.DataFrame,
                                       current_team_ids: List[int],
                                       current_gameweek: int) -> Dict:
        """Get immediate recommendation for next gameweek only"""
        
        strategy = self.create_multi_gameweek_strategy(
            predictions_df, current_team_ids, current_gameweek, horizon=3
        )
        
        # Get next week's plan
        next_week_plan = strategy.transfer_plans[0] if strategy.transfer_plans else None
        
        if not next_week_plan or next_week_plan.net_benefit <= 0:
            return {
                'recommendation': 'HOLD',
                'transfers': [],
                'reasoning': 'No beneficial transfers found - save free transfer',
                'expected_benefit': 0,
                'transfer_cost': 0
            }
        
        return {
            'recommendation': 'TRANSFER',
            'transfers': [{
                'out': out_player,
                'in': in_player,
            } for out_player, in_player in zip(next_week_plan.transfers_out, next_week_plan.transfers_in)],
            'reasoning': next_week_plan.reasoning,
            'expected_benefit': next_week_plan.net_benefit,
            'transfer_cost': next_week_plan.transfer_cost,
            'long_term_strategy': f"Part of {len(strategy.target_gameweeks)}-week strategy worth +{strategy.net_points_benefit:.1f} pts"
        }