from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .fpl_client import FPLClient


@dataclass
class FPLTeam:
    """Represents an FPL team selection"""
    starters: List[Dict]  # 11 players
    bench: List[Dict]    # 4 players  
    captain_id: int
    vice_captain_id: int
    total_cost: float
    predicted_points: float
    formation: str       # e.g., "3-4-3", "4-3-3"
    chip_recommendation: Optional[str] = None


class TeamSelector:
    """Intelligent team selection with formation optimization and chip strategy"""
    
    def __init__(self):
        self.fpl = FPLClient()
        self.budget = 100.0  # £100m budget
        # All valid FPL formations (min 3 DEF, min 1 FWD)
        self.formations = [
            "3-4-3", "3-5-2", "4-3-3", "4-4-2", "4-5-1", "5-3-2", "5-4-1"
        ]
        # Fixed squad composition per FPL rules
        self.squad_requirements = {
            'GKP': 2,  # Exactly 2 goalkeepers
            'DEF': 5,  # Exactly 5 defenders
            'MID': 5,  # Exactly 5 midfielders  
            'FWD': 3   # Exactly 3 forwards
        }
    
    def select_optimal_team(self, predictions_df: pd.DataFrame, 
                          gameweek: int = 1, 
                          existing_team: Optional[List[int]] = None) -> FPLTeam:
        """Select the optimal 15-player team using formation optimization"""
        
        # Clean and prepare data
        df = predictions_df.copy()
        df = self._prepare_data(df)
        
        best_team = None
        best_score = -1
        
        # Try each formation using the improved selection method
        for formation in self.formations:
            try:
                team = self._build_team_with_formation(df, formation)
                if team and team.predicted_points > best_score:
                    best_team = team
                    best_score = team.predicted_points
                    print(f"Found better team with {formation}: {team.predicted_points:.1f} points")
            except Exception as e:
                print(f"Failed to build team for {formation}: {e}")
                continue
        
        if not best_team:
            # Use improved fallback selection
            print("Using improved team selection algorithm")
            best_team = self._fallback_team_selection(df)
        else:
            print(f"Optimal team uses {best_team.formation} formation with {best_team.predicted_points:.1f} points")
        
        # Add chip recommendation
        best_team.chip_recommendation = self._recommend_chip(best_team, gameweek, df)
        
        return best_team
    
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare prediction data for team selection"""
        # Ensure required columns exist with defaults
        required_cols = {
            'predicted_points': 0.0,
            'price': 5.0,
            'position': 'MID',
            'fpl_status': 'a',
            'chance_next': 100.0,
            'selected_by_percent': 5.0,
            'team_id': 1
        }
        
        for col, default in required_cols.items():
            if col not in df.columns:
                df[col] = default
            else:
                df[col] = df[col].fillna(default)
        
        # Filter available players (exclude long-term injuries)
        df = df[~((df['fpl_status'].isin(['i', 's'])) & (df['chance_next'] < 5))].copy()
        
        # Calculate value metrics
        df['value_score'] = df['predicted_points'] / df['price']
        df['adjusted_points'] = df['predicted_points'] * (df['chance_next'] / 100.0)
        
        return df.sort_values('adjusted_points', ascending=False)
    
    def _build_team_with_formation(self, df: pd.DataFrame, formation: str) -> Optional[FPLTeam]:
        """Build team using improved algorithm with specific formation"""
        
        # Parse formation
        def_count, mid_count, fwd_count = map(int, formation.split('-'))
        
        # Validate formation
        if def_count < 3 or fwd_count < 1 or (1 + def_count + mid_count + fwd_count) != 11:
            return None
            
        # Use the same improved selection logic as fallback but with specific formation
        selected = []
        budget = self.budget
        
        # Smart budget allocation for this formation
        # Allow more flexibility for premium players by increasing budgets
        position_budgets = {
            'GKP': min(15.0, budget * 0.15),    # Increased for premium GKs
            'DEF': min(40.0, budget * 0.35),    # Increased for premium DEFs
            'MID': min(50.0, budget * 0.45),    # Keep high for premium MIDs  
            'FWD': min(40.0, budget * 0.35)     # Increased significantly for premium FWDs
        }
        
        # Select squad (always 2-5-5-3) with premium player priority
        for position, count in self.squad_requirements.items():
            pos_budget = position_budgets[position]
            pos_df = df[df['position'] == position].copy()
            
            # Two-pass selection: premium players first, then value players
            premium_df = pos_df[pos_df['price'] >= 10.0].sort_values('adjusted_points', ascending=False)
            value_df = pos_df[pos_df['price'] < 10.0].sort_values('adjusted_points', ascending=False)
            
            selected_count = 0
            position_spent = 0
            
            # Pass 1: Try to select premium players first (if affordable)
            for _, player in premium_df.iterrows():
                if selected_count >= count:
                    break
                    
                # Allow premium players to exceed position budget if we have overall budget
                if player['price'] > budget:
                    continue
                
                team_count = sum(1 for p in selected if p['team_id'] == player['team_id'])
                if team_count >= 3:
                    continue
                
                # For premium players, be more flexible with position budget
                if position_spent + player['price'] <= pos_budget * 1.5 or selected_count == 0:
                    selected.append(player.to_dict())
                    budget -= player['price']
                    position_spent += player['price']
                    selected_count += 1
            
            # Pass 2: Fill remaining slots with value players
            for _, player in value_df.iterrows():
                if selected_count >= count:
                    break
                
                if position_spent + player['price'] > pos_budget and selected_count < count - 1:
                    continue
                    
                if player['price'] > budget:
                    continue
                
                team_count = sum(1 for p in selected if p['team_id'] == player['team_id'])
                if team_count >= 3:
                    continue
                
                selected.append(player.to_dict())
                budget -= player['price']
                position_spent += player['price']
                selected_count += 1
            
            # Fill remaining slots with cheaper players if needed
            if selected_count < count:
                pos_df_cheap = pos_df.sort_values('price')
                for _, player in pos_df_cheap.iterrows():
                    if selected_count >= count:
                        break
                    
                    if player['price'] > budget:
                        continue
                        
                    if any(p['player_id'] == player['player_id'] for p in selected):
                        continue
                    
                    team_count = sum(1 for p in selected if p['team_id'] == player['team_id'])
                    if team_count >= 3:
                        continue
                    
                    selected.append(player.to_dict())
                    budget -= player['price']
                    selected_count += 1
        
        if len(selected) != 15:
            return None
            
        # Assign starters based on specific formation
        starting_requirements = {'GKP': 1, 'DEF': def_count, 'MID': mid_count, 'FWD': fwd_count}
        starters, bench = self._assign_starters_and_bench(selected, starting_requirements)
        
        captain_id, vice_id = self._select_captains(starters)
        
        return FPLTeam(
            starters=starters,
            bench=bench,
            captain_id=captain_id,
            vice_captain_id=vice_id,
            total_cost=sum(p['price'] for p in selected),
            predicted_points=sum(p['adjusted_points'] for p in starters),
            formation=formation
        )
    
    def _build_team_for_formation(self, df: pd.DataFrame, formation: str,
                                existing_team: Optional[List[int]] = None) -> Optional[FPLTeam]:
        """Build optimal team for specific formation following FPL rules"""
        
        # Parse formation (e.g., "3-4-3" -> [3, 4, 3])
        def_count, mid_count, fwd_count = map(int, formation.split('-'))
        
        # Validate formation meets FPL rules
        if def_count < 3 or fwd_count < 1:
            return None  # Invalid formation per FPL rules
        
        # Required positions for starting XI
        starting_requirements = {
            'GKP': 1,  # Always 1 goalkeeper
            'DEF': def_count,   # 3-5 defenders
            'MID': mid_count,   # 2-5 midfielders  
            'FWD': fwd_count    # 1-3 forwards
        }
        
        # Verify starting XI adds up to 11
        if sum(starting_requirements.values()) != 11:
            return None
        
        # Fixed squad composition per FPL rules (always 2-5-5-3)
        total_requirements = self.squad_requirements
        
        selected_players = []
        remaining_budget = self.budget
        used_teams = set()
        
        # Smart selection considering budget allocation
        # Reserve budget for minimum requirements first
        min_prices = {
            'GKP': 4.0,  # Minimum GKP price
            'DEF': 4.0,  # Minimum DEF price  
            'MID': 4.5,  # Minimum MID price
            'FWD': 4.5   # Minimum FWD price
        }
        
        # Calculate budget allocation
        total_min_cost = sum(min_prices[pos] * total_requirements[pos] for pos in total_requirements)
        if total_min_cost > self.budget:
            return None  # Impossible with current budget
        
        # Greedy selection with better budget management
        for position in ['GKP', 'FWD', 'DEF', 'MID']:  # Prioritize scarce positions first
            pos_players = df[df['position'] == position].copy()
            pos_requirement = total_requirements[position]
            
            selected_for_position = []
            
            # Calculate remaining budget accounting for other positions
            other_positions = [p for p in total_requirements if p != position]
            other_min_cost = sum(min_prices[p] * (total_requirements[p] - len([pl for pl in selected_players if pl['position'] == p])) 
                               for p in other_positions)
            available_budget = remaining_budget - other_min_cost
            
            for _, player in pos_players.iterrows():
                if len(selected_for_position) >= pos_requirement:
                    break
                
                # Budget constraint (ensure we can complete the team)
                if position == 'GKP' or position == 'FWD':
                    # Be more strict with expensive positions
                    if player['price'] > available_budget / max(1, pos_requirement - len(selected_for_position)):
                        continue
                else:
                    # Regular budget check
                    if player['price'] > remaining_budget:
                        continue
                
                # Team constraint (max 3 per team)
                team_count = sum(1 for p in selected_players if p['team_id'] == player['team_id'])
                if team_count >= 3:
                    continue
                
                # Availability constraint (be conservative for key positions)
                if position == 'GKP' and player['chance_next'] < 75:
                    continue
                
                selected_for_position.append(player.to_dict())
                remaining_budget -= player['price']
                used_teams.add(player['team_id'])
            
            if len(selected_for_position) < pos_requirement:
                # Try fallback with cheaper options
                cheap_players = pos_players[pos_players['price'] <= min_prices[position] + 1.0]
                for _, player in cheap_players.iterrows():
                    if len(selected_for_position) >= pos_requirement:
                        break
                    
                    if player['price'] > remaining_budget:
                        continue
                        
                    team_count = sum(1 for p in selected_players if p['team_id'] == player['team_id'])
                    if team_count >= 3:
                        continue
                        
                    selected_for_position.append(player.to_dict())
                    remaining_budget -= player['price']
                
                if len(selected_for_position) < pos_requirement:
                    return None  # Cannot build valid team with this formation
            
            selected_players.extend(selected_for_position)
        
        if len(selected_players) != 15:
            return None
        
        # Split into starters and bench based on formation
        starters, bench = self._assign_starters_and_bench(
            selected_players, starting_requirements
        )
        
        # Select captain and vice-captain
        captain_id, vice_id = self._select_captains(starters)
        
        # Calculate team metrics
        total_cost = sum(p['price'] for p in selected_players)
        starter_points = sum(p['adjusted_points'] for p in starters)
        bench_points = sum(p['adjusted_points'] * 0.1 for p in bench)  # Bench has 10% chance
        predicted_points = starter_points + bench_points
        
        # Captain bonus (captain gets double points, vice gets nothing extra unless captain doesn't play)
        captain = next(p for p in starters if p['player_id'] == captain_id)
        captain_bonus = captain['adjusted_points']  # Double points for captain
        predicted_points += captain_bonus
        
        return FPLTeam(
            starters=starters,
            bench=bench,
            captain_id=captain_id,
            vice_captain_id=vice_id,
            total_cost=total_cost,
            predicted_points=predicted_points,
            formation=formation
        )
    
    def _assign_starters_and_bench(self, selected_players: List[Dict], 
                                 position_requirements: Dict[str, int]) -> Tuple[List[Dict], List[Dict]]:
        """Assign players to starting XI and bench based on formation and price logic"""
        
        starters = []
        bench = []
        
        # Group players by position
        by_position = {}
        for player in selected_players:
            pos = player['position']
            if pos not in by_position:
                by_position[pos] = []
            by_position[pos].append(player)
        
        # Sort each position by a combination of price and adjusted points
        # Premium players (high price) should almost always start
        for pos in by_position:
            by_position[pos].sort(key=lambda p: (
                p['adjusted_points'] + p['price'] * 0.5,  # Boost expensive players
                p['adjusted_points'],  # Then by pure points
                -p['price']  # Tiebreaker: more expensive first
            ), reverse=True)
        
        # Assign starters first (prioritizing expensive + high-scoring players)
        for position, required_count in position_requirements.items():
            if position in by_position:
                pos_players = by_position[position]
                
                # Special rule: Players over £10m should virtually always start if selected
                premium_players = [p for p in pos_players if p['price'] >= 10.0]
                regular_players = [p for p in pos_players if p['price'] < 10.0]
                
                # Prioritize premium players for starting spots
                available_starters = premium_players + regular_players
                starters.extend(available_starters[:required_count])
                bench.extend(available_starters[required_count:])
        
        return starters, bench
    
    def _select_captains(self, starters: List[Dict]) -> Tuple[int, int]:
        """Select captain and vice-captain from starters"""
        
        # Sort starters by expected points (considering availability)
        candidates = sorted(starters, 
                          key=lambda p: p['adjusted_points'], 
                          reverse=True)
        
        # Captain: highest expected points
        captain_id = candidates[0]['player_id']
        
        # Vice-captain: second highest, but prefer different position/team for safety
        vice_id = candidates[1]['player_id']
        
        # Try to avoid same team for captain and vice
        captain_team = candidates[0]['team_id']
        for candidate in candidates[1:]:
            if candidate['team_id'] != captain_team:
                vice_id = candidate['player_id']
                break
        
        return captain_id, vice_id
    
    def _recommend_chip(self, team: FPLTeam, gameweek: int, df: pd.DataFrame) -> Optional[str]:
        """Recommend optimal chip usage based on gameweek and team composition"""
        
        # Simple chip recommendation logic
        if gameweek == 1:
            return None  # Save chips for later
        
        # Check for double gameweeks or favorable fixtures
        starter_teams = set(p['team_id'] for p in team.starters)
        bench_quality = sum(p['adjusted_points'] for p in team.bench) / 4
        
        # Recommend Bench Boost if bench is strong
        if bench_quality > 4.0:
            return "bench_boost"
        
        # Recommend Triple Captain for premium captain
        captain = next(p for p in team.starters if p['player_id'] == team.captain_id)
        if captain['adjusted_points'] > 12.0 and captain['price'] > 12.0:
            return "triple_captain"
        
        return None
    
    def _fallback_team_selection(self, df: pd.DataFrame) -> FPLTeam:
        """Improved team selection that maximizes points within budget"""
        
        # Build squad with exact FPL composition: 2-5-5-3
        selected = []
        budget = self.budget
        
        # Smart budget allocation - spend most on high-impact positions
        position_budgets = {
            'GKP': min(12.0, budget * 0.12),  # ~12% for keepers (max £12m total)
            'DEF': min(35.0, budget * 0.30),  # ~30% for defenders (max £35m total)  
            'MID': min(45.0, budget * 0.45),  # ~45% for midfielders (max £45m total)
            'FWD': min(25.0, budget * 0.25)   # ~25% for forwards (max £25m total)
        }
        
        # Ensure we can afford minimum squad (adjust if too restrictive)
        total_min_cost = 4.0*2 + 4.0*5 + 4.5*5 + 4.5*3  # Minimum viable costs
        if sum(position_budgets.values()) < total_min_cost:
            # Scale budgets proportionally if too restrictive
            scale = budget * 0.9 / sum(position_budgets.values())
            position_budgets = {pos: budget * scale for pos, budget in position_budgets.items()}
        
        # Select players for each position using budget-aware optimization
        for position, count in self.squad_requirements.items():
            pos_budget = position_budgets[position]
            pos_df = df[df['position'] == position].copy()

            # Use predicted_points if available, otherwise points_per_game
            sort_column = 'predicted_points' if 'predicted_points' in pos_df.columns else 'points_per_game'
            pos_df = pos_df.sort_values(sort_column, ascending=False)

            selected_count = 0
            position_spent = 0

            print(f"Selecting {count} {position} players from {len(pos_df)} available")

            # Try to get the best players within position budget
            for _, player in pos_df.iterrows():
                if selected_count >= count:
                    break

                # Check if we can afford this player within position budget
                if position_spent + player['price'] > pos_budget and selected_count < count - 1:
                    continue  # Save budget for remaining slots

                # Check overall budget constraint
                if player['price'] > budget:
                    continue

                # Check team constraint (max 3 per team) - use 'team' column
                team_id = player.get('team', player.get('team_id', 0))
                team_count = sum(1 for p in selected if p.get('team', p.get('team_id', 0)) == team_id)
                if team_count >= 3:
                    continue

                # Convert pandas Series to dict properly
                player_dict = player.to_dict() if hasattr(player, 'to_dict') else dict(player)
                selected.append(player_dict)
                budget -= player['price']
                position_spent += player['price']
                selected_count += 1
            
            # If we couldn't fill the position, get cheapest available players
            if selected_count < count:
                print(f"Only selected {selected_count}/{count} {position}, trying cheaper options...")
                pos_df_cheap = pos_df.sort_values('price')
                for _, player in pos_df_cheap.iterrows():
                    if selected_count >= count:
                        break

                    if player['price'] > budget:
                        continue

                    # Skip already selected players
                    if any(p['player_id'] == player['player_id'] for p in selected):
                        continue

                    # Check team constraint - use 'team' column
                    team_id = player.get('team', player.get('team_id', 0))
                    team_count = sum(1 for p in selected if p.get('team', p.get('team_id', 0)) == team_id)
                    if team_count >= 3:
                        continue

                    # Convert pandas Series to dict properly
                    player_dict = player.to_dict() if hasattr(player, 'to_dict') else dict(player)
                    selected.append(player_dict)
                    budget -= player['price']
                    position_spent += player['price']
                    selected_count += 1

            print(f"Final {position} selection: {selected_count}/{count} players")
                    
        print(f"Team selection: {len(selected)} players, £{sum(p['price'] for p in selected):.1f}m spent, £{budget:.1f}m remaining")
        
        # Default to 3-4-3 formation for fallback
        starting_requirements = {'GKP': 1, 'DEF': 3, 'MID': 4, 'FWD': 3}
        
        # Assign starters and bench
        starters, bench = self._assign_starters_and_bench(selected, starting_requirements)
        
        captain_id, vice_id = self._select_captains(starters)
        
        return FPLTeam(
            starters=starters,
            bench=bench,
            captain_id=captain_id,
            vice_captain_id=vice_id,
            total_cost=sum(p['price'] for p in selected),
            predicted_points=sum(p['adjusted_points'] for p in starters),
            formation="3-4-3"
        )
    
    def evaluate_transfers(self, current_team: List[int], predictions_df: pd.DataFrame,
                          gameweek: int) -> List[Dict]:
        """Suggest transfers to improve the team"""
        
        optimal_team = self.select_optimal_team(predictions_df, gameweek, current_team)
        
        current_ids = set(current_team)
        optimal_ids = set(p['player_id'] for p in optimal_team.starters + optimal_team.bench)
        
        # Players to transfer out
        out_players = current_ids - optimal_ids
        
        # Players to transfer in  
        in_players = optimal_ids - current_ids
        
        suggestions = []
        for out_id in list(out_players)[:3]:  # Max 3 transfers typically
            out_player = predictions_df[predictions_df['player_id'] == out_id].iloc[0]
            
            # Find best replacement in same position
            same_pos = predictions_df[
                (predictions_df['position'] == out_player['position']) &
                (predictions_df['player_id'].isin(in_players))
            ]
            
            if not same_pos.empty:
                in_player = same_pos.iloc[0]
                
                suggestions.append({
                    'out': {
                        'id': out_id,
                        'name': out_player['name'],
                        'price': out_player['price'],
                        'predicted_points': out_player['predicted_points']
                    },
                    'in': {
                        'id': in_player['player_id'],
                        'name': in_player['name'], 
                        'price': in_player['price'],
                        'predicted_points': in_player['predicted_points']
                    },
                    'points_gain': in_player['predicted_points'] - out_player['predicted_points'],
                    'cost_change': in_player['price'] - out_player['price']
                })
        
        return sorted(suggestions, key=lambda x: x['points_gain'], reverse=True)
    
    def apply_automatic_substitutions(self, team: FPLTeam) -> FPLTeam:
        """Apply FPL automatic substitution rules for non-playing players"""
        
        # Identify non-playing starters (chance_next < 50% or injured status)
        non_playing_starters = []
        for player in team.starters:
            chance = pd.to_numeric(player.get('chance_next', 100), errors='coerce')
            status = player.get('fpl_status', 'a')
            
            if status in ['i', 's', 'u'] or (pd.notna(chance) and chance < 50):
                non_playing_starters.append(player)
        
        if not non_playing_starters:
            return team  # No substitutions needed
        
        # Available bench players for substitution
        available_bench = [p for p in team.bench 
                          if p.get('fpl_status', 'a') not in ['i', 's', 'u']
                          and pd.to_numeric(p.get('chance_next', 100), errors='coerce') >= 50]
        
        new_starters = [p for p in team.starters if p not in non_playing_starters]
        new_bench = list(team.bench)
        substitutions_made = []
        
        # FPL substitution rules: maintain valid formation (min 3 DEF, min 1 FWD)
        for non_player in non_playing_starters:
            best_sub = None
            
            # First, try same position substitution
            same_position_subs = [p for p in available_bench if p['position'] == non_player['position']]
            if same_position_subs:
                best_sub = max(same_position_subs, key=lambda p: p.get('adjusted_points', 0))
            
            # If no same position available, try any valid substitution
            if not best_sub:
                for bench_player in available_bench:
                    # Check if this substitution maintains valid formation
                    temp_starters = new_starters + [bench_player]
                    position_counts = {'GKP': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
                    
                    for p in temp_starters:
                        position_counts[p['position']] += 1
                    
                    # Must have exactly 1 GKP, at least 3 DEF, at least 1 FWD
                    if (position_counts['GKP'] == 1 and 
                        position_counts['DEF'] >= 3 and 
                        position_counts['FWD'] >= 1):
                        best_sub = bench_player
                        break
            
            if best_sub:
                new_starters.append(best_sub)
                new_bench.remove(best_sub)
                new_bench.append(non_player)
                available_bench.remove(best_sub)
                substitutions_made.append(f"{best_sub['name']} → {non_player['name']}")
        
        # Create new team with substitutions
        return FPLTeam(
            starters=new_starters,
            bench=new_bench,
            captain_id=team.captain_id,
            vice_captain_id=team.vice_captain_id,
            total_cost=team.total_cost,
            predicted_points=sum(p.get('adjusted_points', 0) for p in new_starters),
            formation=team.formation,
            chip_recommendation=team.chip_recommendation
        )