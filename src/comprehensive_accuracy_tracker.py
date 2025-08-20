from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import logging
from dataclasses import dataclass
from collections import defaultdict

from .fpl_client import FPLClient
from .team_selector import TeamSelector
from .weekly_retrainer import WeeklyMLRetrainer


@dataclass
class GameweekSuggestion:
    """Track team suggestions for each gameweek"""
    gameweek: int
    timestamp: datetime
    starters: List[Dict]
    bench: List[Dict]
    captain_id: int
    vice_captain_id: int
    formation: str
    predicted_points: float
    total_cost: float
    chip_recommendation: Optional[str]
    transfer_suggestions: List[Dict]


@dataclass 
class GameweekResult:
    """Actual results for a gameweek"""
    gameweek: int
    team_selection: GameweekSuggestion
    actual_team_points: float
    actual_captain_points: float
    actual_vice_points: float
    points_with_transfers: float  # After transfer costs
    points_vs_optimal: float     # vs best possible team
    prediction_accuracy: Dict[str, float]  # MAE, RMSE etc


class ComprehensiveAccuracyTracker:
    """
    Complete system for tracking prediction accuracy and team performance
    across gameweeks with comparison against actual results
    """
    
    def __init__(self):
        self.fpl = FPLClient()
        self.selector = TeamSelector() 
        self.retrainer = WeeklyMLRetrainer()
        self.logger = logging.getLogger(__name__)
        
        # Storage paths
        self.tracking_dir = Path("data/accuracy_tracking")
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        
        self.suggestions_file = self.tracking_dir / "gameweek_suggestions.json"
        self.results_file = self.tracking_dir / "gameweek_results.json" 
        self.performance_file = self.tracking_dir / "overall_performance.json"
        
        # Load existing data
        self.suggestions_history: List[GameweekSuggestion] = self._load_suggestions_history()
        self.results_history: List[GameweekResult] = self._load_results_history()
        
    def log_gameweek_suggestion(self, gameweek: int, predictions_df: pd.DataFrame, 
                               transfer_suggestions: List[Dict] = None) -> GameweekSuggestion:
        """Log team suggestions for a specific gameweek"""
        try:
            # Generate optimal team
            optimal_team = self.selector.select_optimal_team(predictions_df, gameweek)
            
            # Create suggestion record
            suggestion = GameweekSuggestion(
                gameweek=gameweek,
                timestamp=datetime.now(),
                starters=optimal_team.starters,
                bench=optimal_team.bench,
                captain_id=optimal_team.captain_id,
                vice_captain_id=optimal_team.vice_captain_id,
                formation=optimal_team.formation,
                predicted_points=optimal_team.predicted_points,
                total_cost=optimal_team.total_cost,
                chip_recommendation=optimal_team.chip_recommendation,
                transfer_suggestions=transfer_suggestions or []
            )
            
            # Store in history
            self.suggestions_history.append(suggestion)
            self._save_suggestions_history()
            
            # Also save gameweek-specific predictions for later comparison
            pred_file = self.tracking_dir / f"gw{gameweek}_team_predictions.csv"
            team_predictions = []
            
            for player in optimal_team.starters + optimal_team.bench:
                pred_row = predictions_df[predictions_df['player_id'] == player['player_id']]
                if not pred_row.empty:
                    team_predictions.append({
                        'player_id': player['player_id'],
                        'name': player['name'],
                        'position': player['position'],
                        'predicted_points': pred_row.iloc[0]['predicted_points'],
                        'price': player['price'],
                        'is_captain': player['player_id'] == optimal_team.captain_id,
                        'is_vice': player['player_id'] == optimal_team.vice_captain_id,
                        'is_starter': player in optimal_team.starters
                    })
            
            if team_predictions:
                pd.DataFrame(team_predictions).to_csv(pred_file, index=False)
            
            self.logger.info(f"Logged team suggestion for GW{gameweek}: {optimal_team.predicted_points:.1f} predicted points")
            return suggestion
            
        except Exception as e:
            self.logger.error(f"Failed to log GW{gameweek} suggestion: {e}")
            raise
    
    def collect_gameweek_results(self, gameweek: int) -> Optional[GameweekResult]:
        """Collect actual results for a completed gameweek and compare with predictions"""
        try:
            # Find the suggestion for this gameweek
            suggestion = next(
                (s for s in self.suggestions_history if s.gameweek == gameweek), 
                None
            )
            
            if not suggestion:
                self.logger.warning(f"No suggestion found for GW{gameweek}")
                return None
            
            # Get actual player results
            actual_results = self.retrainer.collect_gameweek_results(gameweek)
            if actual_results is None:
                self.logger.warning(f"No actual results available for GW{gameweek}")
                return None
            
            # Calculate team performance
            team_performance = self._calculate_team_performance(suggestion, actual_results)
            
            # Calculate prediction accuracy
            prediction_accuracy = self._calculate_prediction_accuracy(gameweek, actual_results)
            
            # Calculate optimal team performance (what we could have achieved)
            optimal_performance = self._calculate_optimal_performance(gameweek, actual_results)
            
            # Create result record
            result = GameweekResult(
                gameweek=gameweek,
                team_selection=suggestion,
                actual_team_points=team_performance['total_points'],
                actual_captain_points=team_performance['captain_points'], 
                actual_vice_points=team_performance['vice_points'],
                points_with_transfers=team_performance['points_after_transfers'],
                points_vs_optimal=team_performance['total_points'] - optimal_performance,
                prediction_accuracy=prediction_accuracy
            )
            
            # Store result
            self.results_history.append(result)
            self._save_results_history()
            
            self.logger.info(f"Collected GW{gameweek} results: {result.actual_team_points} points "
                           f"(vs {result.predicted_points:.1f} predicted)")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to collect GW{gameweek} results: {e}")
            return None
    
    def _calculate_team_performance(self, suggestion: GameweekSuggestion, 
                                  actual_results: pd.DataFrame) -> Dict[str, float]:
        """Calculate how the suggested team actually performed"""
        
        team_points = 0
        captain_points = 0
        vice_points = 0
        players_found = 0
        
        # Calculate starter points
        for player in suggestion.starters:
            player_result = actual_results[actual_results['player_id'] == player['player_id']]
            if not player_result.empty:
                points = player_result.iloc[0]['actual_points']
                
                if player['player_id'] == suggestion.captain_id:
                    captain_points = points
                    team_points += points * 2  # Captain gets double
                elif player['player_id'] == suggestion.vice_captain_id:
                    vice_points = points
                    team_points += points
                else:
                    team_points += points
                    
                players_found += 1
        
        # Handle bench (only if starter didn't play)
        # Simplified - assume all starters played
        
        # Calculate transfer costs (if any transfer suggestions were followed)
        transfer_cost = len(suggestion.transfer_suggestions) * 4  # 4 points per transfer
        points_after_transfers = max(0, team_points - transfer_cost)
        
        return {
            'total_points': team_points,
            'captain_points': captain_points,
            'vice_points': vice_points,
            'points_after_transfers': points_after_transfers,
            'players_found': players_found,
            'transfer_cost': transfer_cost
        }
    
    def _calculate_prediction_accuracy(self, gameweek: int, 
                                     actual_results: pd.DataFrame) -> Dict[str, float]:
        """Calculate accuracy metrics for this gameweek's predictions"""
        try:
            # Load team predictions
            pred_file = self.tracking_dir / f"gw{gameweek}_team_predictions.csv"
            if not pred_file.exists():
                return {}
            
            team_preds = pd.read_csv(pred_file)
            
            # Merge with actual results
            comparison = team_preds.merge(
                actual_results[['player_id', 'actual_points']], 
                on='player_id', 
                how='inner'
            )
            
            if comparison.empty:
                return {}
            
            # Calculate metrics
            mae = np.mean(np.abs(comparison['predicted_points'] - comparison['actual_points']))
            rmse = np.sqrt(np.mean((comparison['predicted_points'] - comparison['actual_points']) ** 2))
            correlation = comparison['predicted_points'].corr(comparison['actual_points'])
            
            # Captain accuracy (most important)
            captain_data = comparison[comparison['is_captain']]
            captain_error = 0
            if not captain_data.empty:
                captain_error = abs(captain_data.iloc[0]['predicted_points'] - 
                                  captain_data.iloc[0]['actual_points'])
            
            return {
                'mae': float(mae),
                'rmse': float(rmse),
                'correlation': float(correlation) if not pd.isna(correlation) else 0.0,
                'captain_error': float(captain_error),
                'players_evaluated': len(comparison)
            }
            
        except Exception as e:
            self.logger.error(f"Prediction accuracy calculation failed for GW{gameweek}: {e}")
            return {}
    
    def _calculate_optimal_performance(self, gameweek: int, 
                                     actual_results: pd.DataFrame) -> float:
        """Calculate what the optimal team would have scored (hindsight)"""
        try:
            # Sort by actual points and select optimal team within budget constraints
            # Simplified version - top 15 players by actual points
            optimal_selection = actual_results.nlargest(15, 'actual_points')
            
            # Calculate optimal points (top player as captain)
            if len(optimal_selection) >= 11:
                starters = optimal_selection.head(11)
                captain_points = starters.iloc[0]['actual_points'] * 2
                other_points = starters.iloc[1:]['actual_points'].sum()
                return captain_points + other_points
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Optimal performance calculation failed: {e}")
            return 0
    
    def get_comprehensive_report(self, last_n_gameweeks: int = 10) -> Dict[str, Any]:
        """Generate comprehensive accuracy and performance report"""
        
        recent_results = self.results_history[-last_n_gameweeks:] if self.results_history else []
        
        if not recent_results:
            return {"error": "No results available"}
        
        # Overall performance metrics
        total_predicted = sum(r.team_selection.predicted_points for r in recent_results)
        total_actual = sum(r.actual_team_points for r in recent_results)
        
        prediction_errors = [
            abs(r.team_selection.predicted_points - r.actual_team_points) 
            for r in recent_results
        ]
        
        # Captain performance
        captain_predictions = []
        captain_actuals = []
        
        for result in recent_results:
            captain_id = result.team_selection.captain_id
            captain_actual = result.actual_captain_points
            
            # Find predicted points for captain
            captain_pred = 0
            pred_file = self.tracking_dir / f"gw{result.gameweek}_team_predictions.csv"
            if pred_file.exists():
                preds = pd.read_csv(pred_file)
                captain_data = preds[preds['is_captain']]
                if not captain_data.empty:
                    captain_pred = captain_data.iloc[0]['predicted_points']
            
            if captain_pred > 0:
                captain_predictions.append(captain_pred)
                captain_actuals.append(captain_actual)
        
        # Transfer effectiveness
        transfer_results = []
        for result in recent_results:
            if result.team_selection.transfer_suggestions:
                transfer_benefit = result.points_with_transfers - result.actual_team_points
                transfer_results.append({
                    'gameweek': result.gameweek,
                    'transfers_made': len(result.team_selection.transfer_suggestions),
                    'net_benefit': transfer_benefit
                })
        
        # Points vs optimal analysis
        vs_optimal = [r.points_vs_optimal for r in recent_results]
        
        report = {
            'summary': {
                'gameweeks_analyzed': len(recent_results),
                'total_predicted_points': total_predicted,
                'total_actual_points': total_actual,
                'prediction_error': abs(total_predicted - total_actual),
                'average_weekly_error': np.mean(prediction_errors),
                'prediction_accuracy_pct': max(0, 100 - (np.mean(prediction_errors) / 
                                                       (total_actual / len(recent_results)) * 100))
            },
            'captain_performance': {
                'captain_predictions': captain_predictions,
                'captain_actuals': captain_actuals,
                'captain_accuracy': np.corrcoef(captain_predictions, captain_actuals)[0,1] 
                                  if len(captain_predictions) > 1 else 0,
                'average_captain_error': np.mean([abs(p-a) for p,a in 
                                                zip(captain_predictions, captain_actuals)])
            },
            'transfer_analysis': {
                'transfer_results': transfer_results,
                'average_transfer_benefit': np.mean([t['net_benefit'] for t in transfer_results]) 
                                          if transfer_results else 0
            },
            'vs_optimal': {
                'average_points_lost': np.mean([abs(x) for x in vs_optimal]),
                'percentage_of_optimal': (total_actual / 
                                        (total_actual + sum(abs(x) for x in vs_optimal))) * 100
            },
            'gameweek_breakdown': [
                {
                    'gameweek': r.gameweek,
                    'predicted': r.team_selection.predicted_points,
                    'actual': r.actual_team_points,
                    'error': abs(r.team_selection.predicted_points - r.actual_team_points),
                    'vs_optimal': r.points_vs_optimal
                }
                for r in recent_results
            ]
        }
        
        return report
    
    def get_improvement_suggestions(self) -> List[str]:
        """Analyze performance and suggest improvements"""
        
        if len(self.results_history) < 3:
            return ["Need more gameweeks of data to provide suggestions"]
        
        recent_results = self.results_history[-5:]
        suggestions = []
        
        # Analyze prediction accuracy trend
        errors = [abs(r.team_selection.predicted_points - r.actual_team_points) 
                 for r in recent_results]
        
        if len(errors) >= 3:
            recent_error = np.mean(errors[-3:])
            if recent_error > 15:
                suggestions.append("Prediction accuracy declining - consider model retraining")
        
        # Analyze captain performance
        captain_errors = []
        for result in recent_results:
            if result.actual_captain_points > 0:
                captain_errors.append(
                    abs(result.actual_captain_points - 
                        self._get_captain_prediction(result.gameweek))
                )
        
        if captain_errors and np.mean(captain_errors) > 5:
            suggestions.append("Captain predictions need improvement - review captaincy logic")
        
        # Analyze transfer effectiveness
        transfer_benefits = []
        for result in recent_results:
            if result.team_selection.transfer_suggestions:
                transfer_benefits.append(result.points_with_transfers - result.actual_team_points)
        
        if transfer_benefits and np.mean(transfer_benefits) < -2:
            suggestions.append("Transfer suggestions are costing points - review transfer logic")
        
        # Analyze vs optimal performance
        vs_optimal = [r.points_vs_optimal for r in recent_results]
        avg_lost = np.mean([abs(x) for x in vs_optimal])
        
        if avg_lost > 20:
            suggestions.append("Large gap vs optimal team - consider more aggressive player selection")
        
        return suggestions or ["Performance looks good - continue current approach"]
    
    def _get_captain_prediction(self, gameweek: int) -> float:
        """Get predicted points for captain in a gameweek"""
        pred_file = self.tracking_dir / f"gw{gameweek}_team_predictions.csv"
        if pred_file.exists():
            preds = pd.read_csv(pred_file)
            captain_data = preds[preds['is_captain']]
            if not captain_data.empty:
                return captain_data.iloc[0]['predicted_points']
        return 0
    
    def _load_suggestions_history(self) -> List[GameweekSuggestion]:
        """Load saved gameweek suggestions"""
        if not self.suggestions_file.exists():
            return []
        
        try:
            with open(self.suggestions_file, 'r') as f:
                data = json.load(f)
            
            suggestions = []
            for item in data:
                suggestion = GameweekSuggestion(
                    gameweek=item['gameweek'],
                    timestamp=datetime.fromisoformat(item['timestamp']),
                    starters=item['starters'],
                    bench=item['bench'],
                    captain_id=item['captain_id'],
                    vice_captain_id=item['vice_captain_id'],
                    formation=item['formation'],
                    predicted_points=item['predicted_points'],
                    total_cost=item['total_cost'],
                    chip_recommendation=item.get('chip_recommendation'),
                    transfer_suggestions=item.get('transfer_suggestions', [])
                )
                suggestions.append(suggestion)
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Failed to load suggestions history: {e}")
            return []
    
    def _save_suggestions_history(self):
        """Save gameweek suggestions to disk"""
        try:
            data = []
            for suggestion in self.suggestions_history:
                data.append({
                    'gameweek': suggestion.gameweek,
                    'timestamp': suggestion.timestamp.isoformat(),
                    'starters': suggestion.starters,
                    'bench': suggestion.bench,
                    'captain_id': suggestion.captain_id,
                    'vice_captain_id': suggestion.vice_captain_id,
                    'formation': suggestion.formation,
                    'predicted_points': suggestion.predicted_points,
                    'total_cost': suggestion.total_cost,
                    'chip_recommendation': suggestion.chip_recommendation,
                    'transfer_suggestions': suggestion.transfer_suggestions
                })
            
            with open(self.suggestions_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save suggestions history: {e}")
    
    def _load_results_history(self) -> List[GameweekResult]:
        """Load saved gameweek results"""
        if not self.results_file.exists():
            return []
        
        try:
            with open(self.results_file, 'r') as f:
                data = json.load(f)
            
            results = []
            for item in data:
                # Reconstruct suggestion from stored data
                suggestion_data = item['team_selection']
                suggestion = GameweekSuggestion(
                    gameweek=suggestion_data['gameweek'],
                    timestamp=datetime.fromisoformat(suggestion_data['timestamp']),
                    starters=suggestion_data['starters'],
                    bench=suggestion_data['bench'],
                    captain_id=suggestion_data['captain_id'],
                    vice_captain_id=suggestion_data['vice_captain_id'],
                    formation=suggestion_data['formation'],
                    predicted_points=suggestion_data['predicted_points'],
                    total_cost=suggestion_data['total_cost'],
                    chip_recommendation=suggestion_data.get('chip_recommendation'),
                    transfer_suggestions=suggestion_data.get('transfer_suggestions', [])
                )
                
                result = GameweekResult(
                    gameweek=item['gameweek'],
                    team_selection=suggestion,
                    actual_team_points=item['actual_team_points'],
                    actual_captain_points=item['actual_captain_points'],
                    actual_vice_points=item['actual_vice_points'],
                    points_with_transfers=item['points_with_transfers'],
                    points_vs_optimal=item['points_vs_optimal'],
                    prediction_accuracy=item['prediction_accuracy']
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to load results history: {e}")
            return []
    
    def _save_results_history(self):
        """Save gameweek results to disk"""
        try:
            data = []
            for result in self.results_history:
                # Convert suggestion to dict
                suggestion_dict = {
                    'gameweek': result.team_selection.gameweek,
                    'timestamp': result.team_selection.timestamp.isoformat(),
                    'starters': result.team_selection.starters,
                    'bench': result.team_selection.bench,
                    'captain_id': result.team_selection.captain_id,
                    'vice_captain_id': result.team_selection.vice_captain_id,
                    'formation': result.team_selection.formation,
                    'predicted_points': result.team_selection.predicted_points,
                    'total_cost': result.team_selection.total_cost,
                    'chip_recommendation': result.team_selection.chip_recommendation,
                    'transfer_suggestions': result.team_selection.transfer_suggestions
                }
                
                data.append({
                    'gameweek': result.gameweek,
                    'team_selection': suggestion_dict,
                    'actual_team_points': result.actual_team_points,
                    'actual_captain_points': result.actual_captain_points,
                    'actual_vice_points': result.actual_vice_points,
                    'points_with_transfers': result.points_with_transfers,
                    'points_vs_optimal': result.points_vs_optimal,
                    'prediction_accuracy': result.prediction_accuracy
                })
            
            with open(self.results_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save results history: {e}")