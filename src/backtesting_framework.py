from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import json
from dataclasses import dataclass
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .fpl_client import FPLClient
from .ml_pipeline import MLPipeline
from .weekly_retrainer import WeeklyMLRetrainer


@dataclass
class BacktestResult:
    """Results from a backtesting run"""
    gameweek: int
    mae: float
    rmse: float
    accuracy_pct: float
    captain_accuracy: float
    bonus_accuracy: float
    predictions_count: int
    average_predicted_points: float
    average_actual_points: float
    top_player_accuracy: float
    prediction_errors: List[float]


class BacktestingFramework:
    """
    Comprehensive backtesting framework for validating FPL predictions
    against historical data to measure and improve model accuracy
    """

    def __init__(self):
        self.fpl = FPLClient()
        self.ml_pipeline = MLPipeline()
        self.retrainer = WeeklyMLRetrainer()
        self.logger = logging.getLogger(__name__)

        # Create output directories
        Path("data/backtesting").mkdir(parents=True, exist_ok=True)
        Path("reports/backtesting").mkdir(parents=True, exist_ok=True)

        # Backtesting parameters
        self.min_gameweeks = 5  # Minimum GWs for meaningful backtest
        self.captain_threshold = 8.0  # Points for successful captain
        self.top_players_count = 20  # Top N players to check accuracy

    def run_walk_forward_validation(self, start_gameweek: int = 5, end_gameweek: int = 15) -> List[BacktestResult]:
        """
        Run walk-forward validation across multiple gameweeks
        Train on historical data, predict next GW, validate against actual results
        """

        if end_gameweek - start_gameweek < self.min_gameweeks:
            raise ValueError(f"Need at least {self.min_gameweeks} gameweeks for validation")

        results = []

        for test_gw in range(start_gameweek, end_gameweek + 1):
            self.logger.info(f"Running backtest for gameweek {test_gw}")

            try:
                result = self._validate_single_gameweek(test_gw)
                if result:
                    results.append(result)
                    self.logger.info(f"GW{test_gw} - MAE: {result.mae:.2f}, Accuracy: {result.accuracy_pct:.1f}%")
                else:
                    self.logger.warning(f"Failed to validate gameweek {test_gw}")

            except Exception as e:
                self.logger.error(f"Error validating GW {test_gw}: {e}")
                continue

        # Save results
        self._save_backtest_results(results)

        return results

    def _validate_single_gameweek(self, gameweek: int) -> Optional[BacktestResult]:
        """Validate predictions for a single gameweek against actual results"""

        try:
            # Step 1: Generate predictions for this gameweek
            # (In real backtest, we'd train on data up to gameweek-1)
            predictions = self.ml_pipeline.predict_current()
            if predictions.empty:
                self.logger.warning(f"No predictions generated for GW {gameweek}")
                return None

            # Step 2: Get actual results for this gameweek
            actual_results = self.retrainer.collect_gameweek_results(gameweek)
            if actual_results is None or actual_results.empty:
                self.logger.warning(f"No actual results for GW {gameweek}")
                return None

            # Step 3: Align predictions with actual results
            comparison_df = self._align_predictions_with_actuals(predictions, actual_results)
            if comparison_df.empty:
                self.logger.warning(f"No aligned data for GW {gameweek}")
                return None

            # Step 4: Calculate metrics
            return self._calculate_backtest_metrics(gameweek, comparison_df)

        except Exception as e:
            self.logger.error(f"Failed to validate gameweek {gameweek}: {e}")
            return None

    def _align_predictions_with_actuals(self, predictions: pd.DataFrame,
                                      actual_results: pd.DataFrame) -> pd.DataFrame:
        """Align prediction data with actual results for comparison"""

        # Get common prediction column name
        pred_col = 'predicted_points'
        if pred_col not in predictions.columns:
            pred_col = 'ep_ml'  # Fallback
        if pred_col not in predictions.columns:
            self.logger.error("No prediction column found")
            return pd.DataFrame()

        # Prepare predictions dataframe
        pred_df = predictions[['id', pred_col]].copy()
        pred_df = pred_df.rename(columns={pred_col: 'predicted_points', 'id': 'player_id'})

        # Prepare actuals dataframe
        actual_df = actual_results[['player_id', 'points']].copy()
        actual_df = actual_df.rename(columns={'points': 'actual_points'})

        # Merge on player_id
        comparison = pred_df.merge(actual_df, on='player_id', how='inner')

        # Add additional prediction data if available
        if 'position_id' in predictions.columns:
            pos_data = predictions[['id', 'position_id']].rename(columns={'id': 'player_id'})
            comparison = comparison.merge(pos_data, on='player_id', how='left')

        if 'price' in predictions.columns:
            price_data = predictions[['id', 'price']].rename(columns={'id': 'player_id'})
            comparison = comparison.merge(price_data, on='player_id', how='left')

        # Calculate prediction error
        comparison['prediction_error'] = comparison['actual_points'] - comparison['predicted_points']
        comparison['absolute_error'] = comparison['prediction_error'].abs()

        return comparison

    def _calculate_backtest_metrics(self, gameweek: int, comparison_df: pd.DataFrame) -> BacktestResult:
        """Calculate comprehensive metrics for backtest validation"""

        predicted = comparison_df['predicted_points'].values
        actual = comparison_df['actual_points'].values

        # Basic accuracy metrics
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))

        # Percentage accuracy (how often predictions are within reasonable range)
        accurate_predictions = np.sum(comparison_df['absolute_error'] <= 2.0)  # Within 2 points
        accuracy_pct = (accurate_predictions / len(comparison_df)) * 100

        # Captain accuracy (top predicted players actually scored well)
        top_predicted = comparison_df.nlargest(5, 'predicted_points')
        captain_success = (top_predicted['actual_points'] >= self.captain_threshold).mean()

        # Bonus point accuracy (players predicted to get bonus actually did)
        bonus_accuracy = self._calculate_bonus_accuracy(comparison_df)

        # Top players accuracy (how well we predicted the highest scorers)
        top_actual = comparison_df.nlargest(self.top_players_count, 'actual_points')
        top_player_mae = mean_absolute_error(
            top_actual['actual_points'].values,
            top_actual['predicted_points'].values
        )
        top_player_accuracy = max(0, 100 - (top_player_mae * 10))  # Convert to percentage

        return BacktestResult(
            gameweek=gameweek,
            mae=mae,
            rmse=rmse,
            accuracy_pct=accuracy_pct,
            captain_accuracy=captain_success * 100,
            bonus_accuracy=bonus_accuracy,
            predictions_count=len(comparison_df),
            average_predicted_points=comparison_df['predicted_points'].mean(),
            average_actual_points=comparison_df['actual_points'].mean(),
            top_player_accuracy=top_player_accuracy,
            prediction_errors=comparison_df['prediction_error'].tolist()
        )

    def _calculate_bonus_accuracy(self, comparison_df: pd.DataFrame) -> float:
        """Calculate accuracy of bonus point predictions"""

        # If we have BPS data, use it; otherwise use point thresholds
        try:
            # Players predicted to get 2+ bonus points
            high_predicted = comparison_df[comparison_df['predicted_points'] >= 8.0]
            if len(high_predicted) == 0:
                return 0.0

            # Check how many actually got bonus (assuming bonus players score 6+ with 2-3 bonus)
            bonus_success = (high_predicted['actual_points'] >= 8.0).mean()
            return bonus_success * 100

        except Exception:
            return 0.0

    def run_position_analysis(self, gameweek_range: Tuple[int, int]) -> Dict[str, Dict]:
        """Analyze prediction accuracy by position"""

        position_results = {
            'GKP': {'mae': [], 'accuracy': []},
            'DEF': {'mae': [], 'accuracy': []},
            'MID': {'mae': [], 'accuracy': []},
            'FWD': {'mae': [], 'accuracy': []}
        }

        for gw in range(gameweek_range[0], gameweek_range[1] + 1):
            try:
                predictions = self.ml_pipeline.predict_current()
                actual_results = self.retrainer.collect_gameweek_results(gw)

                if predictions.empty or actual_results is None:
                    continue

                comparison = self._align_predictions_with_actuals(predictions, actual_results)

                # Group by position and calculate metrics
                for position_id, position_name in [(1, 'GKP'), (2, 'DEF'), (3, 'MID'), (4, 'FWD')]:
                    pos_data = comparison[comparison.get('position_id', 0) == position_id]
                    if len(pos_data) > 0:
                        mae = mean_absolute_error(pos_data['actual_points'], pos_data['predicted_points'])
                        accuracy = (pos_data['absolute_error'] <= 2.0).mean() * 100

                        position_results[position_name]['mae'].append(mae)
                        position_results[position_name]['accuracy'].append(accuracy)

            except Exception as e:
                self.logger.error(f"Error in position analysis for GW {gw}: {e}")
                continue

        # Calculate averages
        summary = {}
        for pos, data in position_results.items():
            if data['mae']:
                summary[pos] = {
                    'average_mae': np.mean(data['mae']),
                    'average_accuracy': np.mean(data['accuracy']),
                    'sample_size': len(data['mae'])
                }
            else:
                summary[pos] = {'average_mae': 0, 'average_accuracy': 0, 'sample_size': 0}

        return summary

    def _save_backtest_results(self, results: List[BacktestResult]) -> None:
        """Save backtesting results to JSON file"""

        # Convert results to serializable format
        results_data = []
        for result in results:
            results_data.append({
                'gameweek': result.gameweek,
                'mae': result.mae,
                'rmse': result.rmse,
                'accuracy_pct': result.accuracy_pct,
                'captain_accuracy': result.captain_accuracy,
                'bonus_accuracy': result.bonus_accuracy,
                'predictions_count': result.predictions_count,
                'average_predicted_points': result.average_predicted_points,
                'average_actual_points': result.average_actual_points,
                'top_player_accuracy': result.top_player_accuracy,
                'prediction_errors': result.prediction_errors[:50]  # Limit to first 50 errors
            })

        # Save to file
        output_file = f"data/backtesting/backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': results_data,
                'summary': self._generate_summary(results)
            }, f, indent=2)

        self.logger.info(f"Backtest results saved to {output_file}")

    def _generate_summary(self, results: List[BacktestResult]) -> Dict:
        """Generate summary statistics from backtest results"""

        if not results:
            return {}

        mae_values = [r.mae for r in results]
        accuracy_values = [r.accuracy_pct for r in results]
        captain_accuracy_values = [r.captain_accuracy for r in results]

        return {
            'total_gameweeks': len(results),
            'average_mae': np.mean(mae_values),
            'mae_std': np.std(mae_values),
            'average_accuracy': np.mean(accuracy_values),
            'accuracy_std': np.std(accuracy_values),
            'average_captain_accuracy': np.mean(captain_accuracy_values),
            'best_gameweek_mae': min(mae_values),
            'worst_gameweek_mae': max(mae_values),
            'consistent_performance': np.std(accuracy_values) < 15.0  # Low variability = consistent
        }

    def generate_backtest_report(self, results: List[BacktestResult]) -> str:
        """Generate a comprehensive backtest report"""

        if not results:
            return "No backtest results available"

        summary = self._generate_summary(results)

        report = f"""
🏆 FPL BACKTESTING REPORT
========================

📊 OVERALL PERFORMANCE:
• Total Gameweeks Tested: {summary['total_gameweeks']}
• Average MAE: {summary['average_mae']:.2f} points
• Average Accuracy: {summary['average_accuracy']:.1f}%
• Captain Success Rate: {summary['average_captain_accuracy']:.1f}%

📈 ACCURACY TRENDS:
• Best GW MAE: {summary['best_gameweek_mae']:.2f} points
• Worst GW MAE: {summary['worst_gameweek_mae']:.2f} points
• Consistency: {'High' if summary['consistent_performance'] else 'Variable'}

🎯 GAMEWEEK BREAKDOWN:
"""

        for result in results:
            report += f"""
GW{result.gameweek:2d}: MAE {result.mae:.2f} | Accuracy {result.accuracy_pct:.1f}% | Captain {result.captain_accuracy:.1f}%
     Predicted avg: {result.average_predicted_points:.1f} | Actual avg: {result.average_actual_points:.1f}
"""

        # Performance assessment
        target_mae = 2.5
        target_accuracy = 70.0
        current_mae = summary['average_mae']
        current_accuracy = summary['average_accuracy']

        report += f"""

🔍 PERFORMANCE ASSESSMENT:
• MAE Target: {target_mae} points | Current: {current_mae:.2f} {'✅' if current_mae <= target_mae else '❌'}
• Accuracy Target: {target_accuracy}% | Current: {current_accuracy:.1f}% {'✅' if current_accuracy >= target_accuracy else '❌'}

💡 RECOMMENDATIONS:
"""

        if current_mae > target_mae:
            report += "• Focus on reducing prediction errors - consider model tuning\n"
        if current_accuracy < target_accuracy:
            report += "• Improve overall accuracy - review feature engineering\n"
        if summary['average_captain_accuracy'] < 50:
            report += "• Captain predictions need improvement - focus on high-ceiling players\n"

        return report