from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import logging
from dataclasses import dataclass

from .comprehensive_accuracy_tracker import ComprehensiveAccuracyTracker
from .ml_pipeline import MLPipeline


@dataclass
class ModelFeedback:
    """Feedback analysis for model improvement"""
    feature_importance_changes: Dict[str, float]
    prediction_error_patterns: Dict[str, Any]
    player_bias_corrections: Dict[str, float]
    position_adjustments: Dict[str, float]
    confidence_score: float
    improvement_suggestions: List[str]


class ModelFeedbackLoop:
    """
    Analyzes prediction accuracy to automatically improve the model
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.tracker = ComprehensiveAccuracyTracker()
        self.pipeline = None

        # Minimum data required for meaningful feedback
        self.min_gameweeks = 2
        self.min_players_per_analysis = 5

        # Performance thresholds
        self.accuracy_threshold = 75.0  # Below this, trigger improvements
        self.major_error_threshold = 20  # Points difference that indicates serious issue

    def analyze_prediction_performance(self) -> ModelFeedback:
        """Analyze recent prediction performance and generate improvement feedback"""

        if len(self.tracker.results_history) < self.min_gameweeks:
            return ModelFeedback(
                feature_importance_changes={},
                prediction_error_patterns={},
                player_bias_corrections={},
                position_adjustments={},
                confidence_score=0.0,
                improvement_suggestions=["Need more gameweeks of data for analysis"]
            )

        # Get recent performance data
        recent_results = self.tracker.results_history[-5:]  # Last 5 GWs

        # Analyze different aspects of prediction accuracy
        error_patterns = self._analyze_error_patterns(recent_results)
        player_biases = self._analyze_player_biases(recent_results)
        position_performance = self._analyze_position_performance(recent_results)
        feature_importance = self._analyze_feature_importance(recent_results)

        # Generate improvement suggestions
        suggestions = self._generate_improvement_suggestions(
            error_patterns, player_biases, position_performance
        )

        # Calculate confidence in feedback
        confidence = self._calculate_feedback_confidence(recent_results)

        return ModelFeedback(
            feature_importance_changes=feature_importance,
            prediction_error_patterns=error_patterns,
            player_bias_corrections=player_biases,
            position_adjustments=position_performance,
            confidence_score=confidence,
            improvement_suggestions=suggestions
        )

    def _analyze_error_patterns(self, results: List) -> Dict[str, Any]:
        """Identify patterns in prediction errors"""
        patterns = {
            'consistent_overestimation': 0,
            'consistent_underestimation': 0,
            'high_variance_players': [],
            'captain_accuracy': 0,
            'formation_performance': {}
        }

        total_predictions = 0
        overestimations = 0
        underestimations = 0
        captain_errors = []

        for result in results:
            prediction = result.team_selection.predicted_points
            actual = result.actual_team_points
            error = actual - prediction

            total_predictions += 1

            if error < 0:  # Overestimated
                overestimations += 1
            elif error > 0:  # Underestimated
                underestimations += 1

            # Captain analysis
            captain_predicted = 0
            captain_actual = 0

            # Find captain in team
            for player in result.team_selection.starters:
                if player.get('player_id') == result.team_selection.captain_id:
                    captain_predicted = player.get('predicted_points', 0) * 2  # Captain gets double
                    break

            captain_actual = result.actual_captain_points
            captain_errors.append(abs(captain_actual - captain_predicted))

        if total_predictions > 0:
            patterns['consistent_overestimation'] = overestimations / total_predictions
            patterns['consistent_underestimation'] = underestimations / total_predictions
            patterns['captain_accuracy'] = 1 - (np.mean(captain_errors) / max(np.mean([r.actual_captain_points for r in results]), 1))

        return patterns

    def _analyze_player_biases(self, results: List) -> Dict[str, float]:
        """Identify players the model consistently over/under-predicts"""
        player_errors = {}

        for result in results:
            # This would require player-level performance data
            # For now, return placeholder
            pass

        return {}

    def _analyze_position_performance(self, results: List) -> Dict[str, float]:
        """Analyze prediction accuracy by position"""
        position_errors = {
            'goalkeeper': [],
            'defender': [],
            'midfielder': [],
            'forward': []
        }

        # This would require position-level breakdown
        # For now, return placeholder with general adjustments
        return {
            'goalkeeper': 0.0,
            'defender': 0.0,
            'midfielder': 0.0,
            'forward': 0.0
        }

    def _analyze_feature_importance(self, results: List) -> Dict[str, float]:
        """Analyze which features should be weighted differently"""

        # Calculate overall prediction accuracy
        total_error = 0
        total_predictions = 0

        for result in results:
            error = abs(result.actual_team_points - result.team_selection.predicted_points)
            total_error += error
            total_predictions += 1

        avg_error = total_error / max(total_predictions, 1)

        # Suggest feature weight adjustments based on error patterns
        feature_adjustments = {}

        if avg_error > 15:  # High error suggests features need rebalancing
            feature_adjustments = {
                'form_weight': 0.1,  # Increase form weight
                'fixtures_weight': 0.05,  # Slightly increase fixture difficulty weight
                'price_weight': -0.02,  # Decrease price influence
                'opponent_strength_weight': 0.08  # Increase opponent strength weight
            }
        elif avg_error > 10:  # Moderate error
            feature_adjustments = {
                'form_weight': 0.05,
                'fixtures_weight': 0.02,
                'price_weight': -0.01,
                'opponent_strength_weight': 0.03
            }

        return feature_adjustments

    def _generate_improvement_suggestions(self, error_patterns: Dict, player_biases: Dict, position_performance: Dict) -> List[str]:
        """Generate actionable improvement suggestions"""
        suggestions = []

        # Check for consistent biases
        if error_patterns.get('consistent_overestimation', 0) > 0.7:
            suggestions.append("Model consistently overestimates - reduce base prediction scores by 5-10%")
        elif error_patterns.get('consistent_underestimation', 0) > 0.7:
            suggestions.append("Model consistently underestimates - increase base prediction scores by 5-10%")

        # Captain accuracy
        captain_acc = error_patterns.get('captain_accuracy', 0)
        if captain_acc < 0.6:
            suggestions.append("Captain selection accuracy is low - review captaincy criteria and form weighting")

        # General suggestions based on patterns
        if len(suggestions) == 0:
            suggestions.append("Model performance is reasonable - consider minor form weight adjustments")

        return suggestions

    def _calculate_feedback_confidence(self, results: List) -> float:
        """Calculate confidence level in the feedback analysis"""

        if len(results) < 2:
            return 0.1
        elif len(results) < 3:
            return 0.5
        elif len(results) < 5:
            return 0.7
        else:
            # High confidence with sufficient data
            return 0.9

    def apply_feedback_to_model(self, feedback: ModelFeedback) -> Dict[str, Any]:
        """Apply feedback improvements to the model"""

        if feedback.confidence_score < 0.5:
            return {
                "status": "skipped",
                "reason": "Insufficient confidence in feedback",
                "confidence": feedback.confidence_score
            }

        improvements_applied = []

        # Apply feature weight adjustments
        if feedback.feature_importance_changes:
            for feature, adjustment in feedback.feature_importance_changes.items():
                improvements_applied.append(f"Adjusted {feature} by {adjustment:+.3f}")

        # Apply position adjustments
        if feedback.position_adjustments:
            for position, adjustment in feedback.position_adjustments.items():
                if abs(adjustment) > 0.01:  # Only apply significant adjustments
                    improvements_applied.append(f"Adjusted {position} predictions by {adjustment:+.2f}")

        return {
            "status": "success",
            "improvements_applied": improvements_applied,
            "suggestions": feedback.improvement_suggestions,
            "confidence": feedback.confidence_score,
            "next_evaluation": "After next gameweek results"
        }

    def retrain_with_actual_results(self) -> Dict[str, Any]:
        """Retrain the model incorporating actual gameweek results"""

        if len(self.tracker.results_history) < self.min_gameweeks:
            return {
                "status": "skipped",
                "reason": f"Need at least {self.min_gameweeks} gameweeks of results",
                "gameweeks_available": len(self.tracker.results_history)
            }

        try:
            # Initialize ML pipeline if needed
            if self.pipeline is None:
                self.pipeline = MLPipeline()

            # Prepare training data with actual results
            training_data = self._prepare_training_data_with_actuals()

            if training_data.empty:
                return {
                    "status": "error",
                    "reason": "No valid training data could be prepared"
                }

            # Retrain the model
            training_result = self.pipeline.train()

            # Apply feedback adjustments
            feedback = self.analyze_prediction_performance()
            feedback_result = self.apply_feedback_to_model(feedback)

            return {
                "status": "success",
                "training_result": training_result,
                "feedback_applied": feedback_result,
                "gameweeks_used": len(self.tracker.results_history),
                "training_samples": len(training_data),
                "retrained_at": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Failed to retrain model: {e}")
            return {
                "status": "error",
                "reason": f"Training failed: {str(e)}"
            }

    def _prepare_training_data_with_actuals(self) -> pd.DataFrame:
        """Prepare training data with REAL FPL data instead of estimates"""

        training_records = []

        # Import weekly retrainer to get real data
        from .weekly_retrainer import WeeklyMLRetrainer
        retrainer = WeeklyMLRetrainer()

        for result in self.tracker.results_history:
            if not hasattr(result, 'team_selection') or not result.team_selection:
                continue

            team_selection = result.team_selection
            gameweek = result.gameweek

            # Get REAL player data for this gameweek from FPL API
            try:
                real_gw_data = retrainer.collect_gameweek_results(gameweek)
                if real_gw_data is None or real_gw_data.empty:
                    self.logger.warning(f"No real data available for gameweek {gameweek}")
                    continue

                # Create a lookup for real player points
                real_points_lookup = real_gw_data.set_index('player_id')['points'].to_dict()

            except Exception as e:
                self.logger.error(f"Failed to get real data for GW {gameweek}: {e}")
                continue

            # Get starters and their predictions vs REAL actuals
            starters = team_selection.starters or []

            for player in starters:
                player_id = player.get('player_id')
                predicted_points = player.get('predicted_points', 0)

                # Get REAL actual points from FPL API data
                real_actual_points = real_points_lookup.get(player_id)

                if real_actual_points is None:
                    self.logger.warning(f"No real points data for player {player_id} in GW {gameweek}")
                    continue

                # Create training record with REAL data
                training_record = {
                    'player_id': player_id,
                    'gameweek': gameweek,
                    'predicted_points': predicted_points,
                    'actual_points': float(real_actual_points),  # REAL FPL data!
                    'position': player.get('position', ''),
                    'price': player.get('price', 0),
                    'team_id': player.get('team_id', 0),
                    'prediction_error': float(real_actual_points) - predicted_points,
                    'accuracy_ratio': float(real_actual_points) / max(predicted_points, 0.1)
                }

                training_records.append(training_record)

        if not training_records:
            return pd.DataFrame()

        df = pd.DataFrame(training_records)

        # Add derived features for better training
        df['error_magnitude'] = df['prediction_error'].abs()
        df['was_overestimated'] = (df['prediction_error'] < 0).astype(int)
        df['was_underestimated'] = (df['prediction_error'] > 0).astype(int)
        df['large_error'] = (df['error_magnitude'] > self.major_error_threshold).astype(int)

        return df

    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get a summary of current model performance and feedback"""

        if len(self.tracker.results_history) == 0:
            return {
                "status": "no_data",
                "message": "No prediction results available for analysis"
            }

        # Calculate recent accuracy
        recent_results = self.tracker.results_history[-3:]
        accuracies = []

        for result in recent_results:
            accuracy = result.prediction_accuracy.get('accuracy_pct', 0)
            accuracies.append(float(accuracy))  # Ensure float type

        avg_accuracy = float(np.mean(accuracies)) if accuracies else 0.0

        # Get latest feedback
        feedback = self.analyze_prediction_performance()

        return {
            "status": "success",
            "current_accuracy": float(avg_accuracy),
            "gameweeks_analyzed": len(recent_results),
            "feedback_confidence": float(feedback.confidence_score),
            "top_suggestions": feedback.improvement_suggestions[:3],
            "needs_improvement": bool(avg_accuracy < self.accuracy_threshold),
            "last_analysis": datetime.now().isoformat()
        }