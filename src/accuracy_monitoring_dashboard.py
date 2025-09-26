"""
Real-time Accuracy Monitoring Dashboard for Maximum Accuracy System
Provides comprehensive tracking, visualization, and alerting for FPL prediction accuracy
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import json
from enum import Enum
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

from .fpl_client import FPLClient
from .cache_manager import CacheManager


class AccuracyAlert(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    SUCCESS = "success"


@dataclass
class AccuracySnapshot:
    """Snapshot of accuracy metrics at a point in time"""
    timestamp: datetime
    gameweek: int
    overall_accuracy: float
    position_accuracies: Dict[str, float]
    captain_accuracy: float
    differential_accuracy: float
    value_pick_accuracy: float

    # Model performance
    model_confidence: float
    prediction_variance: float
    ensemble_stability: float

    # Target progress
    target_accuracy: float
    progress_percentage: float
    weeks_behind_schedule: int

    # Alerts
    active_alerts: List[Dict[str, Any]]


@dataclass
class PerformanceTrend:
    """Performance trend analysis"""
    period_days: int
    accuracy_trend: float  # Change in accuracy
    confidence_trend: float
    stability_trend: float
    prediction_quality_trend: float
    trend_direction: str  # "improving", "declining", "stable"


class AccuracyMonitoringDashboard:
    """
    Comprehensive real-time accuracy monitoring and alerting system
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.fpl = FPLClient()
        self.cache_manager = CacheManager()

        # Monitoring configuration
        self.target_accuracy = 0.75  # 75% target
        self.warning_threshold = 0.65  # Below 65% triggers warning
        self.critical_threshold = 0.55  # Below 55% triggers critical alert

        # Data storage
        self.snapshots_file = Path("data/monitoring/accuracy_snapshots.jsonl")
        self.alerts_file = Path("data/monitoring/alerts_history.jsonl")
        self.dashboard_cache = Path("data/monitoring/dashboard_cache.json")

        # Create directories
        self.snapshots_file.parent.mkdir(parents=True, exist_ok=True)

        # Historical data
        self.accuracy_snapshots: List[AccuracySnapshot] = []
        self.performance_trends: List[PerformanceTrend] = []

        # Setup logging
        self._setup_logging()

        # Load historical data
        self._load_historical_data()

    def _setup_logging(self):
        """Setup monitoring dashboard logging"""
        log_file = Path("logs/accuracy_monitoring.log")
        log_file.parent.mkdir(exist_ok=True)

        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def _load_historical_data(self):
        """Load historical accuracy data"""
        try:
            if self.snapshots_file.exists():
                with open(self.snapshots_file, 'r') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                            snapshot = AccuracySnapshot(**data)
                            self.accuracy_snapshots.append(snapshot)
                        except Exception as e:
                            self.logger.warning(f"Could not load snapshot: {e}")

            self.logger.info(f"Loaded {len(self.accuracy_snapshots)} historical snapshots")

        except Exception as e:
            self.logger.warning(f"Could not load historical data: {e}")

    def capture_accuracy_snapshot(self,
                                 predictions_df: pd.DataFrame,
                                 actual_results_df: Optional[pd.DataFrame] = None,
                                 gameweek: Optional[int] = None) -> AccuracySnapshot:
        """Capture comprehensive accuracy snapshot"""
        try:
            if gameweek is None:
                gameweek = self._get_current_gameweek()

            # Calculate accuracy metrics
            accuracy_metrics = self._calculate_accuracy_metrics(predictions_df, actual_results_df)

            # Calculate model performance metrics
            model_metrics = self._calculate_model_metrics(predictions_df)

            # Calculate target progress
            progress_metrics = self._calculate_target_progress(accuracy_metrics['overall_accuracy'], gameweek)

            # Generate alerts
            alerts = self._generate_alerts(accuracy_metrics, model_metrics, progress_metrics)

            # Create snapshot
            snapshot = AccuracySnapshot(
                timestamp=datetime.now(),
                gameweek=gameweek,
                overall_accuracy=accuracy_metrics['overall_accuracy'],
                position_accuracies=accuracy_metrics['position_accuracies'],
                captain_accuracy=accuracy_metrics['captain_accuracy'],
                differential_accuracy=accuracy_metrics['differential_accuracy'],
                value_pick_accuracy=accuracy_metrics['value_pick_accuracy'],
                model_confidence=model_metrics['confidence'],
                prediction_variance=model_metrics['variance'],
                ensemble_stability=model_metrics['stability'],
                target_accuracy=self.target_accuracy,
                progress_percentage=progress_metrics['progress_percentage'],
                weeks_behind_schedule=progress_metrics['weeks_behind'],
                active_alerts=alerts
            )

            # Save snapshot
            self._save_snapshot(snapshot)
            self.accuracy_snapshots.append(snapshot)

            # Update trends
            self._update_performance_trends()

            self.logger.info(f"Captured accuracy snapshot for GW {gameweek}: {accuracy_metrics['overall_accuracy']:.1%}")
            return snapshot

        except Exception as e:
            self.logger.error(f"Error capturing accuracy snapshot: {e}")
            # Return empty snapshot
            return AccuracySnapshot(
                timestamp=datetime.now(),
                gameweek=gameweek or 1,
                overall_accuracy=0.0,
                position_accuracies={},
                captain_accuracy=0.0,
                differential_accuracy=0.0,
                value_pick_accuracy=0.0,
                model_confidence=0.0,
                prediction_variance=0.0,
                ensemble_stability=0.0,
                target_accuracy=self.target_accuracy,
                progress_percentage=0.0,
                weeks_behind_schedule=0,
                active_alerts=[]
            )

    def _calculate_accuracy_metrics(self,
                                   predictions_df: pd.DataFrame,
                                   actual_results_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Calculate comprehensive accuracy metrics"""
        metrics = {
            'overall_accuracy': 0.0,
            'position_accuracies': {},
            'captain_accuracy': 0.0,
            'differential_accuracy': 0.0,
            'value_pick_accuracy': 0.0
        }

        if actual_results_df is None:
            # Use historical accuracy estimation if no actual results
            if self.accuracy_snapshots:
                latest = self.accuracy_snapshots[-1]
                metrics['overall_accuracy'] = latest.overall_accuracy
                metrics['position_accuracies'] = latest.position_accuracies
            return metrics

        try:
            # Merge predictions with actual results
            merged = predictions_df.merge(
                actual_results_df,
                left_on='player_id',
                right_on='player_id',
                how='inner'
            )

            if merged.empty:
                return metrics

            # Overall accuracy (within ±2 points)
            pred_points = pd.to_numeric(merged['predicted_points'], errors='coerce').fillna(0)
            actual_points = pd.to_numeric(merged['total_points'], errors='coerce').fillna(0)
            within_range = np.abs(pred_points - actual_points) <= 2
            metrics['overall_accuracy'] = within_range.mean()

            # Position-specific accuracy
            for position in ['GKP', 'DEF', 'MID', 'FWD']:
                pos_mask = merged['position'] == position
                if pos_mask.sum() > 0:
                    pos_accuracy = within_range[pos_mask].mean()
                    metrics['position_accuracies'][position] = pos_accuracy

            # Captain accuracy
            if 'captain_candidate' in merged.columns:
                captain_candidates = merged['captain_candidate'] == True
                if captain_candidates.sum() > 0:
                    captain_accuracy = within_range[captain_candidates].mean()
                    metrics['captain_accuracy'] = captain_accuracy

            # Differential accuracy
            if 'differential_candidate' in merged.columns:
                differentials = merged['differential_candidate'] == True
                if differentials.sum() > 0:
                    diff_accuracy = within_range[differentials].mean()
                    metrics['differential_accuracy'] = diff_accuracy

            # Value pick accuracy
            if 'value_efficiency' in merged.columns:
                value_threshold = merged['value_efficiency'].quantile(0.8)
                value_picks = merged['value_efficiency'] > value_threshold
                if value_picks.sum() > 0:
                    value_accuracy = within_range[value_picks].mean()
                    metrics['value_pick_accuracy'] = value_accuracy

        except Exception as e:
            self.logger.error(f"Error calculating accuracy metrics: {e}")

        return metrics

    def _calculate_model_metrics(self, predictions_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate model performance metrics"""
        try:
            # Model confidence
            confidence = predictions_df.get('prediction_confidence', pd.Series([0.5] * len(predictions_df))).mean()

            # Prediction variance
            pred_points = pd.to_numeric(predictions_df['predicted_points'], errors='coerce').fillna(0)
            variance = pred_points.var()

            # Ensemble stability (if multiple model predictions available)
            stability = 1.0  # Default stable

            model_cols = [col for col in predictions_df.columns if col.startswith('pred_')]
            if len(model_cols) > 1:
                model_preds = predictions_df[model_cols]
                pred_std = model_preds.std(axis=1).mean()
                stability = max(0.0, 1.0 - (pred_std / pred_points.mean()))

            return {
                'confidence': float(confidence),
                'variance': float(variance),
                'stability': float(stability)
            }

        except Exception as e:
            self.logger.error(f"Error calculating model metrics: {e}")
            return {'confidence': 0.5, 'variance': 0.0, 'stability': 1.0}

    def _calculate_target_progress(self, current_accuracy: float, gameweek: int) -> Dict[str, Any]:
        """Calculate progress toward accuracy targets"""
        # Target timeline (aggressive improvement schedule)
        target_timeline = {
            1: 0.55, 2: 0.58, 3: 0.60, 4: 0.62, 5: 0.65,
            6: 0.67, 7: 0.69, 8: 0.70, 9: 0.72, 10: 0.74,
            15: 0.75, 20: 0.76, 25: 0.77, 30: 0.78
        }

        # Get target for current gameweek
        target_for_gw = target_timeline.get(gameweek, self.target_accuracy)

        # Calculate progress
        progress_percentage = (current_accuracy / target_for_gw) * 100

        # Calculate weeks behind schedule
        weeks_behind = 0
        if current_accuracy < target_for_gw:
            # Simple estimation based on improvement rate needed
            weekly_improvement_needed = 0.02  # 2% per week
            accuracy_gap = target_for_gw - current_accuracy
            weeks_behind = int(accuracy_gap / weekly_improvement_needed)

        return {
            'progress_percentage': float(progress_percentage),
            'weeks_behind': weeks_behind,
            'target_for_gameweek': target_for_gw
        }

    def _generate_alerts(self,
                        accuracy_metrics: Dict[str, Any],
                        model_metrics: Dict[str, float],
                        progress_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate accuracy alerts based on thresholds"""
        alerts = []
        current_accuracy = accuracy_metrics['overall_accuracy']

        # Critical accuracy alert
        if current_accuracy < self.critical_threshold:
            alerts.append({
                'type': AccuracyAlert.CRITICAL.value,
                'title': 'Critical Accuracy Alert',
                'message': f'Accuracy {current_accuracy:.1%} is critically low (< {self.critical_threshold:.1%})',
                'timestamp': datetime.now().isoformat(),
                'action_required': 'Immediate model retraining and strategy adjustment needed'
            })

        # Warning accuracy alert
        elif current_accuracy < self.warning_threshold:
            alerts.append({
                'type': AccuracyAlert.WARNING.value,
                'title': 'Accuracy Warning',
                'message': f'Accuracy {current_accuracy:.1%} is below warning threshold ({self.warning_threshold:.1%})',
                'timestamp': datetime.now().isoformat(),
                'action_required': 'Consider model retraining or strategy adjustment'
            })

        # Behind schedule alert
        if progress_metrics['weeks_behind'] > 2:
            alerts.append({
                'type': AccuracyAlert.WARNING.value,
                'title': 'Behind Target Schedule',
                'message': f'{progress_metrics["weeks_behind"]} weeks behind accuracy target schedule',
                'timestamp': datetime.now().isoformat(),
                'action_required': 'Accelerate improvement strategy'
            })

        # Low model confidence alert
        if model_metrics['confidence'] < 0.5:
            alerts.append({
                'type': AccuracyAlert.WARNING.value,
                'title': 'Low Model Confidence',
                'message': f'Model confidence {model_metrics["confidence"]:.1%} is concerning',
                'timestamp': datetime.now().isoformat(),
                'action_required': 'Review feature quality and model performance'
            })

        # High prediction variance alert
        if model_metrics['variance'] > 25:
            alerts.append({
                'type': AccuracyAlert.INFO.value,
                'title': 'High Prediction Variance',
                'message': f'Prediction variance {model_metrics["variance"]:.1f} indicates uncertainty',
                'timestamp': datetime.now().isoformat(),
                'action_required': 'Monitor prediction stability'
            })

        # Target achievement alert
        if current_accuracy >= self.target_accuracy:
            alerts.append({
                'type': AccuracyAlert.SUCCESS.value,
                'title': 'Target Accuracy Achieved!',
                'message': f'Accuracy {current_accuracy:.1%} meets or exceeds target {self.target_accuracy:.1%}',
                'timestamp': datetime.now().isoformat(),
                'action_required': 'Maintain current strategy'
            })

        return alerts

    def _save_snapshot(self, snapshot: AccuracySnapshot):
        """Save accuracy snapshot to persistent storage"""
        try:
            with open(self.snapshots_file, 'a') as f:
                data = asdict(snapshot)
                data['timestamp'] = snapshot.timestamp.isoformat()
                f.write(json.dumps(data) + '\n')

        except Exception as e:
            self.logger.error(f"Could not save snapshot: {e}")

    def _update_performance_trends(self):
        """Update performance trend analysis"""
        try:
            if len(self.accuracy_snapshots) < 2:
                return

            # Calculate trends for different periods
            periods = [7, 14, 30]  # Days
            self.performance_trends = []

            for period_days in periods:
                cutoff_date = datetime.now() - timedelta(days=period_days)
                recent_snapshots = [
                    s for s in self.accuracy_snapshots
                    if s.timestamp >= cutoff_date
                ]

                if len(recent_snapshots) >= 2:
                    trend = self._calculate_trend(recent_snapshots, period_days)
                    self.performance_trends.append(trend)

        except Exception as e:
            self.logger.error(f"Error updating performance trends: {e}")

    def _calculate_trend(self, snapshots: List[AccuracySnapshot], period_days: int) -> PerformanceTrend:
        """Calculate trend for given period"""
        if len(snapshots) < 2:
            return PerformanceTrend(
                period_days=period_days,
                accuracy_trend=0.0,
                confidence_trend=0.0,
                stability_trend=0.0,
                prediction_quality_trend=0.0,
                trend_direction="stable"
            )

        # Sort by timestamp
        snapshots_sorted = sorted(snapshots, key=lambda x: x.timestamp)

        # Calculate trends
        accuracy_trend = snapshots_sorted[-1].overall_accuracy - snapshots_sorted[0].overall_accuracy
        confidence_trend = snapshots_sorted[-1].model_confidence - snapshots_sorted[0].model_confidence
        stability_trend = snapshots_sorted[-1].ensemble_stability - snapshots_sorted[0].ensemble_stability

        # Prediction quality trend (composite score)
        quality_scores = []
        for s in snapshots_sorted:
            quality = (s.overall_accuracy * 0.5 + s.model_confidence * 0.3 + s.ensemble_stability * 0.2)
            quality_scores.append(quality)

        prediction_quality_trend = quality_scores[-1] - quality_scores[0]

        # Determine trend direction
        if accuracy_trend > 0.02:  # 2% improvement
            trend_direction = "improving"
        elif accuracy_trend < -0.02:  # 2% decline
            trend_direction = "declining"
        else:
            trend_direction = "stable"

        return PerformanceTrend(
            period_days=period_days,
            accuracy_trend=float(accuracy_trend),
            confidence_trend=float(confidence_trend),
            stability_trend=float(stability_trend),
            prediction_quality_trend=float(prediction_quality_trend),
            trend_direction=trend_direction
        )

    def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate comprehensive dashboard data"""
        try:
            latest_snapshot = self.accuracy_snapshots[-1] if self.accuracy_snapshots else None

            # Current metrics
            current_metrics = {
                'overall_accuracy': latest_snapshot.overall_accuracy if latest_snapshot else 0.0,
                'target_accuracy': self.target_accuracy,
                'gameweek': latest_snapshot.gameweek if latest_snapshot else 1,
                'last_updated': latest_snapshot.timestamp.isoformat() if latest_snapshot else datetime.now().isoformat()
            }

            # Historical accuracy data
            historical_data = [
                {
                    'gameweek': s.gameweek,
                    'accuracy': s.overall_accuracy,
                    'target': s.target_accuracy,
                    'timestamp': s.timestamp.isoformat()
                }
                for s in self.accuracy_snapshots[-20:]  # Last 20 snapshots
            ]

            # Position breakdown
            position_breakdown = latest_snapshot.position_accuracies if latest_snapshot else {}

            # Active alerts
            active_alerts = latest_snapshot.active_alerts if latest_snapshot else []

            # Performance trends
            trends_data = [
                {
                    'period_days': t.period_days,
                    'accuracy_trend': t.accuracy_trend,
                    'trend_direction': t.trend_direction
                }
                for t in self.performance_trends
            ]

            # Progress metrics
            progress_data = {
                'progress_percentage': latest_snapshot.progress_percentage if latest_snapshot else 0.0,
                'weeks_behind_schedule': latest_snapshot.weeks_behind_schedule if latest_snapshot else 0,
                'on_track': (latest_snapshot.progress_percentage >= 95.0) if latest_snapshot else False
            }

            # Model health
            model_health = {
                'confidence': latest_snapshot.model_confidence if latest_snapshot else 0.0,
                'stability': latest_snapshot.ensemble_stability if latest_snapshot else 0.0,
                'variance': latest_snapshot.prediction_variance if latest_snapshot else 0.0
            }

            return {
                'current_metrics': current_metrics,
                'historical_data': historical_data,
                'position_breakdown': position_breakdown,
                'active_alerts': active_alerts,
                'performance_trends': trends_data,
                'progress_data': progress_data,
                'model_health': model_health,
                'dashboard_generated': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error generating dashboard data: {e}")
            return {
                'error': str(e),
                'dashboard_generated': datetime.now().isoformat()
            }

    def create_accuracy_visualizations(self) -> Dict[str, str]:
        """Create plotly visualizations for accuracy monitoring"""
        try:
            if not self.accuracy_snapshots:
                return {}

            # Accuracy over time chart
            df_accuracy = pd.DataFrame([
                {
                    'gameweek': s.gameweek,
                    'accuracy': s.overall_accuracy,
                    'target': s.target_accuracy,
                    'timestamp': s.timestamp
                }
                for s in self.accuracy_snapshots[-20:]
            ])

            fig_accuracy = go.Figure()
            fig_accuracy.add_trace(go.Scatter(
                x=df_accuracy['gameweek'],
                y=df_accuracy['accuracy'],
                mode='lines+markers',
                name='Actual Accuracy',
                line=dict(color='blue', width=3)
            ))
            fig_accuracy.add_trace(go.Scatter(
                x=df_accuracy['gameweek'],
                y=df_accuracy['target'],
                mode='lines',
                name='Target Accuracy',
                line=dict(color='red', dash='dash')
            ))
            fig_accuracy.update_layout(
                title='FPL Prediction Accuracy Over Time',
                xaxis_title='Gameweek',
                yaxis_title='Accuracy (%)',
                yaxis=dict(tickformat='.0%')
            )

            # Position accuracy breakdown
            if self.accuracy_snapshots:
                latest = self.accuracy_snapshots[-1]
                positions = list(latest.position_accuracies.keys())
                accuracies = list(latest.position_accuracies.values())

                fig_positions = go.Figure(data=[
                    go.Bar(x=positions, y=accuracies, marker_color='lightblue')
                ])
                fig_positions.update_layout(
                    title='Accuracy by Position (Latest)',
                    xaxis_title='Position',
                    yaxis_title='Accuracy (%)',
                    yaxis=dict(tickformat='.0%')
                )

            # Performance trends
            if self.performance_trends:
                trend_data = pd.DataFrame([
                    {
                        'period': f'{t.period_days} days',
                        'accuracy_change': t.accuracy_trend,
                        'direction': t.trend_direction
                    }
                    for t in self.performance_trends
                ])

                colors = {'improving': 'green', 'declining': 'red', 'stable': 'gray'}
                fig_trends = go.Figure(data=[
                    go.Bar(
                        x=trend_data['period'],
                        y=trend_data['accuracy_change'],
                        marker_color=[colors.get(d, 'gray') for d in trend_data['direction']]
                    )
                ])
                fig_trends.update_layout(
                    title='Performance Trends',
                    xaxis_title='Period',
                    yaxis_title='Accuracy Change',
                    yaxis=dict(tickformat='.0%')
                )

            return {
                'accuracy_timeline': fig_accuracy.to_html(include_plotlyjs='cdn'),
                'position_breakdown': fig_positions.to_html(include_plotlyjs='cdn'),
                'performance_trends': fig_trends.to_html(include_plotlyjs='cdn')
            }

        except Exception as e:
            self.logger.error(f"Error creating visualizations: {e}")
            return {}

    def get_real_time_status(self) -> Dict[str, Any]:
        """Get real-time system status"""
        latest_snapshot = self.accuracy_snapshots[-1] if self.accuracy_snapshots else None

        return {
            'system_status': 'operational',
            'last_update': latest_snapshot.timestamp.isoformat() if latest_snapshot else None,
            'current_accuracy': latest_snapshot.overall_accuracy if latest_snapshot else 0.0,
            'target_accuracy': self.target_accuracy,
            'alert_count': len(latest_snapshot.active_alerts) if latest_snapshot else 0,
            'trend_direction': self.performance_trends[-1].trend_direction if self.performance_trends else 'unknown',
            'monitoring_active': True
        }

    def _get_current_gameweek(self) -> int:
        """Get current gameweek from FPL API"""
        try:
            bs = self.fpl.bootstrap_static()
            events = bs.get('events', [])
            current_gw = next((e['id'] for e in events if e.get('is_current')), 1)
            return current_gw
        except Exception:
            return 1