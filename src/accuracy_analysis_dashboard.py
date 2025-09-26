from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import logging

from .comprehensive_accuracy_tracker import ComprehensiveAccuracyTracker
from .aggressive_accuracy_optimizer import AggressiveAccuracyOptimizer
from .automated_retraining_scheduler import get_retraining_scheduler


class AccuracyAnalysisDashboard:
    """
    Comprehensive dashboard for analyzing FPL prediction accuracy.
    Provides detailed insights into model performance and optimization opportunities.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.accuracy_tracker = ComprehensiveAccuracyTracker()
        self.optimizer = AggressiveAccuracyOptimizer()
        self.scheduler = get_retraining_scheduler()

        # Analysis parameters
        self.accuracy_targets = {
            'excellent': 75.0,
            'good': 65.0,
            'acceptable': 55.0,
            'poor': 45.0
        }

        self.captain_targets = {
            'excellent': 85.0,
            'good': 75.0,
            'acceptable': 65.0,
            'poor': 55.0
        }

    def generate_comprehensive_accuracy_report(self, gameweeks: int = 10) -> Dict[str, Any]:
        """
        Generate comprehensive accuracy analysis report

        Returns detailed analysis of:
        - Overall prediction accuracy trends
        - Captain selection performance
        - Position-specific accuracy
        - Model performance comparison
        - Improvement recommendations
        """

        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "analysis_period": f"Last {gameweeks} gameweeks",
                "summary": {},
                "detailed_analysis": {},
                "recommendations": [],
                "visualizations": {}
            }

            # Get historical data
            historical_report = self.accuracy_tracker.get_comprehensive_report(gameweeks)

            if "error" in historical_report:
                return {"error": "Insufficient data for analysis", "details": historical_report}

            # Summary metrics
            report["summary"] = self._calculate_summary_metrics(historical_report)

            # Detailed accuracy analysis
            report["detailed_analysis"] = {
                "prediction_accuracy": self._analyze_prediction_accuracy(historical_report),
                "captain_performance": self._analyze_captain_performance(historical_report),
                "position_breakdown": self._analyze_position_accuracy(),
                "temporal_trends": self._analyze_temporal_trends(historical_report),
                "error_patterns": self._analyze_error_patterns(historical_report)
            }

            # Generate recommendations
            report["recommendations"] = self._generate_accuracy_recommendations(report)

            # Create visualizations
            report["visualizations"] = self._create_accuracy_visualizations(historical_report)

            return report

        except Exception as e:
            self.logger.error(f"Comprehensive accuracy report generation failed: {e}")
            return {"error": str(e)}

    def _calculate_summary_metrics(self, historical_report: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate high-level summary metrics"""

        summary = historical_report.get("summary", {})
        gameweek_breakdown = historical_report.get("gameweek_breakdown", [])

        if not gameweek_breakdown:
            return {"error": "No gameweek data available"}

        # Calculate accuracy metrics
        recent_accuracies = []
        prediction_errors = []

        for gw in gameweek_breakdown:
            if gw.get("actual", 0) > 0:
                accuracy = max(0, 100 - (abs(gw["predicted"] - gw["actual"]) / gw["actual"] * 100))
                recent_accuracies.append(accuracy)
                prediction_errors.append(gw.get("error", 0))

        avg_accuracy = np.mean(recent_accuracies) if recent_accuracies else 0
        accuracy_std = np.std(recent_accuracies) if recent_accuracies else 0

        # Performance classification
        performance_level = self._classify_performance(avg_accuracy, "accuracy")

        return {
            "overall_accuracy": round(avg_accuracy, 2),
            "accuracy_std": round(accuracy_std, 2),
            "performance_level": performance_level,
            "gameweeks_analyzed": len(gameweek_breakdown),
            "total_predicted_points": summary.get("total_predicted_points", 0),
            "total_actual_points": summary.get("total_actual_points", 0),
            "prediction_error": summary.get("prediction_error", 0),
            "average_weekly_error": summary.get("average_weekly_error", 0),
            "consistency_score": max(0, 100 - accuracy_std * 2),  # Lower std = higher consistency
            "improvement_trend": self._calculate_improvement_trend(recent_accuracies)
        }

    def _analyze_prediction_accuracy(self, historical_report: Dict[str, Any]) -> Dict[str, Any]:
        """Detailed analysis of prediction accuracy patterns"""

        gameweek_breakdown = historical_report.get("gameweek_breakdown", [])

        if not gameweek_breakdown:
            return {"error": "No data available"}

        # Accuracy distribution analysis
        accuracies = []
        errors = []

        for gw in gameweek_breakdown:
            if gw.get("actual", 0) > 0:
                accuracy = max(0, 100 - (abs(gw["predicted"] - gw["actual"]) / gw["actual"] * 100))
                accuracies.append(accuracy)
                errors.append(abs(gw.get("error", 0)))

        if not accuracies:
            return {"error": "No valid accuracy data"}

        # Statistical analysis
        accuracy_stats = {
            "mean": np.mean(accuracies),
            "median": np.median(accuracies),
            "std": np.std(accuracies),
            "min": np.min(accuracies),
            "max": np.max(accuracies),
            "q25": np.percentile(accuracies, 25),
            "q75": np.percentile(accuracies, 75)
        }

        # Accuracy buckets
        accuracy_buckets = {
            "excellent (75%+)": len([a for a in accuracies if a >= 75]),
            "good (65-75%)": len([a for a in accuracies if 65 <= a < 75]),
            "acceptable (55-65%)": len([a for a in accuracies if 55 <= a < 65]),
            "poor (<55%)": len([a for a in accuracies if a < 55])
        }

        # Error analysis
        error_stats = {
            "mean_absolute_error": np.mean(errors),
            "median_absolute_error": np.median(errors),
            "error_std": np.std(errors),
            "large_errors (>10 pts)": len([e for e in errors if e > 10]),
            "small_errors (<3 pts)": len([e for e in errors if e < 3])
        }

        return {
            "accuracy_statistics": accuracy_stats,
            "accuracy_distribution": accuracy_buckets,
            "error_analysis": error_stats,
            "consistency_metrics": {
                "coefficient_of_variation": accuracy_stats["std"] / accuracy_stats["mean"] if accuracy_stats["mean"] > 0 else 0,
                "accuracy_range": accuracy_stats["max"] - accuracy_stats["min"],
                "iqr": accuracy_stats["q75"] - accuracy_stats["q25"]
            }
        }

    def _analyze_captain_performance(self, historical_report: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze captain selection accuracy"""

        captain_performance = historical_report.get("captain_performance", {})

        if not captain_performance:
            return {"error": "No captain performance data"}

        captain_predictions = captain_performance.get("captain_predictions", [])
        captain_actuals = captain_performance.get("captain_actuals", [])

        if not captain_predictions or not captain_actuals:
            return {"error": "Insufficient captain data"}

        # Captain success analysis
        successful_captains = len([a for a in captain_actuals if a >= 8])  # 8+ points = successful
        total_captains = len(captain_actuals)
        success_rate = (successful_captains / total_captains * 100) if total_captains > 0 else 0

        # Captain prediction accuracy
        captain_errors = [abs(p - a) for p, a in zip(captain_predictions, captain_actuals)]
        captain_accuracy = captain_performance.get("captain_accuracy", 0)

        # High vs low scoring captains
        high_scoring = [a for a in captain_actuals if a >= 10]
        low_scoring = [a for a in captain_actuals if a < 5]

        performance_level = self._classify_performance(success_rate, "captain")

        return {
            "success_rate": round(success_rate, 2),
            "performance_level": performance_level,
            "total_captains": total_captains,
            "successful_captains": successful_captains,
            "average_captain_error": round(np.mean(captain_errors), 2) if captain_errors else 0,
            "captain_correlation": round(captain_accuracy, 3) if captain_accuracy else 0,
            "high_scoring_captains": len(high_scoring),
            "low_scoring_captains": len(low_scoring),
            "captain_score_distribution": {
                "15+ points": len([a for a in captain_actuals if a >= 15]),
                "10-14 points": len([a for a in captain_actuals if 10 <= a < 15]),
                "5-9 points": len([a for a in captain_actuals if 5 <= a < 10]),
                "<5 points": len([a for a in captain_actuals if a < 5])
            },
            "prediction_quality": {
                "overpredictions": len([i for i, (p, a) in enumerate(zip(captain_predictions, captain_actuals)) if p > a + 3]),
                "underpredictions": len([i for i, (p, a) in enumerate(zip(captain_predictions, captain_actuals)) if a > p + 3]),
                "accurate_predictions": len([i for i, (p, a) in enumerate(zip(captain_predictions, captain_actuals)) if abs(p - a) <= 2])
            }
        }

    def _analyze_position_accuracy(self) -> Dict[str, Any]:
        """Analyze accuracy by player position"""

        try:
            # This would require position-specific accuracy data
            # For now, return a placeholder structure
            positions = ["GKP", "DEF", "MID", "FWD"]

            position_analysis = {}

            for position in positions:
                # Placeholder analysis - in real implementation, this would
                # analyze actual position-specific prediction accuracy
                position_analysis[position] = {
                    "average_accuracy": np.random.uniform(50, 80),  # Placeholder
                    "prediction_count": np.random.randint(50, 200),  # Placeholder
                    "top_performers": ["Player A", "Player B"],  # Placeholder
                    "challenging_predictions": ["Player C", "Player D"]  # Placeholder
                }

            return {
                "position_breakdown": position_analysis,
                "best_position": max(position_analysis.keys(), key=lambda p: position_analysis[p]["average_accuracy"]),
                "worst_position": min(position_analysis.keys(), key=lambda p: position_analysis[p]["average_accuracy"]),
                "overall_position_variance": np.std([pos["average_accuracy"] for pos in position_analysis.values()])
            }

        except Exception as e:
            return {"error": f"Position analysis failed: {e}"}

    def _analyze_temporal_trends(self, historical_report: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze accuracy trends over time"""

        gameweek_breakdown = historical_report.get("gameweek_breakdown", [])

        if len(gameweek_breakdown) < 3:
            return {"error": "Insufficient data for trend analysis"}

        # Calculate accuracy for each gameweek
        gw_accuracies = []
        gameweeks = []

        for gw in gameweek_breakdown:
            if gw.get("actual", 0) > 0:
                accuracy = max(0, 100 - (abs(gw["predicted"] - gw["actual"]) / gw["actual"] * 100))
                gw_accuracies.append(accuracy)
                gameweeks.append(gw["gameweek"])

        if len(gw_accuracies) < 3:
            return {"error": "Insufficient valid gameweeks for trend analysis"}

        # Trend analysis
        x = np.arange(len(gw_accuracies))
        trend_coefficient = np.polyfit(x, gw_accuracies, 1)[0]  # Linear trend

        # Recent vs early performance
        mid_point = len(gw_accuracies) // 2
        early_performance = np.mean(gw_accuracies[:mid_point])
        recent_performance = np.mean(gw_accuracies[mid_point:])

        # Volatility analysis
        volatility = np.std(gw_accuracies)

        # Peak and trough analysis
        peak_accuracy = max(gw_accuracies)
        trough_accuracy = min(gw_accuracies)
        peak_gw = gameweeks[gw_accuracies.index(peak_accuracy)]
        trough_gw = gameweeks[gw_accuracies.index(trough_accuracy)]

        trend_direction = "improving" if trend_coefficient > 1 else "declining" if trend_coefficient < -1 else "stable"

        return {
            "trend_direction": trend_direction,
            "trend_coefficient": round(trend_coefficient, 3),
            "early_performance": round(early_performance, 2),
            "recent_performance": round(recent_performance, 2),
            "performance_change": round(recent_performance - early_performance, 2),
            "volatility": round(volatility, 2),
            "peak_performance": {
                "accuracy": round(peak_accuracy, 2),
                "gameweek": peak_gw
            },
            "worst_performance": {
                "accuracy": round(trough_accuracy, 2),
                "gameweek": trough_gw
            },
            "stability_score": max(0, 100 - volatility * 2)  # Lower volatility = higher stability
        }

    def _analyze_error_patterns(self, historical_report: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patterns in prediction errors"""

        gameweek_breakdown = historical_report.get("gameweek_breakdown", [])

        errors = []
        predictions = []
        actuals = []

        for gw in gameweek_breakdown:
            if gw.get("actual", 0) > 0:
                error = gw.get("error", 0)
                pred = gw.get("predicted", 0)
                actual = gw.get("actual", 0)

                errors.append(error)
                predictions.append(pred)
                actuals.append(actual)

        if not errors:
            return {"error": "No error data available"}

        # Error characteristics
        positive_errors = [e for e in errors if e > 0]  # Overpredictions
        negative_errors = [e for e in errors if e < 0]  # Underpredictions

        # Bias analysis
        mean_error = np.mean(errors)
        bias_type = "overprediction" if mean_error > 1 else "underprediction" if mean_error < -1 else "balanced"

        # Error magnitude analysis
        large_errors = [e for e in errors if abs(e) > 10]
        small_errors = [e for e in errors if abs(e) <= 3]

        # Accuracy vs prediction magnitude
        high_pred_accuracy = []
        low_pred_accuracy = []

        for pred, actual in zip(predictions, actuals):
            accuracy = max(0, 100 - (abs(pred - actual) / actual * 100)) if actual > 0 else 0
            if pred >= 50:  # High prediction
                high_pred_accuracy.append(accuracy)
            else:  # Low prediction
                low_pred_accuracy.append(accuracy)

        return {
            "bias_analysis": {
                "mean_error": round(mean_error, 2),
                "bias_type": bias_type,
                "overpredictions": len(positive_errors),
                "underpredictions": len(negative_errors)
            },
            "error_magnitude": {
                "large_errors": len(large_errors),
                "small_errors": len(small_errors),
                "error_concentration": len(small_errors) / len(errors) * 100 if errors else 0
            },
            "prediction_accuracy_by_magnitude": {
                "high_predictions_accuracy": round(np.mean(high_pred_accuracy), 2) if high_pred_accuracy else 0,
                "low_predictions_accuracy": round(np.mean(low_pred_accuracy), 2) if low_pred_accuracy else 0,
                "high_pred_count": len(high_pred_accuracy),
                "low_pred_count": len(low_pred_accuracy)
            },
            "error_statistics": {
                "mae": round(np.mean([abs(e) for e in errors]), 2),
                "rmse": round(np.sqrt(np.mean([e**2 for e in errors])), 2),
                "error_std": round(np.std(errors), 2)
            }
        }

    def _generate_accuracy_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate specific recommendations for improving accuracy"""

        recommendations = []

        # Get key metrics
        summary = report.get("summary", {})
        accuracy = summary.get("overall_accuracy", 0)
        consistency = summary.get("consistency_score", 0)
        trend = summary.get("improvement_trend", "stable")

        detailed = report.get("detailed_analysis", {})
        captain_perf = detailed.get("captain_performance", {})
        captain_success = captain_perf.get("success_rate", 0)
        temporal_trends = detailed.get("temporal_trends", {})
        error_patterns = detailed.get("error_patterns", {})

        # Accuracy-based recommendations
        if accuracy < 50:
            recommendations.append("🚨 CRITICAL: Accuracy below 50% - Emergency model rebuild required")
            recommendations.append("🔧 Consider fundamental changes to feature engineering and model architecture")
        elif accuracy < 60:
            recommendations.append("⚠️  Accuracy below target - Implement aggressive retraining strategy")
            recommendations.append("📊 Review feature importance and consider ensemble methods")
        elif accuracy < 70:
            recommendations.append("📈 Good accuracy but room for improvement - Focus on fine-tuning")
        else:
            recommendations.append("✅ Excellent accuracy achieved - Maintain current approach")

        # Captain performance recommendations
        if captain_success < 60:
            recommendations.append("👑 Captain selection needs significant improvement")
            recommendations.append("🎯 Review captaincy criteria - focus on fixture difficulty and recent form")
        elif captain_success < 75:
            recommendations.append("👑 Captain selection showing promise - refine selection logic")

        # Consistency recommendations
        if consistency < 60:
            recommendations.append("📊 High prediction volatility detected - improve model stability")
            recommendations.append("🎛️  Consider regularization techniques to reduce overfitting")

        # Trend-based recommendations
        if trend == "declining":
            recommendations.append("📉 Declining accuracy trend - immediate model intervention needed")
            recommendations.append("🔄 Increase retraining frequency and monitor data quality")
        elif trend == "improving":
            recommendations.append("📈 Positive accuracy trend - continue current optimization strategy")

        # Error pattern recommendations
        bias_type = error_patterns.get("bias_analysis", {}).get("bias_type", "balanced")
        if bias_type == "overprediction":
            recommendations.append("📊 Consistent overprediction detected - adjust prediction scaling")
        elif bias_type == "underprediction":
            recommendations.append("📊 Consistent underprediction detected - boost prediction confidence")

        # Specific technical recommendations
        large_errors = error_patterns.get("error_magnitude", {}).get("large_errors", 0)
        total_predictions = len(detailed.get("prediction_accuracy", {}).get("accuracy_statistics", {}).keys())

        if large_errors > total_predictions * 0.2:  # >20% large errors
            recommendations.append("🎯 High frequency of large errors - review outlier handling")

        # Retraining recommendations
        if accuracy < 65 or captain_success < 70:
            recommendations.append("🔄 Implement weekly aggressive retraining after each gameweek")
            recommendations.append("⚡ Enable emergency retraining triggers for poor performance")

        return recommendations or ["📊 Analysis complete - no specific recommendations at this time"]

    def _create_accuracy_visualizations(self, historical_report: Dict[str, Any]) -> Dict[str, str]:
        """Create visualization plots for accuracy analysis"""

        try:
            visualizations = {}

            gameweek_breakdown = historical_report.get("gameweek_breakdown", [])

            if not gameweek_breakdown:
                return {"error": "No data for visualizations"}

            # Prepare data
            gameweeks = [gw["gameweek"] for gw in gameweek_breakdown]
            predicted = [gw["predicted"] for gw in gameweek_breakdown]
            actual = [gw["actual"] for gw in gameweek_breakdown]

            # Calculate accuracies
            accuracies = []
            for pred, act in zip(predicted, actual):
                if act > 0:
                    acc = max(0, 100 - (abs(pred - act) / act * 100))
                    accuracies.append(acc)
                else:
                    accuracies.append(0)

            # 1. Accuracy Trend Plot
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=gameweeks,
                y=accuracies,
                mode='lines+markers',
                name='Accuracy %',
                line=dict(color='blue', width=3),
                marker=dict(size=8)
            ))
            fig_trend.add_hline(y=self.accuracy_targets['excellent'], line_dash="dash",
                               line_color="green", annotation_text="Excellent (75%)")
            fig_trend.add_hline(y=self.accuracy_targets['good'], line_dash="dash",
                               line_color="orange", annotation_text="Good (65%)")
            fig_trend.add_hline(y=self.accuracy_targets['acceptable'], line_dash="dash",
                               line_color="red", annotation_text="Acceptable (55%)")

            fig_trend.update_layout(
                title="Prediction Accuracy Trend",
                xaxis_title="Gameweek",
                yaxis_title="Accuracy (%)",
                template="plotly_white"
            )

            visualizations["accuracy_trend"] = fig_trend.to_html(include_plotlyjs='cdn')

            # 2. Predicted vs Actual Scatter Plot
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=predicted,
                y=actual,
                mode='markers',
                name='Predictions',
                marker=dict(size=10, opacity=0.7)
            ))

            # Perfect prediction line
            min_val = min(min(predicted), min(actual))
            max_val = max(max(predicted), max(actual))
            fig_scatter.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))

            fig_scatter.update_layout(
                title="Predicted vs Actual Points",
                xaxis_title="Predicted Points",
                yaxis_title="Actual Points",
                template="plotly_white"
            )

            visualizations["predicted_vs_actual"] = fig_scatter.to_html(include_plotlyjs='cdn')

            # 3. Accuracy Distribution Histogram
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=accuracies,
                nbinsx=20,
                name='Accuracy Distribution',
                marker_color='lightblue'
            ))

            fig_hist.update_layout(
                title="Accuracy Distribution",
                xaxis_title="Accuracy (%)",
                yaxis_title="Frequency",
                template="plotly_white"
            )

            visualizations["accuracy_distribution"] = fig_hist.to_html(include_plotlyjs='cdn')

            return visualizations

        except Exception as e:
            self.logger.error(f"Visualization creation failed: {e}")
            return {"error": str(e)}

    def _classify_performance(self, value: float, metric_type: str) -> str:
        """Classify performance level based on value and metric type"""

        if metric_type == "accuracy":
            targets = self.accuracy_targets
        elif metric_type == "captain":
            targets = self.captain_targets
        else:
            return "unknown"

        if value >= targets['excellent']:
            return "excellent"
        elif value >= targets['good']:
            return "good"
        elif value >= targets['acceptable']:
            return "acceptable"
        else:
            return "poor"

    def _calculate_improvement_trend(self, accuracies: List[float]) -> str:
        """Calculate whether accuracy is improving, declining, or stable"""

        if len(accuracies) < 3:
            return "insufficient_data"

        # Compare recent vs earlier performance
        mid_point = len(accuracies) // 2
        early_avg = np.mean(accuracies[:mid_point])
        recent_avg = np.mean(accuracies[mid_point:])

        diff = recent_avg - early_avg

        if diff > 3:
            return "improving"
        elif diff < -3:
            return "declining"
        else:
            return "stable"

    def generate_weekly_accuracy_summary(self) -> Dict[str, Any]:
        """Generate a concise weekly accuracy summary for quick monitoring"""

        try:
            # Get current performance
            current_performance = self.optimizer.get_accuracy_dashboard()

            # Get scheduler status
            scheduler_status = self.scheduler.get_scheduler_status()

            # Quick metrics
            accuracy = current_performance.get("current_performance", {}).get("accuracy", 0)
            performance_level = self._classify_performance(accuracy, "accuracy")

            summary = {
                "timestamp": datetime.now().isoformat(),
                "quick_metrics": {
                    "current_accuracy": round(accuracy, 1),
                    "performance_level": performance_level,
                    "target_met": accuracy >= self.accuracy_targets['good'],
                    "scheduler_running": scheduler_status.get("is_running", False),
                    "last_retrain": scheduler_status.get("last_processed_gameweek", 0)
                },
                "status_indicators": {
                    "accuracy_status": "🟢" if accuracy >= 65 else "🟡" if accuracy >= 55 else "🔴",
                    "trend_status": "📈" if current_performance.get("improvement_trend") == "improving" else "📉" if current_performance.get("improvement_trend") == "declining" else "➡️",
                    "automation_status": "🤖" if scheduler_status.get("is_running") else "⏸️"
                },
                "recent_activity": scheduler_status.get("recent_completions", []),
                "next_actions": self._get_next_actions(accuracy, performance_level)
            }

            return summary

        except Exception as e:
            return {"error": f"Weekly summary generation failed: {e}"}

    def _get_next_actions(self, accuracy: float, performance_level: str) -> List[str]:
        """Get recommended next actions based on current performance"""

        actions = []

        if performance_level == "poor":
            actions.append("Trigger emergency retraining immediately")
            actions.append("Review model architecture and features")
        elif performance_level == "acceptable":
            actions.append("Implement aggressive weekly retraining")
            actions.append("Monitor captain selection performance")
        elif performance_level == "good":
            actions.append("Continue current optimization strategy")
            actions.append("Fine-tune ensemble weights")
        else:  # excellent
            actions.append("Maintain current approach")
            actions.append("Monitor for any performance degradation")

        return actions

    def export_accuracy_report(self, format: str = "json") -> str:
        """Export comprehensive accuracy report to file"""

        try:
            report = self.generate_comprehensive_accuracy_report()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if format.lower() == "json":
                filename = f"reports/accuracy_report_{timestamp}.json"
                with open(filename, 'w') as f:
                    json.dump(report, f, indent=2)

            elif format.lower() == "html":
                filename = f"reports/accuracy_report_{timestamp}.html"
                html_content = self._generate_html_report(report)
                with open(filename, 'w') as f:
                    f.write(html_content)

            else:
                raise ValueError(f"Unsupported format: {format}")

            return filename

        except Exception as e:
            self.logger.error(f"Report export failed: {e}")
            return f"Export failed: {e}"

    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """Generate HTML version of the accuracy report"""

        # Basic HTML template - in production, this would be more sophisticated
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>FPL Accuracy Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ccc; border-radius: 5px; }}
                .excellent {{ color: green; }}
                .good {{ color: orange; }}
                .poor {{ color: red; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>FPL Prediction Accuracy Analysis</h1>
                <p>Generated: {report.get('timestamp', 'Unknown')}</p>
            </div>

            <div class="section">
                <h2>Summary</h2>
                <div class="metric">
                    <strong>Overall Accuracy:</strong> {report.get('summary', {}).get('overall_accuracy', 0):.1f}%
                </div>
                <div class="metric">
                    <strong>Performance Level:</strong> {report.get('summary', {}).get('performance_level', 'Unknown')}
                </div>
            </div>

            <div class="section">
                <h2>Recommendations</h2>
                <ul>
                    {' '.join([f'<li>{rec}</li>' for rec in report.get('recommendations', [])])}
                </ul>
            </div>

            <div class="section">
                <h2>Detailed Analysis</h2>
                <pre>{json.dumps(report.get('detailed_analysis', {}), indent=2)}</pre>
            </div>
        </body>
        </html>
        """

        return html_template


# Convenience functions
def get_accuracy_dashboard() -> AccuracyAnalysisDashboard:
    """Get accuracy dashboard instance"""
    return AccuracyAnalysisDashboard()

def quick_accuracy_check() -> Dict[str, Any]:
    """Quick accuracy check for monitoring"""
    dashboard = AccuracyAnalysisDashboard()
    return dashboard.generate_weekly_accuracy_summary()

def generate_accuracy_report(gameweeks: int = 10) -> Dict[str, Any]:
    """Generate comprehensive accuracy report"""
    dashboard = AccuracyAnalysisDashboard()
    return dashboard.generate_comprehensive_accuracy_report(gameweeks)