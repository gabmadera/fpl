#!/usr/bin/env python3
"""
Comprehensive workflow for FPL prediction accuracy tracking and model improvement.

This script automates the entire process of:
1. Logging team suggestions before each gameweek
2. Collecting actual results after gameweeks complete
3. Comparing predictions vs reality
4. Triggering model improvements based on performance
5. Generating accuracy reports

Usage:
python src/accuracy_workflow.py log-suggestion --gameweek 5
python src/accuracy_workflow.py collect-results --gameweek 4
python src/accuracy_workflow.py generate-report --last-n 10
python src/accuracy_workflow.py auto-improve
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from comprehensive_accuracy_tracker import ComprehensiveAccuracyTracker
from ml_pipeline import MLPipeline
from weekly_retrainer import WeeklyMLRetrainer
from fpl_client import FPLClient


def setup_logging():
    """Setup logging for the workflow"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/accuracy_workflow.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def get_current_gameweek() -> int:
    """Get current gameweek from FPL API"""
    try:
        fpl = FPLClient()
        bootstrap = fpl.bootstrap_static()
        events = bootstrap.get("events", [])
        
        for event in events:
            if event.get("is_current", False):
                return event.get("id", 1)
        return 1
    except Exception as e:
        logging.error(f"Failed to get current gameweek: {e}")
        return 1


def log_team_suggestion(gameweek: Optional[int] = None) -> bool:
    """Log team suggestion for the specified gameweek"""
    try:
        if gameweek is None:
            gameweek = get_current_gameweek() + 1  # Next gameweek
        
        logging.info(f"Logging team suggestion for GW{gameweek}")
        
        # Initialize components
        tracker = ComprehensiveAccuracyTracker()
        ml_pipeline = MLPipeline()
        
        # Generate predictions
        logging.info("Generating predictions...")
        predictions_df = ml_pipeline.predict_current()
        if predictions_df.empty:
            logging.error("No predictions available")
            return False
        
        # Get transfer suggestions (simplified)
        transfer_suggestions = []  # Could integrate with transfer API
        
        # Log the suggestion
        suggestion = tracker.log_gameweek_suggestion(
            gameweek, predictions_df, transfer_suggestions
        )
        
        logging.info(f"Team suggestion logged for GW{gameweek}:")
        logging.info(f"  - Formation: {suggestion.formation}")
        logging.info(f"  - Predicted points: {suggestion.predicted_points:.1f}")
        logging.info(f"  - Total cost: £{suggestion.total_cost}m")
        logging.info(f"  - Captain: {next((p['name'] for p in suggestion.starters if p['player_id'] == suggestion.captain_id), 'Unknown')}")
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to log team suggestion: {e}")
        return False


def collect_actual_results(gameweek: Optional[int] = None) -> bool:
    """Collect actual results for a completed gameweek"""
    try:
        if gameweek is None:
            gameweek = get_current_gameweek() - 1  # Previous gameweek
        
        logging.info(f"Collecting actual results for GW{gameweek}")
        
        # Initialize tracker
        tracker = ComprehensiveAccuracyTracker()
        
        # Collect results
        result = tracker.collect_gameweek_results(gameweek)
        if result is None:
            logging.error(f"Could not collect results for GW{gameweek}")
            return False
        
        # Log summary
        prediction_error = abs(result.team_selection.predicted_points - result.actual_team_points)
        accuracy_pct = max(0, 100 - (prediction_error / result.team_selection.predicted_points * 100))
        
        logging.info(f"Results collected for GW{gameweek}:")
        logging.info(f"  - Predicted points: {result.team_selection.predicted_points:.1f}")
        logging.info(f"  - Actual points: {result.actual_team_points}")
        logging.info(f"  - Prediction error: {prediction_error:.1f}")
        logging.info(f"  - Accuracy: {accuracy_pct:.1f}%")
        logging.info(f"  - Captain points: {result.actual_captain_points}")
        logging.info(f"  - Points vs optimal: {result.points_vs_optimal:.1f}")
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to collect results: {e}")
        return False


def generate_accuracy_report(last_n_gameweeks: int = 10) -> bool:
    """Generate and display accuracy report"""
    try:
        logging.info(f"Generating accuracy report for last {last_n_gameweeks} gameweeks")
        
        tracker = ComprehensiveAccuracyTracker()
        report = tracker.get_comprehensive_report(last_n_gameweeks)
        
        if "error" in report:
            logging.error(f"Report generation failed: {report['error']}")
            return False
        
        # Display summary
        summary = report.get("summary", {})
        logging.info("=== ACCURACY REPORT ===")
        logging.info(f"Gameweeks analyzed: {summary.get('gameweeks_analyzed', 0)}")
        logging.info(f"Total predicted points: {summary.get('total_predicted_points', 0):.1f}")
        logging.info(f"Total actual points: {summary.get('total_actual_points', 0):.1f}")
        logging.info(f"Average weekly error: {summary.get('average_weekly_error', 0):.1f}")
        logging.info(f"Prediction accuracy: {summary.get('prediction_accuracy_pct', 0):.1f}%")
        
        # Captain performance
        captain_perf = report.get("captain_performance", {})
        logging.info(f"Captain accuracy correlation: {captain_perf.get('captain_accuracy', 0):.3f}")
        logging.info(f"Average captain error: {captain_perf.get('average_captain_error', 0):.1f}")
        
        # Vs optimal performance
        vs_optimal = report.get("vs_optimal", {})
        logging.info(f"Average points lost vs optimal: {vs_optimal.get('average_points_lost', 0):.1f}")
        logging.info(f"Percentage of optimal achieved: {vs_optimal.get('percentage_of_optimal', 0):.1f}%")
        
        # Gameweek breakdown
        logging.info("\n=== GAMEWEEK BREAKDOWN ===")
        for gw_data in report.get("gameweek_breakdown", []):
            gw = gw_data.get("gameweek", 0)
            pred = gw_data.get("predicted", 0)
            actual = gw_data.get("actual", 0)
            error = gw_data.get("error", 0)
            vs_opt = gw_data.get("vs_optimal", 0)
            
            logging.info(f"GW{gw}: {pred:.1f} pred, {actual:.1f} actual, {error:.1f} error, {vs_opt:.1f} vs optimal")
        
        # Get improvement suggestions
        suggestions = tracker.get_improvement_suggestions()
        logging.info("\n=== IMPROVEMENT SUGGESTIONS ===")
        for i, suggestion in enumerate(suggestions, 1):
            logging.info(f"{i}. {suggestion}")
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to generate report: {e}")
        return False


def auto_improve_models() -> bool:
    """Automatically improve models based on performance"""
    try:
        logging.info("Starting automatic model improvement process")
        
        # Get improvement suggestions
        tracker = ComprehensiveAccuracyTracker()
        suggestions = tracker.get_improvement_suggestions()
        
        # Check if retraining is needed
        needs_retrain = any("retraining" in s.lower() or "accuracy declining" in s.lower() 
                          for s in suggestions)
        
        if needs_retrain:
            logging.info("Performance issues detected - triggering model retraining")
            
            retrainer = WeeklyMLRetrainer()
            current_gw = get_current_gameweek()
            
            result = retrainer.adaptive_retrain(current_gw)
            
            if result.get("status") == "success":
                logging.info(f"Model retraining successful: {result.get('retrain_mode', 'unknown')} mode")
                
                # Log the new model performance
                recent_performance = result.get("recent_performance", [])
                if recent_performance:
                    latest_perf = recent_performance[-1]
                    logging.info(f"Latest performance metrics:")
                    logging.info(f"  - MAE: {latest_perf.get('mae', 0):.2f}")
                    logging.info(f"  - RMSE: {latest_perf.get('rmse', 0):.2f}")
                    logging.info(f"  - Correlation: {latest_perf.get('correlation', 0):.3f}")
                
            else:
                logging.error(f"Model retraining failed: {result.get('error', 'Unknown error')}")
                return False
        else:
            logging.info("Performance is good - no retraining needed")
        
        return True
        
    except Exception as e:
        logging.error(f"Auto-improvement failed: {e}")
        return False


def automated_weekly_workflow(gameweek: Optional[int] = None) -> bool:
    """Run the complete weekly workflow"""
    try:
        current_gw = get_current_gameweek()
        if gameweek is None:
            gameweek = current_gw
        
        logging.info(f"Running automated weekly workflow for GW{gameweek}")
        
        success = True
        
        # Step 1: Log team suggestion for next gameweek
        logging.info("Step 1: Logging team suggestion for next gameweek")
        if not log_team_suggestion(gameweek + 1):
            logging.warning("Team suggestion logging failed")
            success = False
        
        # Step 2: Collect results from previous gameweek (if available)
        if gameweek > 1:
            logging.info("Step 2: Collecting results from previous gameweek")
            if not collect_actual_results(gameweek - 1):
                logging.warning("Results collection failed")
                success = False
        
        # Step 3: Generate accuracy report
        logging.info("Step 3: Generating accuracy report")
        if not generate_accuracy_report(5):
            logging.warning("Report generation failed")
            success = False
        
        # Step 4: Auto-improve models if needed
        logging.info("Step 4: Checking for model improvements")
        if not auto_improve_models():
            logging.warning("Auto-improvement failed")
            success = False
        
        if success:
            logging.info("Automated weekly workflow completed successfully")
        else:
            logging.warning("Automated weekly workflow completed with some errors")
        
        return success
        
    except Exception as e:
        logging.error(f"Automated workflow failed: {e}")
        return False


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="FPL Prediction Accuracy Tracking Workflow"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Log suggestion command
    log_parser = subparsers.add_parser("log-suggestion", help="Log team suggestion")
    log_parser.add_argument("--gameweek", type=int, help="Gameweek to log suggestion for")
    
    # Collect results command
    results_parser = subparsers.add_parser("collect-results", help="Collect actual results")
    results_parser.add_argument("--gameweek", type=int, help="Gameweek to collect results for")
    
    # Generate report command
    report_parser = subparsers.add_parser("generate-report", help="Generate accuracy report")
    report_parser.add_argument("--last-n", type=int, default=10, 
                             help="Number of recent gameweeks to analyze")
    
    # Auto-improve command
    subparsers.add_parser("auto-improve", help="Automatically improve models")
    
    # Automated workflow command
    workflow_parser = subparsers.add_parser("weekly-workflow", help="Run complete weekly workflow")
    workflow_parser.add_argument("--gameweek", type=int, help="Current gameweek")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    success = False
    
    if args.command == "log-suggestion":
        success = log_team_suggestion(args.gameweek)
    elif args.command == "collect-results":
        success = collect_actual_results(args.gameweek)
    elif args.command == "generate-report":
        success = generate_accuracy_report(args.last_n)
    elif args.command == "auto-improve":
        success = auto_improve_models()
    elif args.command == "weekly-workflow":
        success = automated_weekly_workflow(args.gameweek)
    else:
        parser.print_help()
        return
    
    if success:
        logging.info("Command completed successfully")
        sys.exit(0)
    else:
        logging.error("Command failed")
        sys.exit(1)


if __name__ == "__main__":
    main()