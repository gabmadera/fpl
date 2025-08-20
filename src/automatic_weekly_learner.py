from __future__ import annotations

import schedule
import time
import logging
from datetime import datetime
from pathlib import Path

from .comprehensive_accuracy_tracker import ComprehensiveAccuracyTracker
from .weekly_retrainer import WeeklyMLRetrainer
from .fpl_client import FPLClient
from .ml_pipeline import MLPipeline


class AutomaticWeeklyLearner:
    """
    Automatic system that runs weekly to:
    1. Collect gameweek results
    2. Log team suggestions for next week
    3. Optionally retrain models based on performance
    4. Update predictions with latest data
    """
    
    def __init__(self, auto_retrain_mode: str = "smart"):
        """
        auto_retrain_mode options:
        - "always": Retrain every week regardless of performance
        - "smart": Only retrain when performance degrades (default)
        - "never": Never automatically retrain (manual only)
        """
        self.tracker = ComprehensiveAccuracyTracker()
        self.retrainer = WeeklyMLRetrainer()
        self.fpl = FPLClient()
        self.ml_pipeline = MLPipeline()
        self.auto_retrain_mode = auto_retrain_mode
        self.logger = logging.getLogger(__name__)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/weekly_learner.log'),
                logging.StreamHandler()
            ]
        )
    
    def run_weekly_learning_cycle(self) -> dict:
        """Complete weekly learning cycle"""
        try:
            current_gw = self._get_current_gameweek()
            self.logger.info(f"Starting weekly learning cycle for GW{current_gw}")
            
            results = {
                "gameweek": current_gw,
                "timestamp": datetime.now().isoformat(),
                "steps_completed": [],
                "errors": []
            }
            
            # Step 1: Collect previous gameweek results (if available)
            if current_gw > 1:
                try:
                    self.logger.info(f"Collecting results for GW{current_gw - 1}")
                    result = self.tracker.collect_gameweek_results(current_gw - 1)
                    if result:
                        results["steps_completed"].append("collected_results")
                        results["actual_points"] = result.actual_team_points
                        results["prediction_accuracy"] = result.prediction_accuracy
                    else:
                        results["errors"].append("Failed to collect previous gameweek results")
                except Exception as e:
                    self.logger.error(f"Failed to collect results: {e}")
                    results["errors"].append(f"Results collection error: {str(e)}")
            
            # Step 2: Update model based on mode
            try:
                if self.auto_retrain_mode == "always":
                    self.logger.info("Performing weekly model retraining (always mode)")
                    retrain_result = self.retrainer.adaptive_retrain(current_gw)
                    results["steps_completed"].append("model_retrained")
                    results["retrain_result"] = retrain_result
                    
                elif self.auto_retrain_mode == "smart":
                    self.logger.info("Evaluating if model retraining is needed (smart mode)")
                    retrain_result = self.retrainer.adaptive_retrain(current_gw)
                    results["steps_completed"].append("smart_retrain_check")
                    results["retrain_result"] = retrain_result
                    
                else:  # "never" mode
                    self.logger.info("Skipping automatic retraining (never mode)")
                    results["steps_completed"].append("retrain_skipped")
                    
            except Exception as e:
                self.logger.error(f"Model update failed: {e}")
                results["errors"].append(f"Model update error: {str(e)}")
            
            # Step 3: Generate fresh predictions
            try:
                self.logger.info("Generating fresh predictions")
                predictions = self.ml_pipeline.predict_current()
                if not predictions.empty:
                    results["steps_completed"].append("predictions_generated")
                    results["prediction_count"] = len(predictions)
                    
                    # Save predictions
                    output_file = Path("data/processed/predictions_current.csv")
                    predictions.to_csv(output_file, index=False)
                    
            except Exception as e:
                self.logger.error(f"Prediction generation failed: {e}")
                results["errors"].append(f"Prediction error: {str(e)}")
            
            # Step 4: Log team suggestion for next gameweek
            try:
                self.logger.info(f"Logging team suggestion for GW{current_gw + 1}")
                suggestion = self.tracker.log_gameweek_suggestion(
                    current_gw + 1, predictions, []
                )
                results["steps_completed"].append("suggestion_logged")
                results["suggested_points"] = suggestion.predicted_points
                results["suggested_formation"] = suggestion.formation
                
            except Exception as e:
                self.logger.error(f"Team suggestion logging failed: {e}")
                results["errors"].append(f"Suggestion logging error: {str(e)}")
            
            # Step 5: Clear prediction cache to force fresh data
            try:
                from .cache_manager import CacheManager
                cache_manager = CacheManager()
                cache_manager.clear_cache("predictions")
                results["steps_completed"].append("cache_cleared")
            except Exception as e:
                self.logger.warning(f"Cache clearing failed: {e}")
            
            success = len(results["errors"]) == 0
            results["status"] = "success" if success else "partial_success"
            
            self.logger.info(f"Weekly learning cycle completed: {results['status']}")
            return results
            
        except Exception as e:
            self.logger.error(f"Weekly learning cycle failed: {e}")
            return {
                "status": "failed", 
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_current_gameweek(self) -> int:
        """Get current gameweek from FPL API"""
        try:
            bootstrap = self.fpl.bootstrap_static()
            events = bootstrap.get("events", [])
            for event in events:
                if event.get("is_current", False):
                    return event.get("id", 1)
            return 1
        except Exception:
            return 1
    
    def start_scheduler(self, run_time: str = "09:00"):
        """
        Start automatic weekly scheduler
        
        Args:
            run_time: Time to run each day (e.g., "09:00")
        """
        self.logger.info(f"Starting weekly learning scheduler - runs daily at {run_time}")
        
        # Schedule to run every day (will check if it's appropriate to run)
        schedule.every().day.at(run_time).do(self._scheduled_run)
        
        print(f"🤖 Automatic Weekly Learner started!")
        print(f"⏰ Will run daily at {run_time}")
        print(f"🔄 Mode: {self.auto_retrain_mode}")
        print(f"📝 Logs: logs/weekly_learner.log")
        print(f"🛑 Press Ctrl+C to stop")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            self.logger.info("Weekly learner scheduler stopped by user")
            print("\n👋 Weekly learner stopped")
    
    def _scheduled_run(self):
        """Run the weekly learning cycle if appropriate"""
        try:
            # Check if we should run today
            current_gw = self._get_current_gameweek()
            
            # Check if we already processed this gameweek today
            log_file = Path(f"logs/weekly_run_gw{current_gw}.log")
            if log_file.exists():
                # Check if file was created today
                file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                if file_time.date() == datetime.now().date():
                    self.logger.info(f"Already processed GW{current_gw} today, skipping")
                    return
            
            self.logger.info(f"Running scheduled weekly learning for GW{current_gw}")
            result = self.run_weekly_learning_cycle()
            
            # Log that we ran today
            with open(log_file, 'w') as f:
                f.write(f"Ran on {datetime.now().isoformat()}\n")
                f.write(f"Result: {result['status']}\n")
                f.write(f"Steps: {result.get('steps_completed', [])}\n")
            
            self.logger.info(f"Scheduled run completed: {result['status']}")
            
        except Exception as e:
            self.logger.error(f"Scheduled run failed: {e}")


def main():
    """CLI interface for automatic weekly learner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="FPL Automatic Weekly Learner")
    parser.add_argument(
        "--mode", 
        choices=["always", "smart", "never"],
        default="smart",
        help="Retraining mode: always, smart, or never"
    )
    parser.add_argument(
        "--time",
        default="09:00", 
        help="Time to run daily (e.g., 09:00)"
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Run once immediately instead of scheduling"
    )
    
    args = parser.parse_args()
    
    learner = AutomaticWeeklyLearner(auto_retrain_mode=args.mode)
    
    if args.run_once:
        print("🚀 Running weekly learning cycle once...")
        result = learner.run_weekly_learning_cycle()
        print(f"✅ Result: {result['status']}")
        print(f"📊 Steps completed: {result.get('steps_completed', [])}")
        if result.get('errors'):
            print(f"⚠️  Errors: {result['errors']}")
    else:
        learner.start_scheduler(args.time)


if __name__ == "__main__":
    main()