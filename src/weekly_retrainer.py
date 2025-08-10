from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import joblib
import logging

from .ml_pipeline import MLPipeline
from .fpl_client import FPLClient
from .data_prep import DataPrep


class WeeklyMLRetrainer:
    """System for continuous ML model improvement throughout the FPL season"""
    
    def __init__(self):
        self.ml_pipeline = MLPipeline()
        self.fpl = FPLClient()
        self.data_prep = DataPrep()
        self.logger = logging.getLogger(__name__)
        
        # Create directories
        Path("models/weekly").mkdir(parents=True, exist_ok=True)
        Path("data/weekly_performance").mkdir(parents=True, exist_ok=True)
    
    def should_retrain(self, current_gameweek: int) -> bool:
        """Determine if model should be retrained based on conditions"""
        
        # Retrain after every gameweek once season starts
        if current_gameweek < 2:
            return False  # Need at least one completed gameweek
        
        # Check if we already retrained this week
        model_path = f"models/weekly/model_gw{current_gameweek}.pkl"
        if Path(model_path).exists():
            return False
        
        # Check if new data is available
        try:
            latest_data = self.fpl.bootstrap_static()
            if not latest_data or not latest_data.get("elements"):
                return False
        except Exception:
            return False
        
        return True
    
    def collect_gameweek_results(self, gameweek: int) -> Optional[pd.DataFrame]:
        """Collect actual player performance data for completed gameweek"""
        
        try:
            # Get player data for the gameweek
            players_data = []
            
            # Fetch from FPL API - element-summary for each player
            bootstrap = self.fpl.bootstrap_static()
            elements = bootstrap.get("elements", [])
            
            for element in elements:
                player_id = element["id"]
                
                try:
                    # Get detailed player data including gameweek history
                    player_detail = self.fpl.element_summary(player_id)
                    
                    if "history" in player_detail:
                        # Find the specific gameweek data
                        for gw_data in player_detail["history"]:
                            if gw_data.get("round") == gameweek:
                                players_data.append({
                                    "player_id": player_id,
                                    "gameweek": gameweek,
                                    "actual_points": gw_data.get("total_points", 0),
                                    "minutes": gw_data.get("minutes", 0),
                                    "goals_scored": gw_data.get("goals_scored", 0),
                                    "assists": gw_data.get("assists", 0),
                                    "clean_sheets": gw_data.get("clean_sheets", 0),
                                    "goals_conceded": gw_data.get("goals_conceded", 0),
                                    "bonus": gw_data.get("bonus", 0),
                                    "bps": gw_data.get("bps", 0),
                                    "influence": gw_data.get("influence", 0),
                                    "creativity": gw_data.get("creativity", 0),
                                    "threat": gw_data.get("threat", 0),
                                    "selected": gw_data.get("selected", 0),
                                    "transfers_in": gw_data.get("transfers_in", 0),
                                    "transfers_out": gw_data.get("transfers_out", 0),
                                    "value": gw_data.get("value", 50)
                                })
                                break
                                
                except Exception as e:
                    # Skip individual player errors
                    continue
            
            if not players_data:
                return None
                
            df = pd.DataFrame(players_data)
            
            # Save for analysis
            df.to_csv(f"data/weekly_performance/gw{gameweek}_results.csv", index=False)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to collect GW{gameweek} results: {e}")
            return None
    
    def evaluate_previous_predictions(self, gameweek: int) -> Dict[str, float]:
        """Evaluate how well previous predictions performed"""
        
        try:
            # Load previous predictions
            pred_file = f"data/weekly_performance/gw{gameweek}_predictions.csv"
            if not Path(pred_file).exists():
                return {"error": "No previous predictions found"}
            
            predictions = pd.read_csv(pred_file)
            
            # Load actual results
            results_file = f"data/weekly_performance/gw{gameweek}_results.csv"
            if not Path(results_file).exists():
                return {"error": "No actual results found"}
            
            actuals = pd.read_csv(results_file)
            
            # Merge predictions and actuals
            comparison = predictions.merge(actuals, on="player_id", how="inner")
            
            if comparison.empty:
                return {"error": "No matching data found"}
            
            # Calculate accuracy metrics
            mae = np.mean(np.abs(comparison["predicted_points"] - comparison["actual_points"]))
            rmse = np.sqrt(np.mean((comparison["predicted_points"] - comparison["actual_points"]) ** 2))
            
            # Calculate correlation
            correlation = comparison["predicted_points"].corr(comparison["actual_points"])
            
            # Top/bottom performer accuracy
            pred_top_10 = comparison.nlargest(10, "predicted_points")["player_id"].tolist()
            actual_top_10 = comparison.nlargest(10, "actual_points")["player_id"].tolist()
            top_overlap = len(set(pred_top_10) & set(actual_top_10)) / 10.0
            
            metrics = {
                "mae": float(mae),
                "rmse": float(rmse),
                "correlation": float(correlation) if not pd.isna(correlation) else 0.0,
                "top_10_overlap": float(top_overlap),
                "total_players": len(comparison),
                "gameweek": gameweek
            }
            
            # Save evaluation
            eval_df = pd.DataFrame([metrics])
            eval_df.to_csv(f"data/weekly_performance/gw{gameweek}_evaluation.csv", index=False)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate GW{gameweek} predictions: {e}")
            return {"error": str(e)}
    
    def adaptive_retrain(self, current_gameweek: int) -> Dict[str, any]:
        """Perform adaptive retraining based on recent performance"""
        
        try:
            # Collect latest results
            results_df = self.collect_gameweek_results(current_gameweek - 1)
            if results_df is None:
                return {"status": "failed", "reason": "No recent results available"}
            
            # Evaluate recent performance
            recent_performance = []
            for gw in range(max(1, current_gameweek - 4), current_gameweek):
                metrics = self.evaluate_previous_predictions(gw)
                if "error" not in metrics:
                    recent_performance.append(metrics)
            
            # Determine if retraining is beneficial
            if len(recent_performance) >= 2:
                recent_mae = np.mean([p["mae"] for p in recent_performance[-2:]])
                baseline_mae = 2.0  # Baseline expectation
                
                if recent_mae > baseline_mae * 1.2:
                    # Performance is degrading, do full retrain
                    retrain_mode = "full"
                elif recent_mae < baseline_mae * 0.9:
                    # Performance is good, do incremental update
                    retrain_mode = "incremental"
                else:
                    # Performance is stable, light refresh
                    retrain_mode = "refresh"
            else:
                retrain_mode = "full"
            
            # Execute retraining
            if retrain_mode == "full":
                result = self._full_retrain(current_gameweek)
            elif retrain_mode == "incremental":
                result = self._incremental_update(current_gameweek, results_df)
            else:
                result = self._refresh_features(current_gameweek)
            
            result["retrain_mode"] = retrain_mode
            result["recent_performance"] = recent_performance
            
            # Save predictions for next evaluation
            predictions = self.ml_pipeline.predict_current()
            predictions[["player_id", "predicted_points"]].to_csv(
                f"data/weekly_performance/gw{current_gameweek}_predictions.csv", index=False
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Adaptive retrain failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def _full_retrain(self, gameweek: int) -> Dict[str, any]:
        """Full model retraining with all available data"""
        
        # Build comprehensive training dataset
        X, y = self.data_prep.build_training_table()
        
        if X.empty or y.empty:
            return {"status": "failed", "reason": "No training data available"}
        
        # Train new model
        result = self.ml_pipeline.xgb.train(X, y)
        
        # Save gameweek-specific model
        model_path = f"models/weekly/model_gw{gameweek}.pkl"
        joblib.dump({
            "model": self.ml_pipeline.xgb.model,
            "feature_names": self.ml_pipeline.xgb.feature_names,
            "gameweek": gameweek,
            "training_metrics": result,
            "timestamp": datetime.now().isoformat()
        }, model_path)
        
        result["status"] = "success"
        result["type"] = "full_retrain"
        return result
    
    def _incremental_update(self, gameweek: int, new_data: pd.DataFrame) -> Dict[str, any]:
        """Incremental model update with new gameweek data"""
        
        try:
            # Load existing model
            latest_model_path = self._get_latest_model_path()
            if latest_model_path:
                model_data = joblib.load(latest_model_path)
                self.ml_pipeline.xgb.model = model_data["model"]
                self.ml_pipeline.xgb.feature_names = model_data.get("feature_names", [])
            
            # Prepare new training examples from recent gameweek
            # This would require converting gameweek results to feature format
            # For now, fall back to refresh
            return self._refresh_features(gameweek)
            
        except Exception as e:
            # Fall back to full retrain if incremental fails
            return self._full_retrain(gameweek)
    
    def _refresh_features(self, gameweek: int) -> Dict[str, any]:
        """Light refresh of features without full retraining"""
        
        # Update feature engineering with latest data
        predictions = self.ml_pipeline.predict_current()
        
        return {
            "status": "success",
            "type": "feature_refresh", 
            "predictions_count": len(predictions),
            "gameweek": gameweek
        }
    
    def _get_latest_model_path(self) -> Optional[str]:
        """Get path to most recent model"""
        
        model_dir = Path("models/weekly")
        if not model_dir.exists():
            return None
        
        model_files = list(model_dir.glob("model_gw*.pkl"))
        if not model_files:
            return None
        
        # Sort by gameweek number
        model_files.sort(key=lambda x: int(x.stem.split("gw")[1]))
        return str(model_files[-1])
    
    def get_performance_history(self) -> List[Dict]:
        """Get historical model performance metrics"""
        
        performance_dir = Path("data/weekly_performance")
        if not performance_dir.exists():
            return []
        
        eval_files = list(performance_dir.glob("gw*_evaluation.csv"))
        
        history = []
        for eval_file in sorted(eval_files):
            try:
                df = pd.read_csv(eval_file)
                if not df.empty:
                    history.append(df.iloc[0].to_dict())
            except Exception:
                continue
        
        return history
    
    def schedule_weekly_retrain(self) -> Dict[str, any]:
        """Main entry point for scheduled weekly retraining"""
        
        try:
            # Get current gameweek
            bootstrap = self.fpl.bootstrap_static()
            events = bootstrap.get("events", [])
            
            current_gw = 1
            for event in events:
                if event.get("is_current", False):
                    current_gw = event.get("id", 1)
                    break
            
            # Check if retraining is needed
            if not self.should_retrain(current_gw):
                return {
                    "status": "skipped", 
                    "reason": f"No retrain needed for GW{current_gw}",
                    "gameweek": current_gw
                }
            
            # Perform adaptive retraining
            result = self.adaptive_retrain(current_gw)
            result["gameweek"] = current_gw
            
            self.logger.info(f"Weekly retrain completed for GW{current_gw}: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Weekly retrain failed: {e}")
            return {"status": "failed", "error": str(e)}