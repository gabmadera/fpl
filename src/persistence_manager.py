from __future__ import annotations

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging


class PersistenceManager:
    """Comprehensive data persistence and memory management for FPL AI"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Create necessary directories
        self.data_dirs = {
            'gameweek_results': Path("data/gameweek_results"),
            'team_selections': Path("data/team_selections"),
            'model_performance': Path("data/model_performance"),
            'user_sessions': Path("data/user_sessions"),
            'system_state': Path("data/system_state")
        }
        
        for dir_path in self.data_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Load system state on initialization
        self.system_state = self._load_system_state()
    
    def save_gameweek_results(self, gameweek: int, results_data: Dict[str, Any]) -> bool:
        """Save gameweek results for future reference"""
        try:
            results_file = self.data_dirs['gameweek_results'] / f"gw{gameweek}_results.json"
            
            # Add metadata
            results_data['gameweek'] = gameweek
            results_data['timestamp'] = datetime.now().isoformat()
            results_data['season'] = self.get_current_season()
            
            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            
            self.logger.info(f"Saved GW{gameweek} results to {results_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save GW{gameweek} results: {e}")
            return False
    
    def load_gameweek_results(self, gameweek: int) -> Optional[Dict[str, Any]]:
        """Load results for a specific gameweek"""
        try:
            results_file = self.data_dirs['gameweek_results'] / f"gw{gameweek}_results.json"
            
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                return results
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to load GW{gameweek} results: {e}")
            return None
    
    def save_team_selection(self, gameweek: int, team_data: Dict[str, Any], 
                          selection_type: str = "optimal") -> bool:
        """Save team selection for a gameweek"""
        try:
            team_file = self.data_dirs['team_selections'] / f"gw{gameweek}_{selection_type}_team.json"
            
            # Add metadata
            team_data['gameweek'] = gameweek
            team_data['selection_type'] = selection_type
            team_data['timestamp'] = datetime.now().isoformat()
            team_data['season'] = self.get_current_season()
            
            with open(team_file, 'w') as f:
                json.dump(team_data, f, indent=2, default=str)
            
            self.logger.info(f"Saved {selection_type} team for GW{gameweek}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save team selection for GW{gameweek}: {e}")
            return False
    
    def load_team_selection(self, gameweek: int, selection_type: str = "optimal") -> Optional[Dict[str, Any]]:
        """Load team selection for a gameweek"""
        try:
            team_file = self.data_dirs['team_selections'] / f"gw{gameweek}_{selection_type}_team.json"
            
            if team_file.exists():
                with open(team_file, 'r') as f:
                    team_data = json.load(f)
                return team_data
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to load team selection for GW{gameweek}: {e}")
            return None
    
    def save_model_performance(self, gameweek: int, performance_data: Dict[str, Any]) -> bool:
        """Save model performance metrics"""
        try:
            perf_file = self.data_dirs['model_performance'] / f"gw{gameweek}_performance.json"
            
            # Add metadata
            performance_data['gameweek'] = gameweek
            performance_data['timestamp'] = datetime.now().isoformat()
            performance_data['season'] = self.get_current_season()
            
            with open(perf_file, 'w') as f:
                json.dump(performance_data, f, indent=2, default=str)
            
            self.logger.info(f"Saved model performance for GW{gameweek}")
            
            # Update cumulative performance tracking
            self._update_cumulative_performance(performance_data)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save model performance for GW{gameweek}: {e}")
            return False
    
    def get_performance_history(self, last_n_weeks: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get historical model performance"""
        try:
            performance_files = list(self.data_dirs['model_performance'].glob("gw*_performance.json"))
            performance_files.sort(key=lambda x: int(x.stem.split('_')[0][2:]))  # Sort by gameweek
            
            if last_n_weeks:
                performance_files = performance_files[-last_n_weeks:]
            
            history = []
            for file in performance_files:
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                    history.append(data)
                except Exception:
                    continue
            
            return history
            
        except Exception as e:
            self.logger.error(f"Failed to get performance history: {e}")
            return []
    
    def save_system_state(self, state_data: Dict[str, Any]) -> bool:
        """Save current system state"""
        try:
            state_file = self.data_dirs['system_state'] / "current_state.json"
            
            # Merge with existing state
            current_state = self.system_state.copy()
            current_state.update(state_data)
            current_state['last_updated'] = datetime.now().isoformat()
            
            with open(state_file, 'w') as f:
                json.dump(current_state, f, indent=2, default=str)
            
            # Update in-memory state
            self.system_state = current_state
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save system state: {e}")
            return False
    
    def _load_system_state(self) -> Dict[str, Any]:
        """Load system state on initialization"""
        try:
            state_file = self.data_dirs['system_state'] / "current_state.json"
            
            if state_file.exists():
                with open(state_file, 'r') as f:
                    state = json.load(f)
                self.logger.info("Loaded existing system state")
                return state
            else:
                # Initialize default state
                default_state = {
                    'current_gameweek': 1,
                    'season': self.get_current_season(),
                    'total_predictions_made': 0,
                    'models_trained': 0,
                    'last_model_training': None,
                    'predictions_accuracy_rolling': [],
                    'best_performing_model': 'ensemble',
                    'initialized_at': datetime.now().isoformat()
                }
                
                self.save_system_state(default_state)
                return default_state
                
        except Exception as e:
            self.logger.error(f"Failed to load system state: {e}")
            return {}
    
    def get_current_season(self) -> str:
        """Determine current FPL season"""
        now = datetime.now()
        if now.month >= 8:  # Season starts in August
            return f"{now.year}/{now.year + 1}"
        else:
            return f"{now.year - 1}/{now.year}"
    
    def is_data_consistent(self) -> Dict[str, Any]:
        """Check data consistency across restarts"""
        try:
            current_season = self.get_current_season()
            system_season = self.system_state.get('season', current_season)
            
            consistency_report = {
                'season_match': current_season == system_season,
                'current_season': current_season,
                'system_season': system_season,
                'gameweek_data_available': len(list(self.data_dirs['gameweek_results'].glob("*.json"))),
                'team_selections_saved': len(list(self.data_dirs['team_selections'].glob("*.json"))),
                'model_performance_records': len(list(self.data_dirs['model_performance'].glob("*.json"))),
                'last_activity': self.system_state.get('last_updated'),
                'total_predictions_made': self.system_state.get('total_predictions_made', 0)
            }
            
            return consistency_report
            
        except Exception as e:
            self.logger.error(f"Consistency check failed: {e}")
            return {"error": str(e)}
    
    def cleanup_old_data(self, days_to_keep: int = 365):
        """Clean up old data files"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            cleaned_files = 0
            
            for dir_name, dir_path in self.data_dirs.items():
                for file in dir_path.glob("*.json"):
                    if file.stat().st_mtime < cutoff_date.timestamp():
                        file.unlink()
                        cleaned_files += 1
            
            self.logger.info(f"Cleaned up {cleaned_files} old files")
            return cleaned_files
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
            return 0
    
    def _update_cumulative_performance(self, performance_data: Dict[str, Any]):
        """Update cumulative performance metrics"""
        try:
            # Track rolling accuracy
            if 'accuracy_metrics' in performance_data:
                current_rolling = self.system_state.get('predictions_accuracy_rolling', [])
                current_rolling.append(performance_data['accuracy_metrics'])
                
                # Keep last 10 gameweeks
                if len(current_rolling) > 10:
                    current_rolling = current_rolling[-10:]
                
                self.system_state['predictions_accuracy_rolling'] = current_rolling
            
            # Update totals
            self.system_state['total_predictions_made'] = self.system_state.get('total_predictions_made', 0) + 1
            
        except Exception as e:
            self.logger.error(f"Failed to update cumulative performance: {e}")
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get a summary of what the system remembers"""
        try:
            consistency = self.is_data_consistent()
            
            # Get recent gameweeks with data
            gameweek_files = list(self.data_dirs['gameweek_results'].glob("gw*.json"))
            gameweeks_with_data = [int(f.stem.split('_')[0][2:]) for f in gameweek_files]
            gameweeks_with_data.sort()
            
            # Get saved team selections
            team_files = list(self.data_dirs['team_selections'].glob("gw*.json"))
            gameweeks_with_teams = [int(f.stem.split('_')[0][2:]) for f in team_files]
            gameweeks_with_teams.sort()
            
            summary = {
                "system_status": "persistent" if consistency['season_match'] else "season_change_detected",
                "current_season": consistency['current_season'],
                "gameweeks_with_results": gameweeks_with_data,
                "gameweeks_with_team_selections": gameweeks_with_teams,
                "total_predictions_made": consistency['total_predictions_made'],
                "model_performance_records": consistency['model_performance_records'],
                "last_activity": consistency['last_activity'],
                "data_directories": {name: str(path) for name, path in self.data_dirs.items()},
                "memory_features": [
                    "Gameweek results and predictions",
                    "Team selections with reasoning",
                    "Model performance tracking", 
                    "Prediction accuracy history",
                    "System state and configuration"
                ]
            }
            
            return summary
            
        except Exception as e:
            return {"error": f"Memory summary failed: {e}"}