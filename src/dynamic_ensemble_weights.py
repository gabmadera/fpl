from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
from pathlib import Path
import joblib


class DynamicEnsembleWeights:
    """Adaptive ensemble weighting based on recent performance and context"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Default weights (fallback)
        self.default_weights = {
            'xgb': 0.5,
            'rf': 0.3, 
            'nn': 0.2
        }
        
        # Position-specific default weights
        self.position_default_weights = {
            'GKP': {'xgb': 0.6, 'rf': 0.25, 'nn': 0.15},  # XGB good for GKP patterns
            'DEF': {'xgb': 0.5, 'rf': 0.35, 'nn': 0.15},  # RF good for defensive patterns
            'MID': {'xgb': 0.45, 'rf': 0.3, 'nn': 0.25},  # NN good for complex mid patterns
            'FWD': {'xgb': 0.5, 'rf': 0.25, 'nn': 0.25}   # Balanced for forwards
        }
        
        # Fixture context weights
        self.fixture_context_weights = {
            'easy_fixtures': {'xgb': 0.45, 'rf': 0.3, 'nn': 0.25},    # NN good for high-scoring games
            'hard_fixtures': {'xgb': 0.55, 'rf': 0.35, 'nn': 0.10},   # XGB/RF better for low-scoring
            'home_games': {'xgb': 0.48, 'rf': 0.32, 'nn': 0.20},
            'away_games': {'xgb': 0.52, 'rf': 0.28, 'nn': 0.20}
        }
        
        # Performance tracking
        self.model_performance_history = {}
        self.current_weights = self.default_weights.copy()
        self.weights_file = Path("data/model_weights/dynamic_weights.joblib")
        
        # Ensure directory exists
        self.weights_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing weights if available
        self._load_weights()
    
    def calculate_adaptive_weights(self, 
                                 recent_performance: Dict[str, List[float]], 
                                 position: str = None,
                                 fixture_context: Dict = None,
                                 gameweek: int = None) -> Dict[str, float]:
        """Calculate adaptive weights based on recent performance and context"""
        try:
            # Start with position-specific base weights
            if position and position in self.position_default_weights:
                base_weights = self.position_default_weights[position].copy()
            else:
                base_weights = self.default_weights.copy()
            
            # Adjust based on recent performance
            if recent_performance:
                performance_weights = self._calculate_performance_weights(recent_performance)
                base_weights = self._blend_weights(base_weights, performance_weights, blend_ratio=0.6)
            
            # Adjust based on fixture context
            if fixture_context:
                context_weights = self._calculate_context_weights(fixture_context)
                base_weights = self._blend_weights(base_weights, context_weights, blend_ratio=0.3)
            
            # Adjust based on gameweek (season context)
            if gameweek:
                season_weights = self._calculate_season_weights(gameweek)
                base_weights = self._blend_weights(base_weights, season_weights, blend_ratio=0.2)
            
            # Normalize weights to sum to 1
            total_weight = sum(base_weights.values())
            normalized_weights = {model: weight/total_weight for model, weight in base_weights.items()}
            
            # Store current weights
            self.current_weights = normalized_weights
            self._save_weights()
            
            self.logger.info(f"Adaptive weights calculated: {normalized_weights}")
            return normalized_weights
            
        except Exception as e:
            self.logger.error(f"Adaptive weight calculation failed: {e}")
            return self.default_weights
    
    def _calculate_performance_weights(self, recent_performance: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate weights based on recent model performance"""
        try:
            model_scores = {}
            
            for model_name, scores in recent_performance.items():
                if not scores:
                    model_scores[model_name] = 0.5  # Neutral score
                    continue
                
                # Calculate recent performance with exponential weighting
                weights = np.exp(np.linspace(-1, 0, len(scores)))  # More recent = higher weight
                weights = weights / weights.sum()
                
                # Weighted average of recent scores (assume scores are accuracy metrics)
                weighted_score = np.average(scores, weights=weights)
                model_scores[model_name] = weighted_score
            
            # Convert scores to weights (better performance = higher weight)
            total_score = sum(model_scores.values())
            if total_score == 0:
                return self.default_weights
            
            performance_weights = {
                model: score / total_score for model, score in model_scores.items()
            }
            
            return performance_weights
            
        except Exception as e:
            self.logger.error(f"Performance weight calculation failed: {e}")
            return self.default_weights
    
    def _calculate_context_weights(self, fixture_context: Dict) -> Dict[str, float]:
        """Calculate weights based on fixture difficulty and context"""
        try:
            avg_difficulty = fixture_context.get('average_difficulty', 3.0)
            is_home_heavy = fixture_context.get('home_games_ratio', 0.5) > 0.6
            
            # Easy fixtures (difficulty < 2.5) - NN performs better in high-scoring games
            if avg_difficulty < 2.5:
                return self.fixture_context_weights['easy_fixtures']
            
            # Hard fixtures (difficulty > 3.5) - XGB/RF better for low-scoring, defensive games
            elif avg_difficulty > 3.5:
                return self.fixture_context_weights['hard_fixtures']
            
            # Home vs away bias
            elif is_home_heavy:
                return self.fixture_context_weights['home_games']
            else:
                return self.fixture_context_weights['away_games']
            
        except Exception as e:
            self.logger.error(f"Context weight calculation failed: {e}")
            return self.default_weights
    
    def _calculate_season_weights(self, gameweek: int) -> Dict[str, float]:
        """Calculate weights based on season phase"""
        try:
            # Early season (GW 1-10) - More uncertainty, favor robust models
            if gameweek <= 10:
                return {'xgb': 0.55, 'rf': 0.35, 'nn': 0.10}
            
            # Mid season (GW 11-28) - Normal distribution
            elif gameweek <= 28:
                return {'xgb': 0.50, 'rf': 0.30, 'nn': 0.20}
            
            # Late season (GW 29+) - More data, NN can perform better
            else:
                return {'xgb': 0.45, 'rf': 0.25, 'nn': 0.30}
                
        except Exception as e:
            self.logger.error(f"Season weight calculation failed: {e}")
            return self.default_weights
    
    def _blend_weights(self, base_weights: Dict[str, float], 
                      adjustment_weights: Dict[str, float], 
                      blend_ratio: float = 0.5) -> Dict[str, float]:
        """Blend two sets of weights with specified ratio"""
        try:
            blended_weights = {}
            
            for model in base_weights.keys():
                base_weight = base_weights[model]
                adj_weight = adjustment_weights.get(model, base_weight)
                
                # Blend: base_ratio * base + (1 - base_ratio) * adjustment
                base_ratio = 1 - blend_ratio
                blended_weights[model] = base_ratio * base_weight + blend_ratio * adj_weight
            
            return blended_weights
            
        except Exception as e:
            self.logger.error(f"Weight blending failed: {e}")
            return base_weights
    
    def update_performance_history(self, model_performance: Dict[str, float], gameweek: int):
        """Update model performance history for adaptive learning"""
        try:
            if gameweek not in self.model_performance_history:
                self.model_performance_history[gameweek] = {}
            
            self.model_performance_history[gameweek] = model_performance
            
            # Keep only last 10 gameweeks for efficiency
            if len(self.model_performance_history) > 10:
                oldest_gw = min(self.model_performance_history.keys())
                del self.model_performance_history[oldest_gw]
            
            self.logger.info(f"Updated performance history for GW{gameweek}: {model_performance}")
            
        except Exception as e:
            self.logger.error(f"Performance history update failed: {e}")
    
    def get_recent_performance(self, last_n_weeks: int = 5) -> Dict[str, List[float]]:
        """Get recent performance for all models"""
        try:
            recent_performance = {'xgb': [], 'rf': [], 'nn': []}
            
            # Get last N gameweeks
            recent_gameweeks = sorted(self.model_performance_history.keys())[-last_n_weeks:]
            
            for gw in recent_gameweeks:
                gw_performance = self.model_performance_history[gw]
                
                for model in recent_performance.keys():
                    if model in gw_performance:
                        recent_performance[model].append(gw_performance[model])
            
            return recent_performance
            
        except Exception as e:
            self.logger.error(f"Recent performance retrieval failed: {e}")
            return {'xgb': [], 'rf': [], 'nn': []}
    
    def get_position_optimized_weights(self, position: str, 
                                     recent_performance: Dict[str, List[float]] = None) -> Dict[str, float]:
        """Get optimized weights for specific position"""
        try:
            # Start with position-specific weights
            if position in self.position_default_weights:
                weights = self.position_default_weights[position].copy()
            else:
                weights = self.default_weights.copy()
            
            # Adjust based on recent performance if available
            if recent_performance:
                performance_weights = self._calculate_performance_weights(recent_performance)
                weights = self._blend_weights(weights, performance_weights, blend_ratio=0.4)
            
            # Normalize
            total = sum(weights.values())
            weights = {model: weight/total for model, weight in weights.items()}
            
            return weights
            
        except Exception as e:
            self.logger.error(f"Position-optimized weight calculation failed: {e}")
            return self.default_weights
    
    def _save_weights(self):
        """Save current weights to disk"""
        try:
            weights_data = {
                'current_weights': self.current_weights,
                'performance_history': self.model_performance_history,
                'last_updated': datetime.now().isoformat()
            }
            
            joblib.dump(weights_data, self.weights_file)
            
        except Exception as e:
            self.logger.error(f"Weight saving failed: {e}")
    
    def _load_weights(self):
        """Load weights from disk if available"""
        try:
            if self.weights_file.exists():
                weights_data = joblib.load(self.weights_file)
                
                self.current_weights = weights_data.get('current_weights', self.default_weights)
                self.model_performance_history = weights_data.get('performance_history', {})
                
                self.logger.info(f"Loaded dynamic weights: {self.current_weights}")
            else:
                self.logger.info("No saved weights found, using defaults")
                
        except Exception as e:
            self.logger.error(f"Weight loading failed: {e}")
            self.current_weights = self.default_weights
    
    def get_model_confidence_scores(self) -> Dict[str, float]:
        """Get confidence scores for each model based on recent performance"""
        try:
            recent_perf = self.get_recent_performance()
            confidence_scores = {}
            
            for model, scores in recent_perf.items():
                if not scores:
                    confidence_scores[model] = 0.5  # Neutral confidence
                else:
                    # Calculate confidence as inverse of score variance (more consistent = higher confidence)
                    if len(scores) == 1:
                        confidence_scores[model] = 0.7  # Single score = moderate confidence
                    else:
                        score_variance = np.var(scores)
                        # Convert variance to confidence (0-1 scale)
                        confidence = 1 / (1 + score_variance * 5)  # Scale factor of 5
                        confidence_scores[model] = min(0.9, max(0.1, confidence))
            
            return confidence_scores
            
        except Exception as e:
            self.logger.error(f"Confidence score calculation failed: {e}")
            return {'xgb': 0.5, 'rf': 0.5, 'nn': 0.5}
    
    def get_weighted_prediction(self, model_predictions: Dict[str, float], 
                              position: str = None,
                              fixture_context: Dict = None) -> float:
        """Get weighted prediction using dynamic weights"""
        try:
            # Get appropriate weights
            if position or fixture_context:
                recent_perf = self.get_recent_performance()
                weights = self.calculate_adaptive_weights(
                    recent_perf, position, fixture_context
                )
            else:
                weights = self.current_weights
            
            # Calculate weighted prediction
            weighted_pred = 0.0
            total_weight = 0.0
            
            for model, prediction in model_predictions.items():
                if model in weights:
                    weight = weights[model]
                    weighted_pred += prediction * weight
                    total_weight += weight
            
            # Normalize if weights don't sum to 1
            if total_weight > 0:
                weighted_pred = weighted_pred / total_weight
            
            return weighted_pred
            
        except Exception as e:
            self.logger.error(f"Weighted prediction failed: {e}")
            # Fallback to simple average
            return np.mean(list(model_predictions.values()))