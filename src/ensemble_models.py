from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib
import warnings
warnings.filterwarnings('ignore')


class MultiModelEnsemble:
    """Advanced multi-model ensemble for FPL predictions"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self.xgb_model = None  # Will be passed from main pipeline
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )
        self.nn_model = MLPRegressor(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42
        )
        
        # Scalers for neural network
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        # Model weights (learned during training)
        self.model_weights = {
            'xgb': 0.5,      # XGBoost - good baseline
            'rf': 0.3,       # Random Forest - captures non-linear patterns
            'nn': 0.2        # Neural Network - learns complex interactions
        }
        
        # Performance tracking
        self.model_performances = {}
        self.feature_names = []
        
    def train_ensemble(self, X: pd.DataFrame, y: pd.Series, xgb_model=None) -> Dict[str, Any]:
        """Train all models in the ensemble"""
        try:
            if X.empty or y.empty:
                return {"status": "failed", "reason": "No training data"}
            
            self.feature_names = list(X.columns)
            
            # Store XGBoost model from main pipeline
            if xgb_model:
                self.xgb_model = xgb_model
            
            # Prepare data
            X_clean = self._prepare_features(X)
            y_clean = self._prepare_targets(y)
            
            if len(X_clean) == 0:
                return {"status": "failed", "reason": "No valid data after cleaning"}
            
            results = {}
            
            # Train Random Forest
            print("Training Random Forest...")
            rf_score = self._train_random_forest(X_clean, y_clean)
            results['random_forest'] = rf_score
            
            # Train Neural Network
            print("Training Neural Network...")
            nn_score = self._train_neural_network(X_clean, y_clean)
            results['neural_network'] = nn_score
            
            # Evaluate XGBoost if available
            if self.xgb_model:
                print("Evaluating XGBoost...")
                xgb_score = self._evaluate_xgboost(X_clean, y_clean)
                results['xgboost'] = xgb_score
            
            # Calculate optimal ensemble weights
            self._calculate_optimal_weights(X_clean, y_clean)
            
            # Overall ensemble performance
            ensemble_score = self._evaluate_ensemble(X_clean, y_clean)
            results['ensemble'] = ensemble_score
            
            print(f"Ensemble training complete. Final score: {ensemble_score:.4f}")
            
            return {
                "status": "success",
                "model_scores": results,
                "model_weights": self.model_weights,
                "features_count": len(self.feature_names),
                "training_samples": len(X_clean)
            }
            
        except Exception as e:
            self.logger.error(f"Ensemble training failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def predict_ensemble(self, X: pd.DataFrame) -> np.ndarray:
        """Generate ensemble predictions combining all models"""
        try:
            if X.empty:
                return np.array([])
            
            X_clean = self._prepare_features(X)
            if len(X_clean) == 0:
                return np.zeros(len(X))
            
            predictions = {}
            
            # Get XGBoost predictions
            if self.xgb_model:
                try:
                    xgb_pred = self.xgb_model.predict(X_clean)
                    predictions['xgb'] = np.array(xgb_pred)
                except Exception as e:
                    print(f"XGBoost prediction failed: {e}")
                    predictions['xgb'] = np.zeros(len(X_clean))
            else:
                predictions['xgb'] = np.zeros(len(X_clean))
            
            # Get Random Forest predictions
            try:
                rf_pred = self.rf_model.predict(X_clean)
                predictions['rf'] = np.array(rf_pred)
            except Exception as e:
                print(f"Random Forest prediction failed: {e}")
                predictions['rf'] = np.zeros(len(X_clean))
            
            # Get Neural Network predictions
            try:
                X_scaled = self.feature_scaler.transform(X_clean)
                nn_pred_scaled = self.nn_model.predict(X_scaled)
                nn_pred = self.target_scaler.inverse_transform(nn_pred_scaled.reshape(-1, 1)).flatten()
                predictions['nn'] = np.array(nn_pred)
            except Exception as e:
                print(f"Neural Network prediction failed: {e}")
                predictions['nn'] = np.zeros(len(X_clean))
            
            # Ensemble combination
            ensemble_pred = (
                self.model_weights['xgb'] * predictions['xgb'] +
                self.model_weights['rf'] * predictions['rf'] +
                self.model_weights['nn'] * predictions['nn']
            )
            
            # Apply bounds and clean predictions
            ensemble_pred = np.clip(ensemble_pred, 0, 25)  # Reasonable FPL bounds
            ensemble_pred = np.nan_to_num(ensemble_pred, nan=0.0)
            
            return ensemble_pred
            
        except Exception as e:
            self.logger.error(f"Ensemble prediction failed: {e}")
            return np.zeros(len(X))
    
    def get_model_contributions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Get individual model predictions for analysis"""
        try:
            X_clean = self._prepare_features(X)
            contributions = {}
            
            # XGBoost contribution
            if self.xgb_model:
                try:
                    contributions['XGBoost'] = self.xgb_model.predict(X_clean)
                except:
                    contributions['XGBoost'] = np.zeros(len(X_clean))
            
            # Random Forest contribution
            try:
                contributions['Random_Forest'] = self.rf_model.predict(X_clean)
            except:
                contributions['Random_Forest'] = np.zeros(len(X_clean))
            
            # Neural Network contribution
            try:
                X_scaled = self.feature_scaler.transform(X_clean)
                nn_pred = self.nn_model.predict(X_scaled)
                contributions['Neural_Network'] = self.target_scaler.inverse_transform(nn_pred.reshape(-1, 1)).flatten()
            except:
                contributions['Neural_Network'] = np.zeros(len(X_clean))
            
            return contributions
            
        except Exception as e:
            self.logger.error(f"Failed to get model contributions: {e}")
            return {}
    
    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare features for modeling"""
        try:
            # Select numeric columns
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X_clean = X[numeric_cols].copy()
            
            # Handle infinite values
            X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
            
            # Fill NaN values
            X_clean = X_clean.fillna(X_clean.median().fillna(0))
            
            # Remove constant columns
            constant_cols = X_clean.columns[X_clean.var() == 0]
            X_clean = X_clean.drop(columns=constant_cols)
            
            return X_clean
            
        except Exception as e:
            self.logger.error(f"Feature preparation failed: {e}")
            return pd.DataFrame()
    
    def _prepare_targets(self, y: pd.Series) -> pd.Series:
        """Clean and prepare target variable"""
        try:
            y_clean = pd.to_numeric(y, errors='coerce')
            y_clean = y_clean.fillna(0)  # Fill NaN with 0
            y_clean = np.clip(y_clean, 0, 25)  # Clip to reasonable bounds
            return y_clean
        except Exception as e:
            self.logger.error(f"Target preparation failed: {e}")
            return pd.Series([0] * len(y))
    
    def _train_random_forest(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Train Random Forest model"""
        try:
            self.rf_model.fit(X, y)
            
            # Cross-validation score
            cv_scores = cross_val_score(self.rf_model, X, y, cv=3, scoring='neg_mean_absolute_error')
            score = -cv_scores.mean()
            
            self.model_performances['rf'] = score
            return score
            
        except Exception as e:
            self.logger.error(f"Random Forest training failed: {e}")
            return float('inf')
    
    def _train_neural_network(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Train Neural Network model"""
        try:
            # Scale features
            X_scaled = self.feature_scaler.fit_transform(X)
            
            # Scale targets
            y_scaled = self.target_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
            
            # Train model
            self.nn_model.fit(X_scaled, y_scaled)
            
            # Evaluate performance
            y_pred_scaled = self.nn_model.predict(X_scaled)
            y_pred = self.target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            
            score = np.mean(np.abs(y - y_pred))
            self.model_performances['nn'] = score
            
            return score
            
        except Exception as e:
            self.logger.error(f"Neural Network training failed: {e}")
            return float('inf')
    
    def _evaluate_xgboost(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Evaluate XGBoost model performance"""
        try:
            y_pred = self.xgb_model.predict(X)
            score = np.mean(np.abs(y - y_pred))
            self.model_performances['xgb'] = score
            return score
        except Exception as e:
            self.logger.error(f"XGBoost evaluation failed: {e}")
            return float('inf')
    
    def _calculate_optimal_weights(self, X: pd.DataFrame, y: pd.Series):
        """Calculate optimal ensemble weights based on performance"""
        try:
            # Get individual model errors
            performances = self.model_performances
            
            if not performances:
                return  # Keep default weights
            
            # Convert errors to weights (inverse relationship)
            # Add small epsilon to avoid division by zero
            eps = 0.001
            weights = {}
            
            for model, error in performances.items():
                if error == float('inf') or error > 10:  # Bad performance
                    weights[model] = eps
                else:
                    weights[model] = 1.0 / (error + eps)
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                for model in weights:
                    weights[model] = weights[model] / total_weight
                
                # Update model weights
                if 'xgb' in weights:
                    self.model_weights['xgb'] = weights['xgb']
                if 'rf' in weights:
                    self.model_weights['rf'] = weights['rf']
                if 'nn' in weights:
                    self.model_weights['nn'] = weights['nn']
            
            print(f"Optimized ensemble weights: {self.model_weights}")
            
        except Exception as e:
            self.logger.error(f"Weight calculation failed: {e}")
            # Keep default weights
    
    def _evaluate_ensemble(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Evaluate ensemble performance"""
        try:
            ensemble_pred = self.predict_ensemble(pd.DataFrame(X, columns=self.feature_names))
            if len(ensemble_pred) != len(y):
                return float('inf')
            
            score = np.mean(np.abs(y - ensemble_pred))
            return score
            
        except Exception as e:
            self.logger.error(f"Ensemble evaluation failed: {e}")
            return float('inf')
    
    def save_ensemble(self, filepath: str):
        """Save trained ensemble models"""
        try:
            ensemble_data = {
                'rf_model': self.rf_model,
                'nn_model': self.nn_model,
                'feature_scaler': self.feature_scaler,
                'target_scaler': self.target_scaler,
                'model_weights': self.model_weights,
                'model_performances': self.model_performances,
                'feature_names': self.feature_names
            }
            
            joblib.dump(ensemble_data, filepath)
            print(f"Ensemble models saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save ensemble: {e}")
    
    def load_ensemble(self, filepath: str) -> bool:
        """Load trained ensemble models"""
        try:
            ensemble_data = joblib.load(filepath)
            
            self.rf_model = ensemble_data['rf_model']
            self.nn_model = ensemble_data['nn_model']
            self.feature_scaler = ensemble_data['feature_scaler']
            self.target_scaler = ensemble_data['target_scaler']
            self.model_weights = ensemble_data['model_weights']
            self.model_performances = ensemble_data['model_performances']
            self.feature_names = ensemble_data['feature_names']
            
            print(f"Ensemble models loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load ensemble: {e}")
            return False