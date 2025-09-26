from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
from pathlib import Path
import joblib
from collections import deque
from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from dataclasses import dataclass

from .fpl_client import FPLClient
from .weekly_retrainer import WeeklyMLRetrainer
from .dynamic_ensemble_weights import DynamicEnsembleWeights


@dataclass
class OnlineLearningUpdate:
    """Result from an online learning update"""
    gameweek: int
    samples_processed: int
    mae_before: float
    mae_after: float
    improvement: float
    confidence: float
    models_updated: List[str]


class OnlineLearningSystem:
    """
    Continuous learning system that incrementally improves predictions
    after each gameweek using real FPL results
    """

    def __init__(self):
        self.fpl = FPLClient()
        self.retrainer = WeeklyMLRetrainer()
        self.ensemble_weights = DynamicEnsembleWeights()
        self.logger = logging.getLogger(__name__)

        # Create directories
        Path("models/online").mkdir(parents=True, exist_ok=True)
        Path("data/online_learning").mkdir(parents=True, exist_ok=True)

        # Online learning models
        self.online_models = {}
        self.scalers = {}
        self.feature_names = []

        # Performance buffers (FIFO with fixed size)
        self.performance_buffer = deque(maxlen=50)  # Last 50 updates
        self.prediction_buffer = deque(maxlen=1000)  # Last 1000 predictions

        # Online learning parameters
        self.learning_rates = {
            'sgd_base': 0.01,        # Base SGD learning rate
            'sgd_position': 0.005,   # Position-specific learning rate
            'passive_aggressive': 0.1 # Passive-aggressive learning rate
        }

        # Feature update thresholds
        self.update_thresholds = {
            'min_samples': 5,        # Minimum samples to update
            'max_error': 10.0,       # Skip samples with extreme errors
            'confidence_threshold': 0.3  # Minimum confidence for updates
        }

        # Model state
        self.is_initialized = False
        self.last_update_gameweek = 0

        # Initialize or load existing models
        self._initialize_online_models()

    def _initialize_online_models(self) -> None:
        """Initialize online learning models"""
        try:
            # Try to load existing models
            if self._load_online_models():
                self.logger.info("Loaded existing online learning models")
                return

            # Initialize new models
            self.online_models = {
                # Base SGD model for general predictions
                'sgd_base': SGDRegressor(
                    learning_rate='adaptive',
                    eta0=self.learning_rates['sgd_base'],
                    loss='squared_error',
                    penalty='l2',
                    alpha=0.0001,
                    random_state=42
                ),

                # Position-specific models
                'sgd_gkp': SGDRegressor(
                    learning_rate='adaptive',
                    eta0=self.learning_rates['sgd_position'],
                    loss='squared_error',
                    penalty='l2',
                    alpha=0.0001,
                    random_state=42
                ),

                'sgd_def': SGDRegressor(
                    learning_rate='adaptive',
                    eta0=self.learning_rates['sgd_position'],
                    loss='squared_error',
                    penalty='l2',
                    alpha=0.0001,
                    random_state=42
                ),

                'sgd_mid': SGDRegressor(
                    learning_rate='adaptive',
                    eta0=self.learning_rates['sgd_position'],
                    loss='squared_error',
                    penalty='l2',
                    alpha=0.0001,
                    random_state=42
                ),

                'sgd_fwd': SGDRegressor(
                    learning_rate='adaptive',
                    eta0=self.learning_rates['sgd_position'],
                    loss='squared_error',
                    penalty='l2',
                    alpha=0.0001,
                    random_state=42
                ),

                # Passive-Aggressive model for adaptability
                'passive_aggressive': PassiveAggressiveRegressor(
                    C=self.learning_rates['passive_aggressive'],
                    loss='squared_epsilon_insensitive',
                    epsilon=0.1,
                    random_state=42
                )
            }

            # Initialize scalers
            self.scalers = {model_name: StandardScaler() for model_name in self.online_models.keys()}

            self.logger.info("Initialized new online learning models")

        except Exception as e:
            self.logger.error(f"Failed to initialize online models: {e}")

    def update_with_gameweek_results(self, gameweek: int,
                                   predictions: pd.DataFrame,
                                   force_update: bool = False) -> Optional[OnlineLearningUpdate]:
        """Update models with actual gameweek results"""
        try:
            self.logger.info(f"Starting online learning update for GW{gameweek}")

            # Skip if already updated (unless forced)
            if not force_update and gameweek <= self.last_update_gameweek:
                self.logger.info(f"GW{gameweek} already processed")
                return None

            # Get actual results
            actual_results = self.retrainer.collect_gameweek_results(gameweek)
            if actual_results is None or actual_results.empty:
                self.logger.warning(f"No actual results for GW{gameweek}")
                return None

            # Prepare training data
            training_data = self._prepare_online_training_data(predictions, actual_results)
            if training_data.empty:
                self.logger.warning(f"No training data for GW{gameweek}")
                return None

            # Calculate pre-update performance
            mae_before = self._calculate_current_mae(training_data)

            # Perform online updates
            models_updated = self._perform_online_updates(training_data)

            # Calculate post-update performance
            mae_after = self._calculate_current_mae(training_data)

            # Create update result
            update_result = OnlineLearningUpdate(
                gameweek=gameweek,
                samples_processed=len(training_data),
                mae_before=mae_before,
                mae_after=mae_after,
                improvement=(mae_before - mae_after) / mae_before * 100 if mae_before > 0 else 0,
                confidence=self._calculate_update_confidence(training_data),
                models_updated=models_updated
            )

            # Store in performance buffer
            self.performance_buffer.append(update_result)

            # Update last processed gameweek
            self.last_update_gameweek = gameweek

            # Save models
            self._save_online_models()

            # Log results
            self.logger.info(
                f"Online learning update complete - GW{gameweek}: "
                f"MAE {mae_before:.3f} → {mae_after:.3f} "
                f"({update_result.improvement:+.1f}%)"
            )

            return update_result

        except Exception as e:
            self.logger.error(f"Online learning update failed for GW{gameweek}: {e}")
            return None

    def _prepare_online_training_data(self, predictions: pd.DataFrame,
                                    actual_results: pd.DataFrame) -> pd.DataFrame:
        """Prepare aligned training data for online learning"""
        try:
            # Align predictions with actuals
            merged = predictions.merge(actual_results, left_on='id', right_on='player_id', how='inner')
            if merged.empty:
                return pd.DataFrame()

            # Get prediction column
            pred_col = 'predicted_points' if 'predicted_points' in merged.columns else 'ep_ml'
            if pred_col not in merged.columns:
                self.logger.error("No prediction column found")
                return pd.DataFrame()

            # Filter valid data
            merged = merged[
                (merged[pred_col].notna()) &
                (merged['points'].notna()) &
                (merged['points'] >= 0) &  # Valid points
                (merged[pred_col] >= 0)    # Valid predictions
            ].copy()

            # Calculate prediction error
            merged['prediction_error'] = merged['points'] - merged[pred_col]
            merged['abs_error'] = merged['prediction_error'].abs()

            # Filter extreme errors (likely data quality issues)
            merged = merged[merged['abs_error'] <= self.update_thresholds['max_error']]

            # Ensure minimum samples
            if len(merged) < self.update_thresholds['min_samples']:
                return pd.DataFrame()

            return merged

        except Exception as e:
            self.logger.error(f"Training data preparation failed: {e}")
            return pd.DataFrame()

    def _perform_online_updates(self, training_data: pd.DataFrame) -> List[str]:
        """Perform incremental updates on online models"""
        updated_models = []

        try:
            # Prepare features
            features = self._extract_online_features(training_data)
            if features.empty:
                return updated_models

            targets = training_data['points'].values

            # Update base SGD model
            if self._update_model('sgd_base', features, targets):
                updated_models.append('sgd_base')

            # Update position-specific models
            position_mapping = {1: 'sgd_gkp', 2: 'sgd_def', 3: 'sgd_mid', 4: 'sgd_fwd'}

            for position_id, model_name in position_mapping.items():
                position_data = training_data[training_data.get('element_type', 0) == position_id]
                if len(position_data) >= 2:  # Need minimum samples for position-specific updates
                    position_features = features.loc[position_data.index]
                    position_targets = position_data['points'].values

                    if self._update_model(model_name, position_features, position_targets):
                        updated_models.append(model_name)

            # Update Passive-Aggressive model
            if self._update_model('passive_aggressive', features, targets):
                updated_models.append('passive_aggressive')

            return updated_models

        except Exception as e:
            self.logger.error(f"Model updates failed: {e}")
            return updated_models

    def _update_model(self, model_name: str, features: pd.DataFrame, targets: np.ndarray) -> bool:
        """Update a specific online learning model"""
        try:
            if model_name not in self.online_models or model_name not in self.scalers:
                return False

            model = self.online_models[model_name]
            scaler = self.scalers[model_name]

            # Scale features
            if not hasattr(scaler, 'mean_') or scaler.mean_ is None:
                # First time - fit scaler
                scaled_features = scaler.fit_transform(features)
            else:
                # Incremental scaling update
                scaled_features = scaler.transform(features)
                # Update scaler with new data (partial_fit if available)
                if hasattr(scaler, 'partial_fit'):
                    scaler.partial_fit(features)

            # Perform online update
            if not hasattr(model, 'coef_') or model.coef_ is None:
                # First time - initial fit
                model.fit(scaled_features, targets)
            else:
                # Incremental update
                model.partial_fit(scaled_features, targets)

            return True

        except Exception as e:
            self.logger.error(f"Failed to update model {model_name}: {e}")
            return False

    def _extract_online_features(self, training_data: pd.DataFrame) -> pd.DataFrame:
        """Extract features for online learning"""
        try:
            # Use subset of key features that are most predictive and stable
            feature_cols = [
                'total_points', 'form', 'now_cost',
                'minutes', 'goals_scored', 'assists',
                'bonus', 'bps', 'influence', 'creativity', 'threat',
                'element_type', 'selected'
            ]

            # Ensure columns exist
            available_cols = [col for col in feature_cols if col in training_data.columns]

            if not available_cols:
                self.logger.error("No feature columns available")
                return pd.DataFrame()

            features = training_data[available_cols].copy()

            # Convert price
            if 'now_cost' in features.columns:
                features['now_cost'] = features['now_cost'] / 10

            # Add derived features
            features['points_per_game'] = features.get('total_points', 0) / max(1, len(training_data))
            features['value_score'] = features.get('total_points', 0) / features.get('now_cost', 1)

            # Handle missing values
            features = features.fillna(0)

            # Store feature names for later use
            if not self.feature_names:
                self.feature_names = list(features.columns)

            return features

        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return pd.DataFrame()

    def _calculate_current_mae(self, training_data: pd.DataFrame) -> float:
        """Calculate current model MAE on training data"""
        try:
            pred_col = 'predicted_points' if 'predicted_points' in training_data.columns else 'ep_ml'
            if pred_col not in training_data.columns:
                return 0.0

            predictions = training_data[pred_col].values
            actuals = training_data['points'].values

            return mean_absolute_error(actuals, predictions)

        except Exception as e:
            self.logger.error(f"MAE calculation failed: {e}")
            return 0.0

    def _calculate_update_confidence(self, training_data: pd.DataFrame) -> float:
        """Calculate confidence in the online update"""
        try:
            # Factors affecting confidence:
            # 1. Sample size
            sample_size_factor = min(1.0, len(training_data) / 50.0)

            # 2. Error distribution (lower variance = higher confidence)
            errors = training_data['abs_error'].values
            error_std = np.std(errors)
            error_factor = max(0.1, min(1.0, 1.0 / (1.0 + error_std)))

            # 3. Prediction quality (closer predictions = higher confidence)
            pred_col = 'predicted_points' if 'predicted_points' in training_data.columns else 'ep_ml'
            if pred_col in training_data.columns:
                mae = mean_absolute_error(training_data['points'], training_data[pred_col])
                prediction_factor = max(0.1, min(1.0, 1.0 / (1.0 + mae / 2.0)))
            else:
                prediction_factor = 0.5

            # Combined confidence
            confidence = (sample_size_factor + error_factor + prediction_factor) / 3.0
            return min(1.0, max(0.1, confidence))

        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {e}")
            return 0.5

    def predict_with_online_models(self, player_data: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions using online learning models"""
        try:
            if not self.is_initialized:
                self.logger.warning("Online models not initialized")
                return pd.DataFrame()

            # Extract features
            features = self._extract_prediction_features(player_data)
            if features.empty:
                return pd.DataFrame()

            predictions = []

            for _, player_row in player_data.iterrows():
                player_features = features.loc[player_row.name:player_row.name]

                # Get position-specific model
                position_id = player_row.get('element_type', 3)
                position_model_name = {1: 'sgd_gkp', 2: 'sgd_def', 3: 'sgd_mid', 4: 'sgd_fwd'}.get(position_id, 'sgd_mid')

                # Make predictions with different models
                model_predictions = {}

                for model_name in ['sgd_base', position_model_name, 'passive_aggressive']:
                    if model_name in self.online_models and model_name in self.scalers:
                        try:
                            scaled_features = self.scalers[model_name].transform(player_features)
                            pred = self.online_models[model_name].predict(scaled_features)[0]
                            model_predictions[model_name] = max(0, min(50, pred))  # Reasonable bounds
                        except Exception:
                            continue

                # Ensemble prediction (average of available models)
                if model_predictions:
                    online_prediction = np.mean(list(model_predictions.values()))
                else:
                    online_prediction = 0.0

                predictions.append({
                    'player_id': player_row.get('id', 0),
                    'online_prediction': online_prediction,
                    'model_count': len(model_predictions),
                    'individual_predictions': model_predictions
                })

            return pd.DataFrame(predictions)

        except Exception as e:
            self.logger.error(f"Online prediction failed: {e}")
            return pd.DataFrame()

    def _extract_prediction_features(self, player_data: pd.DataFrame) -> pd.DataFrame:
        """Extract features for prediction (similar to training features)"""
        try:
            # Use same features as training
            if not self.feature_names:
                return pd.DataFrame()

            features = pd.DataFrame()

            for col in self.feature_names:
                if col in player_data.columns:
                    features[col] = player_data[col]
                elif col == 'points_per_game':
                    features[col] = player_data.get('total_points', 0) / max(1, 10)  # Assume ~10 games
                elif col == 'value_score':
                    price = player_data.get('now_cost', 50) / 10
                    features[col] = player_data.get('total_points', 0) / max(0.1, price)
                else:
                    features[col] = 0

            # Convert price
            if 'now_cost' in features.columns:
                features['now_cost'] = features['now_cost'] / 10

            return features.fillna(0)

        except Exception as e:
            self.logger.error(f"Prediction feature extraction failed: {e}")
            return pd.DataFrame()

    def get_online_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about online learning performance"""
        try:
            if not self.performance_buffer:
                return {}

            recent_updates = list(self.performance_buffer)[-10:]  # Last 10 updates

            stats = {
                'total_updates': len(self.performance_buffer),
                'last_update_gameweek': self.last_update_gameweek,
                'average_improvement': np.mean([u.improvement for u in recent_updates]),
                'average_confidence': np.mean([u.confidence for u in recent_updates]),
                'total_samples_processed': sum(u.samples_processed for u in recent_updates),
                'models_status': {name: 'active' if hasattr(model, 'coef_') else 'inactive'
                                for name, model in self.online_models.items()}
            }

            return stats

        except Exception as e:
            self.logger.error(f"Stats calculation failed: {e}")
            return {}

    def _save_online_models(self) -> None:
        """Save online learning models and scalers"""
        try:
            # Save models
            for model_name, model in self.online_models.items():
                if hasattr(model, 'coef_'):
                    joblib.dump(model, f"models/online/{model_name}.pkl")

            # Save scalers
            for scaler_name, scaler in self.scalers.items():
                if hasattr(scaler, 'mean_'):
                    joblib.dump(scaler, f"models/online/{scaler_name}_scaler.pkl")

            # Save metadata
            metadata = {
                'feature_names': self.feature_names,
                'last_update_gameweek': self.last_update_gameweek,
                'is_initialized': self.is_initialized
            }
            joblib.dump(metadata, "models/online/metadata.pkl")

            self.logger.info("Online learning models saved")

        except Exception as e:
            self.logger.error(f"Failed to save online models: {e}")

    def _load_online_models(self) -> bool:
        """Load existing online learning models"""
        try:
            # Check if models exist
            model_files = [f"models/online/{name}.pkl" for name in ['sgd_base', 'sgd_gkp', 'sgd_def', 'sgd_mid', 'sgd_fwd', 'passive_aggressive']]

            if not all(Path(f).exists() for f in model_files):
                return False

            # Load models
            for model_name in self.online_models.keys():
                model_file = f"models/online/{model_name}.pkl"
                if Path(model_file).exists():
                    self.online_models[model_name] = joblib.load(model_file)

            # Load scalers
            for scaler_name in self.scalers.keys():
                scaler_file = f"models/online/{scaler_name}_scaler.pkl"
                if Path(scaler_file).exists():
                    self.scalers[scaler_name] = joblib.load(scaler_file)

            # Load metadata
            metadata_file = "models/online/metadata.pkl"
            if Path(metadata_file).exists():
                metadata = joblib.load(metadata_file)
                self.feature_names = metadata.get('feature_names', [])
                self.last_update_gameweek = metadata.get('last_update_gameweek', 0)
                self.is_initialized = metadata.get('is_initialized', False)

            self.is_initialized = True
            return True

        except Exception as e:
            self.logger.error(f"Failed to load online models: {e}")
            return False