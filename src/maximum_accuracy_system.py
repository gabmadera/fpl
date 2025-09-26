"""
Maximum Accuracy FPL System
Target: 70-75%+ accuracy with aggressive prediction strategies
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import joblib
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

# ML imports
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .fpl_client import FPLClient
from .cache_manager import CacheManager
from .data_prep import DataPrep
from .feature_engineer import FeatureEngineer


class AccuracyTarget(Enum):
    CONSERVATIVE = "conservative"  # 55-60%
    BALANCED = "balanced"        # 60-65%
    AGGRESSIVE = "aggressive"    # 65-70%
    MAXIMUM = "maximum"         # 70-75%+


@dataclass
class AccuracyMetrics:
    """Comprehensive accuracy tracking"""
    gameweek: int
    timestamp: datetime

    # Overall accuracy
    overall_accuracy: float
    position_accuracy: Dict[str, float]
    captain_accuracy: float

    # Prediction quality
    mae: float
    rmse: float
    r2: float

    # Strategy metrics
    differential_success_rate: float
    value_pick_accuracy: float
    premium_player_accuracy: float

    # Model performance
    ensemble_weight: Dict[str, float]
    feature_importance: Dict[str, float]

    # Target tracking
    target_progress: float
    weeks_to_target: Optional[int]


@dataclass
class ModelConfig:
    """Configuration for maximum accuracy system"""
    target_accuracy: AccuracyTarget = AccuracyTarget.MAXIMUM

    # Ensemble weights (will be dynamically adjusted)
    xgb_weight: float = 0.4
    lgb_weight: float = 0.35
    rf_weight: float = 0.25

    # Feature engineering
    use_advanced_features: bool = True
    feature_selection_k: int = 150

    # Aggressive strategies
    differential_threshold: float = 0.15  # 15% ownership threshold
    risk_tolerance: float = 0.8  # Higher = more aggressive
    captain_confidence_threshold: float = 0.7

    # Retraining
    retrain_frequency: int = 1  # Every gameweek
    min_data_points: int = 50

    # Performance thresholds
    accuracy_improvement_threshold: float = 0.02  # 2% improvement required
    underperformance_threshold: float = 0.05  # 5% below target triggers adjustment


class MaximumAccuracySystem:
    """
    Production-ready system for maximum FPL prediction accuracy
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.logger = logging.getLogger(__name__)

        # Core components
        self.fpl = FPLClient()
        self.cache_manager = CacheManager()
        self.data_prep = DataPrep()
        self.feature_engineer = FeatureEngineer()

        # ML models
        self.models = {}
        self.ensemble_weights = {
            'xgb': self.config.xgb_weight,
            'lgb': self.config.lgb_weight,
            'rf': self.config.rf_weight
        }

        # Accuracy tracking
        self.accuracy_history: List[AccuracyMetrics] = []
        self.target_timeline = self._build_target_timeline()

        # Model paths
        self.model_dir = Path("models/maximum_accuracy")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Initialize logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup comprehensive logging"""
        log_file = Path("logs/maximum_accuracy_system.log")
        log_file.parent.mkdir(exist_ok=True)

        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def _build_target_timeline(self) -> Dict[int, float]:
        """Build target accuracy progression timeline"""
        timeline = {}

        if self.config.target_accuracy == AccuracyTarget.MAXIMUM:
            # Aggressive timeline for 70-75%+ target
            timeline = {
                1: 0.55,   # Week 1-2: Initial improvements (55%)
                2: 0.58,
                3: 0.60,   # Week 3-5: Model adaptation (60-65%)
                4: 0.62,
                5: 0.65,
                6: 0.67,   # Week 6-10: Target achievement (65-70%)
                7: 0.69,
                8: 0.70,
                9: 0.72,
                10: 0.74,  # Week 10+: Consistent excellence (70-75%+)
                15: 0.75,
                20: 0.76
            }

        return timeline

    def build_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build advanced features for maximum accuracy"""
        try:
            self.logger.info("Building advanced features for maximum accuracy")

            # Start with base features
            features_df = self.feature_engineer.create_features(df)

            # Advanced attacking metrics
            features_df = self._add_attacking_metrics(features_df)

            # Defensive metrics
            features_df = self._add_defensive_metrics(features_df)

            # Form and momentum indicators
            features_df = self._add_momentum_features(features_df)

            # Opposition analysis
            features_df = self._add_opposition_features(features_df)

            # Value and ownership features
            features_df = self._add_value_features(features_df)

            # Situational features
            features_df = self._add_situational_features(features_df)

            # Interaction features
            features_df = self._add_interaction_features(features_df)

            self.logger.info(f"Built {len(features_df.columns)} advanced features")
            return features_df

        except Exception as e:
            self.logger.error(f"Error building advanced features: {e}")
            return df

    def _add_attacking_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced attacking metrics"""
        # xG efficiency
        if 'xg_per90' in df.columns and 'goals_scored' in df.columns:
            df['xg_efficiency'] = df['goals_scored'] / (df['xg_per90'] * df.get('minutes', 90) / 90).replace(0, 1)

        # xA efficiency
        if 'xa_per90' in df.columns and 'assists' in df.columns:
            df['xa_efficiency'] = df['assists'] / (df['xa_per90'] * df.get('minutes', 90) / 90).replace(0, 1)

        # Attacking threat combined
        df['attacking_threat'] = (
            df.get('xg_per90', 0) * 1.2 +
            df.get('xa_per90', 0) * 1.0 +
            df.get('bonus', 0) / 10
        )

        # Position-adjusted threat
        position_multipliers = {'FWD': 1.3, 'MID': 1.1, 'DEF': 0.7, 'GKP': 0.3}
        df['position_adj_threat'] = df['attacking_threat'] * df.get('position', 'MID').map(position_multipliers).fillna(1.0)

        return df

    def _add_defensive_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add defensive metrics"""
        # Clean sheet probability
        df['cs_probability'] = np.where(
            df.get('position', '').isin(['GKP', 'DEF']),
            (df.get('clean_sheets', 0) + 1) / (df.get('minutes', 90) / 90 + 1),
            0
        )

        # Defensive actions (estimated)
        df['defensive_actions'] = (
            df.get('saves', 0) * 0.5 +
            df.get('bonus', 0) * 0.3 +
            (4 - df.get('goals_conceded', 2)) * 0.2
        )

        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add form and momentum indicators"""
        # Form vs season average
        df['form_vs_average'] = df.get('form', 0) - df.get('points_per_game', 0)

        # Recent form trend (estimated from form)
        df['form_momentum'] = np.where(
            df.get('form', 0) > df.get('points_per_game', 0) * 1.2, 1.5,  # Hot
            np.where(df.get('form', 0) < df.get('points_per_game', 0) * 0.8, 0.7, 1.0)  # Cold
        )

        # Value momentum
        df['value_momentum'] = (
            df.get('value_form', 0) * 0.7 +
            df.get('value_season', 0) * 0.3
        )

        return df

    def _add_opposition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add opposition analysis features"""
        try:
            # Get fixture difficulty
            fixtures = pd.DataFrame(self.fpl.fixtures())
            current_gw = self.fpl.next_active_gameweek()

            if not fixtures.empty:
                current_fixtures = fixtures[fixtures['event'] == current_gw]

                # Build opposition mapping
                opposition_map = {}
                for _, fixture in current_fixtures.iterrows():
                    home_team = fixture['team_h']
                    away_team = fixture['team_a']
                    home_diff = fixture.get('team_h_difficulty', 3)
                    away_diff = fixture.get('team_a_difficulty', 3)

                    opposition_map[home_team] = {
                        'difficulty': away_diff,
                        'is_home': True,
                        'opponent': away_team
                    }
                    opposition_map[away_team] = {
                        'difficulty': home_diff,
                        'is_home': False,
                        'opponent': home_team
                    }

                # Map to dataframe
                df['fixture_difficulty'] = df.get('team_id', 0).map(
                    lambda x: opposition_map.get(x, {}).get('difficulty', 3)
                )
                df['is_home'] = df.get('team_id', 0).map(
                    lambda x: opposition_map.get(x, {}).get('is_home', True)
                )

                # Home advantage
                df['home_advantage'] = np.where(df['is_home'], 1.1, 0.9)

                # Fixture-adjusted threat
                df['fixture_adj_threat'] = df.get('attacking_threat', 0) * (5 - df['fixture_difficulty']) / 5

        except Exception as e:
            self.logger.warning(f"Could not add opposition features: {e}")
            df['fixture_difficulty'] = 3
            df['is_home'] = True
            df['home_advantage'] = 1.0
            df['fixture_adj_threat'] = df.get('attacking_threat', 0)

        return df

    def _add_value_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add value and ownership features"""
        # Points per million
        df['points_per_million'] = df.get('total_points', 0) / (df.get('price', 4.0) + 0.1)

        # Form per million
        df['form_per_million'] = df.get('form', 0) / (df.get('price', 4.0) + 0.1)

        # Ownership category
        ownership = df.get('selected_by_percent', 5.0)
        df['ownership_category'] = pd.cut(
            ownership,
            bins=[0, 2, 10, 25, 100],
            labels=['differential', 'low', 'medium', 'high']
        )

        # Value vs ownership
        df['value_vs_ownership'] = df['points_per_million'] / (ownership + 1)

        return df

    def _add_situational_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add situational context features"""
        # Injury risk
        df['injury_risk'] = np.where(
            df.get('chance_of_playing_next_round', 100) < 100,
            1 - df.get('chance_of_playing_next_round', 100) / 100,
            0
        )

        # Rotation risk (based on price and position)
        rotation_risk = np.where(
            df.get('price', 5.0) < 5.0, 0.3,  # Budget players higher risk
            np.where(df.get('price', 5.0) > 10.0, 0.1, 0.2)  # Premium lower risk
        )
        df['rotation_risk'] = rotation_risk

        # Captain potential
        df['captain_potential'] = (
            df.get('attacking_threat', 0) * 0.4 +
            df.get('form', 0) * 0.3 +
            df.get('fixture_adj_threat', 0) * 0.3
        )

        return df

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between important variables"""
        # Form * Price interaction
        df['form_price_interaction'] = df.get('form', 0) * np.log(df.get('price', 4.0) + 1)

        # Threat * Fixture interaction
        df['threat_fixture_interaction'] = df.get('attacking_threat', 0) * (5 - df.get('fixture_difficulty', 3))

        # Position * Price efficiency
        pos_price_map = {'GKP': 0.8, 'DEF': 1.0, 'MID': 1.2, 'FWD': 1.1}
        df['position_price_efficiency'] = (
            df.get('points_per_million', 0) *
            df.get('position', 'MID').map(pos_price_map).fillna(1.0)
        )

        return df

    def build_ensemble_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Build advanced ensemble models for maximum accuracy"""
        try:
            self.logger.info("Building ensemble models for maximum accuracy")

            # Feature selection
            if self.config.use_advanced_features and len(X.columns) > self.config.feature_selection_k:
                selector = SelectKBest(score_func=f_regression, k=self.config.feature_selection_k)
                X_selected = selector.fit_transform(X, y)
                selected_features = X.columns[selector.get_support()].tolist()
                X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
                self.logger.info(f"Selected {len(selected_features)} best features")

            # Scale features
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

            models = {}

            # XGBoost with advanced hyperparameters
            models['xgb'] = xgb.XGBRegressor(
                n_estimators=500,
                learning_rate=0.02,
                max_depth=8,
                subsample=0.85,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                early_stopping_rounds=50,
                eval_metric='mae'
            )

            # LightGBM with optimization
            models['lgb'] = lgb.LGBMRegressor(
                n_estimators=500,
                learning_rate=0.025,
                num_leaves=64,
                feature_fraction=0.8,
                bagging_fraction=0.85,
                bagging_freq=1,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42
            )

            # Random Forest for stability
            models['rf'] = RandomForestRegressor(
                n_estimators=300,
                max_depth=15,
                min_samples_split=8,
                min_samples_leaf=4,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )

            # Train individual models
            model_scores = {}
            for name, model in models.items():
                self.logger.info(f"Training {name} model")

                # Time series cross-validation
                tscv = TimeSeriesSplit(n_splits=5)
                scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='neg_mean_absolute_error')
                model_scores[name] = -scores.mean()

                # Fit full model
                if name == 'xgb':
                    # Split for early stopping
                    split_idx = int(len(X_scaled) * 0.8)
                    model.fit(
                        X_scaled[:split_idx], y[:split_idx],
                        eval_set=[(X_scaled[split_idx:], y[split_idx:])],
                        verbose=False
                    )
                else:
                    model.fit(X_scaled, y)

                self.logger.info(f"{name} CV MAE: {model_scores[name]:.3f}")

            # Dynamic ensemble weighting based on performance
            total_inverse_error = sum(1 / score for score in model_scores.values())
            dynamic_weights = {
                name: (1 / score) / total_inverse_error
                for name, score in model_scores.items()
            }

            self.logger.info(f"Dynamic ensemble weights: {dynamic_weights}")

            # Update ensemble weights
            self.ensemble_weights.update(dynamic_weights)

            # Save models and metadata
            model_data = {
                'models': models,
                'scaler': scaler,
                'feature_names': X.columns.tolist(),
                'ensemble_weights': dynamic_weights,
                'model_scores': model_scores,
                'training_date': datetime.now().isoformat()
            }

            # Save to disk
            joblib.dump(model_data, self.model_dir / "ensemble_models.pkl")

            self.models = models
            return model_data

        except Exception as e:
            self.logger.error(f"Error building ensemble models: {e}")
            raise

    def predict_with_ensemble(self, X: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Make predictions using ensemble with confidence estimation"""
        try:
            # Load models if not in memory
            if not self.models:
                model_data = joblib.load(self.model_dir / "ensemble_models.pkl")
                self.models = model_data['models']
                self.ensemble_weights = model_data['ensemble_weights']

            # Get individual predictions
            individual_predictions = {}
            for name, model in self.models.items():
                individual_predictions[name] = model.predict(X)

            # Ensemble prediction with dynamic weights
            ensemble_pred = np.zeros(len(X))
            for name, pred in individual_predictions.items():
                weight = self.ensemble_weights.get(name, 1/len(self.models))
                ensemble_pred += pred * weight

            return ensemble_pred, individual_predictions

        except Exception as e:
            self.logger.error(f"Error in ensemble prediction: {e}")
            # Fallback to simple average
            if self.models:
                predictions = [model.predict(X) for model in self.models.values()]
                return np.mean(predictions, axis=0), {}
            else:
                return np.zeros(len(X)), {}

    def apply_aggressive_strategies(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Apply aggressive strategies for maximum accuracy"""
        try:
            df = predictions_df.copy()

            # Differential picks boost
            low_ownership = df['selected_by_percent'] < (self.config.differential_threshold * 100)
            df.loc[low_ownership, 'predicted_points'] *= 1.15  # 15% boost for differentials

            # Captain confidence boost
            high_confidence = df['captain_potential'] > self.config.captain_confidence_threshold
            df.loc[high_confidence, 'predicted_points'] *= 1.1  # 10% boost for captaincy

            # Value picks enhancement
            value_threshold = df['points_per_million'].quantile(0.8)
            value_picks = df['points_per_million'] > value_threshold
            df.loc[value_picks, 'predicted_points'] *= 1.08  # 8% boost for value

            # Risk tolerance adjustment
            if self.config.risk_tolerance > 0.7:
                # Boost explosive players
                explosive_potential = (
                    df['attacking_threat'] > df['attacking_threat'].quantile(0.85)
                ) & (df['form_momentum'] > 1.2)
                df.loc[explosive_potential, 'predicted_points'] *= 1.2  # 20% boost

            # Penalty for injury risk
            injury_penalty = 1 - (df['injury_risk'] * 0.3)
            df['predicted_points'] *= injury_penalty

            return df

        except Exception as e:
            self.logger.error(f"Error applying aggressive strategies: {e}")
            return predictions_df

    def evaluate_accuracy(self, predictions: pd.DataFrame, actual_points: pd.DataFrame,
                         gameweek: int) -> AccuracyMetrics:
        """Comprehensive accuracy evaluation"""
        try:
            # Merge predictions with actual points
            merged = predictions.merge(
                actual_points,
                left_on='player_id',
                right_on='player_id',
                how='inner'
            )

            if merged.empty:
                raise ValueError("No matching data for accuracy evaluation")

            # Overall accuracy metrics
            pred_points = merged['predicted_points']
            actual_points_col = merged['total_points']

            mae = mean_absolute_error(actual_points_col, pred_points)
            rmse = np.sqrt(mean_squared_error(actual_points_col, pred_points))
            r2 = r2_score(actual_points_col, pred_points)

            # Calculate accuracy percentage (within ±2 points)
            within_range = np.abs(pred_points - actual_points_col) <= 2
            overall_accuracy = within_range.mean()

            # Position-specific accuracy
            position_accuracy = {}
            for pos in ['GKP', 'DEF', 'MID', 'FWD']:
                pos_mask = merged['position'] == pos
                if pos_mask.sum() > 0:
                    pos_within_range = within_range[pos_mask]
                    position_accuracy[pos] = pos_within_range.mean()

            # Captain accuracy (top predicted player vs actual top scorer)
            captain_pred = merged.loc[merged['predicted_points'].idxmax(), 'player_id']
            actual_top = merged.loc[merged['total_points'].idxmax(), 'player_id']
            captain_accuracy = 1.0 if captain_pred == actual_top else 0.0

            # Differential success rate
            differentials = merged['selected_by_percent'] < 15
            if differentials.sum() > 0:
                diff_success = within_range[differentials].mean()
            else:
                diff_success = 0.0

            # Value pick accuracy
            value_picks = merged['points_per_million'] > merged['points_per_million'].quantile(0.8)
            if value_picks.sum() > 0:
                value_accuracy = within_range[value_picks].mean()
            else:
                value_accuracy = 0.0

            # Premium player accuracy
            premium_players = merged['price'] >= 10.0
            if premium_players.sum() > 0:
                premium_accuracy = within_range[premium_players].mean()
            else:
                premium_accuracy = 0.0

            # Calculate target progress
            target_accuracy = self.target_timeline.get(gameweek, 0.75)
            target_progress = overall_accuracy / target_accuracy

            # Estimate weeks to target
            if overall_accuracy < target_accuracy:
                improvement_rate = 0.02  # 2% per week estimate
                weeks_to_target = int((target_accuracy - overall_accuracy) / improvement_rate)
            else:
                weeks_to_target = 0

            metrics = AccuracyMetrics(
                gameweek=gameweek,
                timestamp=datetime.now(),
                overall_accuracy=overall_accuracy,
                position_accuracy=position_accuracy,
                captain_accuracy=captain_accuracy,
                mae=mae,
                rmse=rmse,
                r2=r2,
                differential_success_rate=diff_success,
                value_pick_accuracy=value_accuracy,
                premium_player_accuracy=premium_accuracy,
                ensemble_weight=self.ensemble_weights.copy(),
                feature_importance={},  # Will be populated by model
                target_progress=target_progress,
                weeks_to_target=weeks_to_target
            )

            self.accuracy_history.append(metrics)
            self._save_accuracy_metrics(metrics)

            return metrics

        except Exception as e:
            self.logger.error(f"Error evaluating accuracy: {e}")
            # Return default metrics
            return AccuracyMetrics(
                gameweek=gameweek,
                timestamp=datetime.now(),
                overall_accuracy=0.0,
                position_accuracy={},
                captain_accuracy=0.0,
                mae=float('inf'),
                rmse=float('inf'),
                r2=0.0,
                differential_success_rate=0.0,
                value_pick_accuracy=0.0,
                premium_player_accuracy=0.0,
                ensemble_weight={},
                feature_importance={},
                target_progress=0.0,
                weeks_to_target=None
            )

    def _save_accuracy_metrics(self, metrics: AccuracyMetrics):
        """Save accuracy metrics to file"""
        try:
            metrics_file = Path("data/accuracy_tracking/metrics_history.jsonl")
            metrics_file.parent.mkdir(parents=True, exist_ok=True)

            with open(metrics_file, 'a') as f:
                f.write(json.dumps(asdict(metrics), default=str) + '\n')

        except Exception as e:
            self.logger.error(f"Error saving accuracy metrics: {e}")

    def automatic_retrain(self, gameweek: int) -> Dict[str, Any]:
        """Automatic retraining after gameweek completion"""
        try:
            self.logger.info(f"Starting automatic retrain for gameweek {gameweek}")

            # Get latest training data
            X, y = self.data_prep.build_training_table()

            if X.empty or y.empty:
                raise ValueError("Insufficient training data")

            # Build advanced features
            X_enhanced = self.build_advanced_features(X)

            # Retrain ensemble
            model_data = self.build_ensemble_models(X_enhanced, y)

            # Get latest accuracy if possible
            latest_accuracy = self.accuracy_history[-1].overall_accuracy if self.accuracy_history else 0.0

            # Adjust strategies based on performance
            if latest_accuracy < self.target_timeline.get(gameweek, 0.65):
                self._adjust_aggressive_strategies(latest_accuracy, gameweek)

            result = {
                'status': 'success',
                'gameweek': gameweek,
                'models_trained': list(model_data['models'].keys()),
                'ensemble_weights': model_data['ensemble_weights'],
                'latest_accuracy': latest_accuracy,
                'target_accuracy': self.target_timeline.get(gameweek, 0.65),
                'retrain_timestamp': datetime.now().isoformat()
            }

            self.logger.info(f"Automatic retrain completed: {result}")
            return result

        except Exception as e:
            self.logger.error(f"Automatic retrain failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'gameweek': gameweek,
                'retrain_timestamp': datetime.now().isoformat()
            }

    def _adjust_aggressive_strategies(self, current_accuracy: float, gameweek: int):
        """Adjust strategies based on underperformance"""
        target = self.target_timeline.get(gameweek, 0.65)
        underperformance = target - current_accuracy

        if underperformance > self.config.underperformance_threshold:
            self.logger.info(f"Adjusting strategies due to underperformance: {underperformance:.3f}")

            # Increase risk tolerance
            self.config.risk_tolerance = min(0.9, self.config.risk_tolerance + 0.1)

            # Lower differential threshold (more aggressive picks)
            self.config.differential_threshold = max(0.1, self.config.differential_threshold - 0.02)

            # Increase captain confidence threshold
            self.config.captain_confidence_threshold = max(0.5, self.config.captain_confidence_threshold - 0.1)

            self.logger.info(f"Adjusted config: risk_tolerance={self.config.risk_tolerance}, "
                           f"differential_threshold={self.config.differential_threshold}, "
                           f"captain_threshold={self.config.captain_confidence_threshold}")

    def generate_maximum_accuracy_predictions(self) -> pd.DataFrame:
        """Generate predictions using maximum accuracy system"""
        try:
            self.logger.info("Generating maximum accuracy predictions")

            # Get base data
            from .ml_pipeline import MLPipeline
            pipeline = MLPipeline()
            base_data = pipeline.prepare_training_data()

            if base_data.empty:
                raise ValueError("No base data available")

            # Build advanced features
            features_df = self.build_advanced_features(base_data)

            # Prepare features for prediction
            feature_cols = [col for col in features_df.columns
                          if col not in ['player_id', 'name', 'position', 'team_id', 'price']]
            X = features_df[feature_cols].fillna(0)

            # Make ensemble predictions
            predictions, individual_preds = self.predict_with_ensemble(X)

            # Build results dataframe
            results_df = features_df[['player_id', 'name', 'position', 'team_id', 'price']].copy()
            results_df['predicted_points'] = predictions

            # Add individual model predictions for transparency
            for model_name, model_preds in individual_preds.items():
                results_df[f'pred_{model_name}'] = model_preds

            # Apply aggressive strategies
            results_df = self.apply_aggressive_strategies(results_df)

            # Add metadata
            results_df['prediction_timestamp'] = datetime.now().isoformat()
            results_df['system_version'] = 'maximum_accuracy_v1.0'
            results_df['ensemble_weights'] = [self.ensemble_weights] * len(results_df)

            # Sort by predicted points
            results_df = results_df.sort_values('predicted_points', ascending=False)

            self.logger.info(f"Generated maximum accuracy predictions for {len(results_df)} players")
            return results_df

        except Exception as e:
            self.logger.error(f"Error generating maximum accuracy predictions: {e}")
            raise

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        latest_accuracy = self.accuracy_history[-1] if self.accuracy_history else None

        return {
            'system_name': 'Maximum Accuracy FPL System',
            'target_accuracy': self.config.target_accuracy.value,
            'current_accuracy': latest_accuracy.overall_accuracy if latest_accuracy else 0.0,
            'target_timeline': self.target_timeline,
            'models_loaded': list(self.models.keys()) if self.models else [],
            'ensemble_weights': self.ensemble_weights,
            'config': asdict(self.config),
            'accuracy_history_count': len(self.accuracy_history),
            'last_retrain': latest_accuracy.timestamp.isoformat() if latest_accuracy else None
        }