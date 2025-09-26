from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

from .fpl_client import FPLClient
from .weekly_retrainer import WeeklyMLRetrainer


@dataclass
class BonusPointPrediction:
    """Prediction result for bonus points"""
    player_id: int
    player_name: str
    position: str
    team: str
    probability_3_bonus: float
    probability_2_bonus: float
    probability_1_bonus: float
    expected_bonus_points: float
    confidence: float
    key_factors: List[str]


class BonusPointPredictor:
    """
    Advanced bonus point prediction system for FPL players.
    Uses BPS (Bonus Point System) data and historical patterns to predict
    which players are most likely to earn bonus points.
    """

    def __init__(self):
        self.fpl = FPLClient()
        self.retrainer = WeeklyMLRetrainer()
        self.logger = logging.getLogger(__name__)

        # Create model directories
        Path("models/bonus").mkdir(parents=True, exist_ok=True)

        # Model parameters
        self.models = {}
        self.feature_names = []
        self.is_trained = False

        # BPS thresholds for different positions (typical values)
        self.bps_thresholds = {
            'GKP': {'3_bonus': 30, '2_bonus': 25, '1_bonus': 20},
            'DEF': {'3_bonus': 35, '2_bonus': 28, '1_bonus': 24},
            'MID': {'3_bonus': 40, '2_bonus': 32, '1_bonus': 28},
            'FWD': {'3_bonus': 35, '2_bonus': 28, '1_bonus': 25}
        }

        # Historical bonus patterns (players who commonly get bonus)
        self.bonus_patterns = {
            'high_bps_players': [],  # Players who consistently score high BPS
            'position_tendencies': {},  # Bonus frequency by position
            'price_correlations': {}   # Bonus frequency by price range
        }

    def train_models(self, historical_gameweeks: List[int]) -> Dict[str, float]:
        """Train bonus point prediction models using historical data"""

        self.logger.info("Training bonus point prediction models...")

        # Collect training data
        training_data = self._collect_bonus_training_data(historical_gameweeks)
        if training_data.empty:
            raise ValueError("No training data available for bonus point prediction")

        # Prepare features
        X, y_3bonus, y_2bonus, y_1bonus = self._prepare_bonus_features(training_data)
        self.feature_names = list(X.columns)

        # Train separate models for 3, 2, and 1 bonus points
        results = {}

        # Model 1: 3 Bonus Points (most important)
        self.models['3_bonus'] = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            scale_pos_weight=len(y_3bonus[y_3bonus == 0]) / len(y_3bonus[y_3bonus == 1])  # Handle class imbalance
        )

        X_train, X_test, y_train, y_test = train_test_split(X, y_3bonus, test_size=0.2, random_state=42)
        self.models['3_bonus'].fit(X_train, y_train)
        y_pred = self.models['3_bonus'].predict(X_test)
        results['3_bonus_accuracy'] = accuracy_score(y_test, y_pred)

        # Model 2: 2 Bonus Points
        self.models['2_bonus'] = XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            scale_pos_weight=len(y_2bonus[y_2bonus == 0]) / len(y_2bonus[y_2bonus == 1])
        )

        X_train, X_test, y_train, y_test = train_test_split(X, y_2bonus, test_size=0.2, random_state=42)
        self.models['2_bonus'].fit(X_train, y_train)
        y_pred = self.models['2_bonus'].predict(X_test)
        results['2_bonus_accuracy'] = accuracy_score(y_test, y_pred)

        # Model 3: Any Bonus Points (1, 2, or 3)
        y_any_bonus = ((y_1bonus == 1) | (y_2bonus == 1) | (y_3bonus == 1)).astype(int)

        self.models['any_bonus'] = XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            scale_pos_weight=len(y_any_bonus[y_any_bonus == 0]) / len(y_any_bonus[y_any_bonus == 1])
        )

        X_train, X_test, y_train, y_test = train_test_split(X, y_any_bonus, test_size=0.2, random_state=42)
        self.models['any_bonus'].fit(X_train, y_train)
        y_pred = self.models['any_bonus'].predict(X_test)
        results['any_bonus_accuracy'] = accuracy_score(y_test, y_pred)

        # Save models
        for model_name, model in self.models.items():
            joblib.dump(model, f"models/bonus/{model_name}_model.pkl")

        # Save feature names and training info
        joblib.dump(self.feature_names, "models/bonus/feature_names.pkl")
        joblib.dump(results, "models/bonus/training_results.pkl")

        self.is_trained = True
        self.logger.info(f"Bonus models trained. Accuracies: {results}")

        return results

    def _collect_bonus_training_data(self, gameweeks: List[int]) -> pd.DataFrame:
        """Collect historical data for training bonus point models"""

        training_records = []

        for gw in gameweeks:
            try:
                # Get actual results with bonus data
                gw_data = self.retrainer.collect_gameweek_results(gw)
                if gw_data is None or gw_data.empty:
                    continue

                # Get detailed stats for each player using FPL client
                for _, player_row in gw_data.iterrows():
                    player_id = player_row['player_id']

                    # Get detailed gameweek stats
                    detailed_stats = self.fpl.get_player_gameweek_stats(player_id, gw)
                    if not detailed_stats:
                        continue

                    # Create training record
                    record = {
                        'player_id': player_id,
                        'gameweek': gw,
                        'points': detailed_stats['points'],
                        'minutes': detailed_stats['minutes'],
                        'goals': detailed_stats['goals'],
                        'assists': detailed_stats['assists'],
                        'clean_sheets': detailed_stats['clean_sheets'],
                        'saves': detailed_stats['saves'],
                        'bonus_actual': detailed_stats['bonus'],
                        'bps': detailed_stats['bps'],
                        'influence': detailed_stats['influence'],
                        'creativity': detailed_stats['creativity'],
                        'threat': detailed_stats['threat'],
                        'ict_index': detailed_stats['ict_index'],
                        'selected': detailed_stats['selected'],
                        'transfers_in': detailed_stats['transfers_in'],
                    }

                    # Add player static data
                    bootstrap = self.fpl.bootstrap_static()
                    elements = bootstrap.get('elements', [])
                    player_info = next((p for p in elements if p['id'] == player_id), None)

                    if player_info:
                        record.update({
                            'position_id': player_info.get('element_type', 0),
                            'team_id': player_info.get('team', 0),
                            'price': player_info.get('now_cost', 0) / 10,
                            'form': player_info.get('form', 0),
                            'total_points_season': player_info.get('total_points', 0),
                        })

                    training_records.append(record)

            except Exception as e:
                self.logger.error(f"Error collecting bonus data for GW {gw}: {e}")
                continue

        return pd.DataFrame(training_records)

    def _prepare_bonus_features(self, training_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Prepare features for bonus point prediction models"""

        # Create target variables
        y_3bonus = (training_data['bonus_actual'] == 3).astype(int)
        y_2bonus = (training_data['bonus_actual'] == 2).astype(int)
        y_1bonus = (training_data['bonus_actual'] == 1).astype(int)

        # Feature engineering
        features = training_data.copy()

        # Basic stats features
        features['goals_per_minute'] = features['goals'] / np.maximum(features['minutes'], 1)
        features['assists_per_minute'] = features['assists'] / np.maximum(features['minutes'], 1)
        features['saves_per_minute'] = features['saves'] / np.maximum(features['minutes'], 1)

        # BPS-based features (most important for bonus)
        features['bps_per_minute'] = features['bps'] / np.maximum(features['minutes'], 1)
        features['high_bps'] = (features['bps'] >= 30).astype(int)
        features['very_high_bps'] = (features['bps'] >= 40).astype(int)

        # ICT Index features
        features['ict_per_minute'] = features['ict_index'] / np.maximum(features['minutes'], 1)
        features['high_influence'] = (features['influence'] >= 50).astype(int)
        features['high_creativity'] = (features['creativity'] >= 50).astype(int)
        features['high_threat'] = (features['threat'] >= 50).astype(int)

        # Position-specific features
        features['is_gkp'] = (features['position_id'] == 1).astype(int)
        features['is_def'] = (features['position_id'] == 2).astype(int)
        features['is_mid'] = (features['position_id'] == 3).astype(int)
        features['is_fwd'] = (features['position_id'] == 4).astype(int)

        # Price-based features (expensive players get bonus more often)
        features['is_premium'] = (features['price'] >= 8.0).astype(int)
        features['is_budget'] = (features['price'] <= 5.0).astype(int)

        # Form features
        features['good_form'] = (features['form'] >= 4.0).astype(int)
        features['excellent_form'] = (features['form'] >= 6.0).astype(int)

        # Playing time features
        features['started_match'] = (features['minutes'] >= 60).astype(int)
        features['full_match'] = (features['minutes'] >= 90).astype(int)

        # Attacking contribution
        features['attacking_returns'] = features['goals'] + features['assists']
        features['multiple_returns'] = (features['attacking_returns'] >= 2).astype(int)

        # Clean sheet bonus (for GKP/DEF)
        features['clean_sheet_bonus'] = ((features['clean_sheets'] == 1) & (features['is_def'] | features['is_gkp'])).astype(int)

        # Select features for model
        feature_columns = [
            'minutes', 'goals', 'assists', 'clean_sheets', 'saves',
            'bps', 'influence', 'creativity', 'threat', 'ict_index',
            'price', 'form', 'goals_per_minute', 'assists_per_minute', 'saves_per_minute',
            'bps_per_minute', 'high_bps', 'very_high_bps', 'ict_per_minute',
            'high_influence', 'high_creativity', 'high_threat',
            'is_gkp', 'is_def', 'is_mid', 'is_fwd',
            'is_premium', 'is_budget', 'good_form', 'excellent_form',
            'started_match', 'full_match', 'attacking_returns', 'multiple_returns', 'clean_sheet_bonus'
        ]

        # Ensure all features exist
        for col in feature_columns:
            if col not in features.columns:
                features[col] = 0

        X = features[feature_columns].fillna(0)

        return X, y_3bonus, y_2bonus, y_1bonus

    def predict_bonus_points(self, player_data: pd.DataFrame) -> List[BonusPointPrediction]:
        """Predict bonus points for current gameweek players"""

        if not self.is_trained:
            self._load_models()

        if not self.is_trained:
            raise ValueError("Bonus models not trained. Call train_models() first.")

        predictions = []

        # Prepare features for current players
        X = self._prepare_current_player_features(player_data)

        for idx, (_, player_row) in enumerate(player_data.iterrows()):
            try:
                player_features = X.iloc[idx:idx+1]

                # Get probabilities for each bonus level
                prob_3bonus = self.models['3_bonus'].predict_proba(player_features)[0][1]
                prob_2bonus = self.models['2_bonus'].predict_proba(player_features)[0][1]
                prob_any_bonus = self.models['any_bonus'].predict_proba(player_features)[0][1]

                # Estimate 1-bonus probability (any_bonus - 2bonus - 3bonus)
                prob_1bonus = max(0, prob_any_bonus - prob_2bonus - prob_3bonus)

                # Calculate expected bonus points
                expected_bonus = (3 * prob_3bonus) + (2 * prob_2bonus) + (1 * prob_1bonus)

                # Calculate confidence (based on model certainty)
                confidence = max(prob_3bonus, prob_2bonus, prob_1bonus, (1 - prob_any_bonus))

                # Get key factors for this prediction
                key_factors = self._get_key_factors(player_features, player_row)

                prediction = BonusPointPrediction(
                    player_id=int(player_row.get('id', 0)),
                    player_name=str(player_row.get('web_name', '')),
                    position=str(player_row.get('position', '')),
                    team=str(player_row.get('team_name', '')),
                    probability_3_bonus=float(prob_3bonus),
                    probability_2_bonus=float(prob_2bonus),
                    probability_1_bonus=float(prob_1bonus),
                    expected_bonus_points=float(expected_bonus),
                    confidence=float(confidence),
                    key_factors=key_factors
                )

                predictions.append(prediction)

            except Exception as e:
                self.logger.error(f"Error predicting bonus for player {player_row.get('id', 'unknown')}: {e}")
                continue

        return predictions

    def _prepare_current_player_features(self, player_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for current players (similar to training features)"""

        features = player_data.copy()

        # Add derived features (same as training)
        features['is_gkp'] = (features.get('element_type', 0) == 1).astype(int)
        features['is_def'] = (features.get('element_type', 0) == 2).astype(int)
        features['is_mid'] = (features.get('element_type', 0) == 3).astype(int)
        features['is_fwd'] = (features.get('element_type', 0) == 4).astype(int)

        # Price features
        price = features.get('now_cost', 0) / 10
        features['price'] = price
        features['is_premium'] = (price >= 8.0).astype(int)
        features['is_budget'] = (price <= 5.0).astype(int)

        # Form features
        form = features.get('form', 0).astype(float)
        features['form'] = form
        features['good_form'] = (form >= 4.0).astype(int)
        features['excellent_form'] = (form >= 6.0).astype(int)

        # Use historical averages for per-minute stats (since we're predicting)
        features['minutes'] = 70  # Assume decent playing time
        features['goals'] = features.get('goals_scored', 0)
        features['assists'] = features.get('assists', 0)
        features['clean_sheets'] = 0  # Will depend on match
        features['saves'] = 0  # Will depend on match for GKPs

        # Estimate BPS and ICT based on season performance
        features['bps'] = features.get('bps', 20)  # Season average BPS
        features['influence'] = features.get('influence', 30)
        features['creativity'] = features.get('creativity', 30)
        features['threat'] = features.get('threat', 30)
        features['ict_index'] = features.get('ict_index', 90)

        # Calculate derived features
        features['goals_per_minute'] = features['goals'] / 70
        features['assists_per_minute'] = features['assists'] / 70
        features['saves_per_minute'] = features['saves'] / 70
        features['bps_per_minute'] = features['bps'] / 70
        features['ict_per_minute'] = features['ict_index'] / 70

        features['high_bps'] = (features['bps'] >= 30).astype(int)
        features['very_high_bps'] = (features['bps'] >= 40).astype(int)
        features['high_influence'] = (features['influence'] >= 50).astype(int)
        features['high_creativity'] = (features['creativity'] >= 50).astype(int)
        features['high_threat'] = (features['threat'] >= 50).astype(int)

        features['started_match'] = 1  # Assume they start
        features['full_match'] = 1
        features['attacking_returns'] = features['goals'] + features['assists']
        features['multiple_returns'] = (features['attacking_returns'] >= 2).astype(int)
        features['clean_sheet_bonus'] = 0  # Unknown until after match

        # Ensure all required features exist and are in correct order
        return features[self.feature_names].fillna(0)

    def _get_key_factors(self, player_features: pd.DataFrame, player_row: pd.Series) -> List[str]:
        """Identify key factors contributing to bonus prediction"""

        factors = []

        # Get feature values
        features_dict = player_features.iloc[0].to_dict()

        # Check important factors
        if features_dict.get('is_premium', 0) == 1:
            factors.append("Premium player")
        if features_dict.get('excellent_form', 0) == 1:
            factors.append("Excellent recent form")
        if features_dict.get('high_bps', 0) == 1:
            factors.append("High BPS tendency")
        if features_dict.get('multiple_returns', 0) == 1:
            factors.append("Multiple attacking returns")

        position = 'Unknown'
        if features_dict.get('is_gkp', 0) == 1:
            position = 'Goalkeeper'
            if features_dict.get('saves', 0) > 3:
                factors.append("High save potential")
        elif features_dict.get('is_def', 0) == 1:
            position = 'Defender'
            factors.append("Clean sheet potential")
        elif features_dict.get('is_mid', 0) == 1:
            position = 'Midfielder'
            factors.append("All-round contributions")
        elif features_dict.get('is_fwd', 0) == 1:
            position = 'Forward'
            factors.append("Goal threat")

        if not factors:
            factors.append(f"{position} with solid fundamentals")

        return factors[:3]  # Return top 3 factors

    def _load_models(self) -> None:
        """Load pre-trained bonus point models"""

        try:
            model_files = {
                '3_bonus': 'models/bonus/3_bonus_model.pkl',
                '2_bonus': 'models/bonus/2_bonus_model.pkl',
                'any_bonus': 'models/bonus/any_bonus_model.pkl'
            }

            for model_name, model_file in model_files.items():
                if Path(model_file).exists():
                    self.models[model_name] = joblib.load(model_file)

            # Load feature names
            if Path("models/bonus/feature_names.pkl").exists():
                self.feature_names = joblib.load("models/bonus/feature_names.pkl")

            if self.models and self.feature_names:
                self.is_trained = True
                self.logger.info("Bonus point models loaded successfully")
            else:
                self.logger.warning("Could not load bonus point models")

        except Exception as e:
            self.logger.error(f"Error loading bonus models: {e}")

    def get_top_bonus_candidates(self, predictions: List[BonusPointPrediction], n: int = 10) -> List[BonusPointPrediction]:
        """Get top N players most likely to earn bonus points"""

        # Sort by expected bonus points (3-bonus weighted heavily)
        sorted_predictions = sorted(
            predictions,
            key=lambda p: p.probability_3_bonus * 3 + p.probability_2_bonus * 2 + p.probability_1_bonus,
            reverse=True
        )

        return sorted_predictions[:n]

    def evaluate_model_performance(self, test_gameweeks: List[int]) -> Dict[str, float]:
        """Evaluate bonus point model performance on test data"""

        if not self.is_trained:
            raise ValueError("Models not trained")

        test_data = self._collect_bonus_training_data(test_gameweeks)
        if test_data.empty:
            return {}

        X, y_3bonus, y_2bonus, y_1bonus = self._prepare_bonus_features(test_data)

        results = {}

        # Evaluate each model
        for bonus_type, y_true in [('3_bonus', y_3bonus), ('2_bonus', y_2bonus)]:
            if bonus_type in self.models:
                y_pred = self.models[bonus_type].predict(X)
                results[f'{bonus_type}_accuracy'] = accuracy_score(y_true, y_pred)

                # Calculate precision/recall for positive class
                precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
                results[f'{bonus_type}_precision'] = precision
                results[f'{bonus_type}_recall'] = recall

        return results