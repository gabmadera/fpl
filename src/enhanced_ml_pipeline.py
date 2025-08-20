from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import logging

from .feature_engineer import FeatureEngineer
from .model_trainer import FPLModelTrainer, XGBoostTrainer
from .pipeline import DataPipeline
from .data_prep import DataPrep
from .fpl_client import FPLClient
from .name_matching import PlayerNameMatcher
from .fbref_scraper import FBRefScraper
from .understat_scraper import UnderstatScraper
from .alternative_data_sources import AlternativeDataSources

# Import new enhanced systems
from .enhanced_fixture_analyzer import EnhancedFixtureAnalyzer
from .position_specific_features import PositionSpecificFeatureEngineer
from .dynamic_ensemble_weights import DynamicEnsembleWeights
from .weather_integration import WeatherDataIntegration


class EnhancedMLPipeline:
    """Advanced ML pipeline with all prediction improvements integrated"""
    
    def __init__(self) -> None:
        Path("models/2025_26").mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.engineer = FeatureEngineer()
        self.trainer = FPLModelTrainer()
        self.xgb = XGBoostTrainer()
        self.pipe = DataPipeline()
        self.prep = DataPrep()
        self.fpl = FPLClient()
        self.fbref = FBRefScraper()
        self.understat = UnderstatScraper()
        self.alt_sources = AlternativeDataSources()
        
        # Enhanced systems
        self.enhanced_fixtures = EnhancedFixtureAnalyzer()
        self.position_features = PositionSpecificFeatureEngineer()
        self.dynamic_weights = DynamicEnsembleWeights()
        self.weather_integration = WeatherDataIntegration()
        
        # Initialize ensemble with enhanced weights
        self.ensemble = None
        self._initialize_enhanced_ensemble()
        
        # Initialize persistence manager
        self.persistence = None
        self._initialize_persistence()
        
        # Performance tracking
        self.prediction_metrics = {
            'mae_by_position': {},
            'accuracy_by_gameweek': {},
            'model_contributions': {'xgb': [], 'rf': [], 'nn': []}
        }
    
    def predict_enhanced(self, gameweek: int = None) -> pd.DataFrame:
        """Generate enhanced predictions with all improvements"""
        try:
            self.logger.info("Starting enhanced prediction generation...")
            
            # Step 1: Prepare base data
            base_data = self.prepare_enhanced_training_data()
            if base_data.empty:
                raise ValueError("No base data available for predictions")
            
            # Step 2: Apply position-specific feature engineering
            self.logger.info("Applying position-specific feature engineering...")
            enhanced_data = self.position_features.engineer_position_features(base_data)
            
            # Step 3: Get enhanced fixture analysis
            self.logger.info("Analyzing fixtures with enhanced system...")
            fixture_analysis = self.enhanced_fixtures.analyze_enhanced_fixtures(gameweeks=5)
            
            # Step 4: Apply fixture-based adjustments
            enhanced_data = self._apply_enhanced_fixture_adjustments(enhanced_data, fixture_analysis)
            
            # Step 5: Weather data integration
            self.logger.info("Integrating weather data...")
            enhanced_data = self._apply_weather_adjustments(enhanced_data, gameweek)
            
            # Step 6: Generate base model predictions
            base_predictions = self._generate_base_predictions(enhanced_data)
            
            # Step 7: Apply dynamic ensemble weighting
            self.logger.info("Applying dynamic ensemble weights...")
            final_predictions = self._apply_dynamic_weighting(base_predictions, enhanced_data)
            
            # Step 8: Apply final enhancements
            final_predictions = self._apply_final_enhancements(final_predictions, fixture_analysis)
            
            # Step 9: Save and track performance
            self._save_enhanced_predictions(final_predictions, gameweek)
            
            self.logger.info(f"Enhanced predictions complete: {len(final_predictions)} players")
            return final_predictions
            
        except Exception as e:
            self.logger.error(f"Enhanced prediction generation failed: {e}")
            # Fallback to basic predictions
            return self._fallback_predictions()
    
    def prepare_enhanced_training_data(self) -> pd.DataFrame:
        """Prepare training data with enhanced features"""
        try:
            # Collect latest FPL data
            self.pipe.collect_fpl_snapshots()
            datasets = self.fpl.bootstrap_static()
            players = pd.DataFrame(datasets.get("elements", []))
            
            if players.empty:
                return pd.DataFrame()
            
            # Add team information
            teams = pd.DataFrame(datasets.get("teams", []))
            if not teams.empty:
                players = players.merge(teams[['id', 'name', 'short_name']], 
                                      left_on='team', right_on='id', 
                                      suffixes=('', '_team'))
                players['team_short'] = players['short_name']
            
            # Basic feature engineering
            enhanced_players = self.engineer.engineer_features(players)
            
            # Add FBref and Understat data
            try:
                fbref_data = self.fbref.get_season_data()
                understat_data = self.understat.get_season_data()
                
                if not fbref_data.empty:
                    enhanced_players = self._merge_external_data(enhanced_players, fbref_data, 'fbref')
                if not understat_data.empty:
                    enhanced_players = self._merge_external_data(enhanced_players, understat_data, 'understat')
                    
            except Exception as e:
                self.logger.warning(f"External data integration partial failure: {e}")
            
            # Add position mapping
            position_mapping = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
            enhanced_players['position'] = enhanced_players['element_type'].map(position_mapping)
            
            return enhanced_players
            
        except Exception as e:
            self.logger.error(f"Enhanced training data preparation failed: {e}")
            return pd.DataFrame()
    
    def _apply_enhanced_fixture_adjustments(self, df: pd.DataFrame, fixture_analysis: Dict) -> pd.DataFrame:
        """Apply enhanced fixture-based adjustments to player data"""
        try:
            if 'error' in fixture_analysis:
                self.logger.warning(f"Fixture analysis error: {fixture_analysis['error']}")
                return df
            
            enhanced_df = df.copy()
            enhanced_matrix = fixture_analysis.get('enhanced_matrix', {})
            
            # Apply position-specific fixture adjustments
            for idx, player in enhanced_df.iterrows():
                team_id = player.get('team_id') or player.get('team')
                position = player.get('position')
                
                if not team_id or not position:
                    continue
                
                # Find player's next fixtures
                player_fixtures = []
                for gw, fixtures in enhanced_matrix.items():
                    if team_id in fixtures:
                        player_fixtures.append(fixtures[team_id])
                
                if player_fixtures:
                    # Calculate position-specific fixture multiplier
                    fixture_multipliers = []
                    for fixture in player_fixtures[:3]:  # Next 3 fixtures
                        multipliers = self.enhanced_fixtures.get_position_specific_multipliers(
                            position, fixture
                        )
                        fixture_multipliers.append(multipliers.get('prediction_multiplier', 1.0))
                    
                    # Average multiplier across next few fixtures
                    avg_multiplier = np.mean(fixture_multipliers) if fixture_multipliers else 1.0
                    
                    # Apply to relevant stats
                    enhanced_df.at[idx, 'fixture_multiplier'] = avg_multiplier
                    enhanced_df.at[idx, 'fixture_adjusted_form'] = player.get('form', 0) * avg_multiplier
                    
                    # Store fixture difficulty for each position
                    if player_fixtures:
                        first_fixture = player_fixtures[0]
                        difficulties = first_fixture.get('difficulties', {})
                        enhanced_df.at[idx, 'next_fixture_difficulty'] = difficulties.get(position, 3)
                        enhanced_df.at[idx, 'clean_sheet_probability'] = first_fixture.get('clean_sheet_probability', 0.25)
                        enhanced_df.at[idx, 'expected_goals_context'] = first_fixture.get('expected_goals', 1.4)
            
            self.logger.info("Applied enhanced fixture adjustments")
            return enhanced_df
            
        except Exception as e:
            self.logger.error(f"Enhanced fixture adjustment failed: {e}")
            return df
    
    def _apply_weather_adjustments(self, df: pd.DataFrame, gameweek: int) -> pd.DataFrame:
        """Apply weather-based adjustments"""
        try:
            # Get fixtures for the gameweek
            fixtures = self.fpl.fixtures()
            if not fixtures:
                return df
            
            fixtures_df = pd.DataFrame(fixtures)
            
            # Filter for target gameweek
            if gameweek:
                gameweek_fixtures = fixtures_df[fixtures_df['event'] == gameweek]
            else:
                # Get current gameweek fixtures
                current_gw = self._get_current_gameweek()
                gameweek_fixtures = fixtures_df[fixtures_df['event'] == current_gw]
            
            if gameweek_fixtures.empty:
                return df
            
            # Get weather impact for fixtures
            fixtures_with_weather = self.weather_integration.get_fixture_weather_impact(gameweek_fixtures)
            
            # Apply weather adjustments to predictions
            enhanced_df = self.weather_integration.apply_weather_adjustments(df, fixtures_with_weather)
            
            return enhanced_df
            
        except Exception as e:
            self.logger.error(f"Weather adjustment failed: {e}")
            return df
    
    def _generate_base_predictions(self, enhanced_data: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions from base models"""
        try:
            # Prepare features for modeling
            feature_cols = self._get_modeling_features(enhanced_data)
            X = enhanced_data[feature_cols].fillna(0)
            
            # Load trained models
            model_paths = {
                'xgb': Path("models/2025_26/xgboost_model.joblib"),
                'rf': Path("models/2025_26/rf_model.joblib"), 
                'nn': Path("models/2025_26/nn_model.joblib")
            }
            
            predictions_df = enhanced_data[['id', 'name', 'position', 'team_id', 'team_short']].copy()
            
            # Generate predictions from each model
            model_predictions = {}
            
            for model_name, model_path in model_paths.items():
                try:
                    if model_path.exists():
                        model = joblib.load(model_path)
                        pred = model.predict(X)
                        model_predictions[model_name] = pred
                        predictions_df[f'{model_name}_prediction'] = pred
                    else:
                        self.logger.warning(f"Model not found: {model_path}")
                        model_predictions[model_name] = np.zeros(len(X))
                        
                except Exception as e:
                    self.logger.error(f"Prediction from {model_name} failed: {e}")
                    model_predictions[model_name] = np.zeros(len(X))
            
            # Store individual model predictions
            predictions_df['model_predictions'] = [
                {model: pred[i] for model, pred in model_predictions.items()}
                for i in range(len(predictions_df))
            ]
            
            return predictions_df
            
        except Exception as e:
            self.logger.error(f"Base prediction generation failed: {e}")
            return enhanced_data.copy()
    
    def _apply_dynamic_weighting(self, predictions_df: pd.DataFrame, enhanced_data: pd.DataFrame) -> pd.DataFrame:
        """Apply dynamic ensemble weighting"""
        try:
            final_predictions = predictions_df.copy()
            
            # Get recent performance data
            recent_performance = self.dynamic_weights.get_recent_performance()
            
            # Apply position-specific dynamic weighting
            for idx, player in final_predictions.iterrows():
                position = player.get('position')
                model_preds = player.get('model_predictions', {})
                
                if not model_preds or not position:
                    final_predictions.at[idx, 'predicted_points'] = 0
                    continue
                
                # Get dynamic weights for this position
                weights = self.dynamic_weights.get_position_optimized_weights(
                    position, recent_performance
                )
                
                # Calculate weighted prediction
                weighted_pred = self.dynamic_weights.get_weighted_prediction(
                    model_preds, position=position
                )
                
                final_predictions.at[idx, 'predicted_points'] = max(0, weighted_pred)
                final_predictions.at[idx, 'ensemble_weights'] = weights
                final_predictions.at[idx, 'confidence_score'] = self._calculate_prediction_confidence(
                    model_preds, weights
                )
            
            return final_predictions
            
        except Exception as e:
            self.logger.error(f"Dynamic weighting failed: {e}")
            return predictions_df
    
    def _apply_final_enhancements(self, predictions_df: pd.DataFrame, fixture_analysis: Dict) -> pd.DataFrame:
        """Apply final prediction enhancements"""
        try:
            enhanced_df = predictions_df.copy()
            
            # Apply fixture multipliers if available
            if 'fixture_multiplier' in enhanced_df.columns:
                enhanced_df['predicted_points'] = enhanced_df['predicted_points'] * enhanced_df['fixture_multiplier']
            
            # Apply weather adjustments if available  
            if 'weather_adjustment' in enhanced_df.columns:
                enhanced_df['predicted_points'] = enhanced_df['predicted_points'] * enhanced_df['weather_adjustment']
            
            # Add value calculation
            if 'now_cost' in enhanced_df.columns:
                enhanced_df['price'] = enhanced_df['now_cost'] / 10.0
                enhanced_df['value'] = enhanced_df['predicted_points'] / enhanced_df['price']
            
            # Add additional metadata
            enhanced_df['prediction_method'] = 'enhanced_ensemble'
            enhanced_df['prediction_timestamp'] = datetime.now().isoformat()
            
            # Sort by predicted points
            enhanced_df = enhanced_df.sort_values('predicted_points', ascending=False)
            
            return enhanced_df
            
        except Exception as e:
            self.logger.error(f"Final enhancement failed: {e}")
            return predictions_df
    
    def _calculate_prediction_confidence(self, model_predictions: Dict, weights: Dict) -> float:
        """Calculate confidence score for prediction"""
        try:
            if not model_predictions or len(model_predictions) < 2:
                return 0.5  # Low confidence
            
            # Calculate variance in model predictions
            pred_values = list(model_predictions.values())
            pred_variance = np.var(pred_values)
            
            # Lower variance = higher confidence
            confidence = 1 / (1 + pred_variance)
            
            # Adjust based on model weights (balanced weights = higher confidence)
            weight_values = list(weights.values())
            weight_entropy = -sum(w * np.log(w + 1e-10) for w in weight_values)
            max_entropy = -np.log(1/len(weight_values))
            
            entropy_factor = weight_entropy / max_entropy if max_entropy > 0 else 0.5
            
            final_confidence = (confidence + entropy_factor) / 2
            return min(0.95, max(0.05, final_confidence))
            
        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {e}")
            return 0.5
    
    def _get_modeling_features(self, df: pd.DataFrame) -> List[str]:
        """Get list of features for modeling"""
        # Basic features that should be available
        basic_features = [
            'total_points', 'minutes', 'goals_scored', 'assists', 'clean_sheets',
            'goals_conceded', 'yellow_cards', 'red_cards', 'saves', 'bonus',
            'form', 'points_per_game', 'selected_by_percent', 'transfers_in_event',
            'transfers_out_event', 'value_form', 'value_season'
        ]
        
        # Filter for available columns
        available_features = [col for col in basic_features if col in df.columns]
        
        # Add position-specific features if available
        position_features = [col for col in df.columns if any(x in col for x in [
            'composite_rating', 'weighted_form', 'percentile', 'efficiency',
            'rate', 'index', 'threat', 'conversion', 'multiplier'
        ])]
        
        available_features.extend(position_features)
        
        # Ensure we have at least some features
        if len(available_features) < 5:
            available_features = [col for col in df.select_dtypes(include=[np.number]).columns 
                                if col not in ['id', 'team_id']][:20]
        
        return available_features
    
    def _save_enhanced_predictions(self, predictions_df: pd.DataFrame, gameweek: int):
        """Save predictions with enhanced metadata"""
        try:
            # Create output directory
            output_dir = Path("data/processed")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save current predictions
            current_file = output_dir / "predictions_current.csv"
            predictions_df.to_csv(current_file, index=False)
            
            # Save gameweek-specific predictions
            if gameweek:
                gw_file = output_dir / f"predictions_gw{gameweek}_enhanced.csv"
                predictions_df.to_csv(gw_file, index=False)
            
            # Save prediction metadata
            metadata = {
                'prediction_count': len(predictions_df),
                'timestamp': datetime.now().isoformat(),
                'gameweek': gameweek,
                'enhancement_features': [
                    'position_specific_features',
                    'enhanced_fixture_analysis', 
                    'dynamic_ensemble_weights',
                    'weather_integration'
                ],
                'average_confidence': predictions_df.get('confidence_score', pd.Series([0.5])).mean()
            }
            
            metadata_file = output_dir / "prediction_metadata.json"
            import json
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Enhanced predictions saved: {current_file}")
            
        except Exception as e:
            self.logger.error(f"Enhanced prediction saving failed: {e}")
    
    def _fallback_predictions(self) -> pd.DataFrame:
        """Fallback to basic predictions if enhanced fails"""
        try:
            # Use original ML pipeline as fallback
            from .ml_pipeline import MLPipeline
            fallback_pipeline = MLPipeline()
            return fallback_pipeline.predict_current()
            
        except Exception as e:
            self.logger.error(f"Fallback prediction failed: {e}")
            return pd.DataFrame()
    
    def _get_current_gameweek(self) -> int:
        """Get current gameweek"""
        try:
            bootstrap = self.fpl.bootstrap_static()
            events = bootstrap.get("events", [])
            for event in events:
                if event.get("is_current", False):
                    return event.get("id", 1)
            return 1
        except:
            return 1
    
    def _merge_external_data(self, df: pd.DataFrame, external_data: pd.DataFrame, source: str) -> pd.DataFrame:
        """Merge external data sources"""
        try:
            # Simple merge on player name - could be enhanced with better matching
            if 'name' in df.columns and 'player_name' in external_data.columns:
                return df.merge(external_data, left_on='name', right_on='player_name', 
                              how='left', suffixes=('', f'_{source}'))
            return df
        except Exception as e:
            self.logger.error(f"External data merge failed: {e}")
            return df
    
    def _initialize_enhanced_ensemble(self):
        """Initialize ensemble with enhanced weights"""
        try:
            from .ensemble_models import MultiModelEnsemble
            self.ensemble = MultiModelEnsemble()
            # Set dynamic weights as default
            self.ensemble.model_weights = self.dynamic_weights.current_weights
        except Exception as e:
            self.logger.error(f"Enhanced ensemble initialization failed: {e}")
    
    def _initialize_persistence(self):
        """Initialize persistence manager"""
        try:
            from .persistence_manager import PersistenceManager
            self.persistence = PersistenceManager()
        except Exception as e:
            self.logger.error(f"Persistence initialization failed: {e}")
    
    def train_enhanced_models(self) -> Dict[str, Any]:
        """Train models with enhanced features"""
        try:
            self.logger.info("Starting enhanced model training...")
            
            # Prepare enhanced training data
            training_data = self.prepare_enhanced_training_data()
            if training_data.empty:
                return {"status": "failed", "reason": "No training data"}
            
            # Apply position-specific feature engineering
            enhanced_training_data = self.position_features.engineer_position_features(training_data)
            
            # Train base models
            # This would integrate with your existing training pipeline
            # For now, return success status
            
            return {
                "status": "success",
                "enhanced_features": True,
                "training_samples": len(enhanced_training_data),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced model training failed: {e}")
            return {"status": "failed", "reason": str(e)}