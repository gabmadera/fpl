from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
from pathlib import Path
import joblib
from dataclasses import dataclass
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from .fpl_client import FPLClient


@dataclass
class PlayerSimilarity:
    """Result from player similarity calculation"""
    player_id: int
    player_name: str
    similarity_score: float
    position: str
    team: str
    price: float
    key_similarities: List[str]


class PlayerEmbeddingSystem:
    """
    Player embedding system for similarity-based predictions and analysis.
    Useful for new players, transfers, and understanding player archetypes.
    """

    def __init__(self, embedding_dim: int = 32):
        self.fpl = FPLClient()
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.embedding_dim = embedding_dim

        # Model components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=embedding_dim)
        self.embedding_matrix = None
        self.player_embeddings = {}
        self.feature_names = []

        # Player metadata
        self.player_metadata = {}
        self.position_clusters = {}
        self.similarity_cache = {}

        # Create directories
        Path("models/embeddings").mkdir(parents=True, exist_ok=True)
        Path("data/embeddings").mkdir(parents=True, exist_ok=True)

        # State
        self.is_fitted = False

    def create_player_embeddings(self, player_data: pd.DataFrame,
                                force_rebuild: bool = False) -> Dict[int, np.ndarray]:
        """Create embeddings for all players based on their characteristics"""
        try:
            self.logger.info("Creating player embeddings...")

            # Check if embeddings already exist
            if not force_rebuild and self._load_embeddings():
                self.logger.info("Loaded existing embeddings")
                return self.player_embeddings

            # Prepare features for embedding
            features_df = self._prepare_embedding_features(player_data)
            if features_df.empty:
                raise ValueError("No features available for embeddings")

            # Scale features
            scaled_features = self.scaler.fit_transform(features_df)

            # Create embeddings using PCA
            embeddings = self.pca.fit_transform(scaled_features)

            # Store embeddings
            self.player_embeddings = {}
            for idx, player_row in player_data.iterrows():
                player_id = int(player_row.get('id', 0))
                if player_id > 0:
                    self.player_embeddings[player_id] = embeddings[idx]

            # Store metadata
            self._store_player_metadata(player_data)

            # Create position clusters
            self._create_position_clusters()

            # Save embeddings
            self._save_embeddings()

            self.is_fitted = True
            self.logger.info(f"Created embeddings for {len(self.player_embeddings)} players")

            return self.player_embeddings

        except Exception as e:
            self.logger.error(f"Failed to create embeddings: {e}")
            return {}

    def _prepare_embedding_features(self, player_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features that capture player similarity"""
        try:
            features = pd.DataFrame()

            # Basic stats (normalized)
            features['total_points_norm'] = self._normalize_feature(player_data.get('total_points', 0))
            features['form_norm'] = self._normalize_feature(player_data.get('form', 0))
            features['price_norm'] = self._normalize_feature(player_data.get('now_cost', 50) / 10)
            features['minutes_norm'] = self._normalize_feature(player_data.get('minutes', 0))

            # Performance metrics
            features['goals_norm'] = self._normalize_feature(player_data.get('goals_scored', 0))
            features['assists_norm'] = self._normalize_feature(player_data.get('assists', 0))
            features['bonus_norm'] = self._normalize_feature(player_data.get('bonus', 0))
            features['bps_norm'] = self._normalize_feature(player_data.get('bps', 0))

            # Advanced metrics
            features['influence_norm'] = self._normalize_feature(player_data.get('influence', 0))
            features['creativity_norm'] = self._normalize_feature(player_data.get('creativity', 0))
            features['threat_norm'] = self._normalize_feature(player_data.get('threat', 0))
            features['ict_index_norm'] = self._normalize_feature(player_data.get('ict_index', 0))

            # Derived features
            features['points_per_game'] = features['total_points_norm'] / np.maximum(1, player_data.get('minutes', 1) / 90)
            features['goals_per_90'] = features['goals_norm'] * 90 / np.maximum(1, player_data.get('minutes', 1))
            features['assists_per_90'] = features['assists_norm'] * 90 / np.maximum(1, player_data.get('minutes', 1))

            # Position encoding
            position_id = player_data.get('element_type', 3)
            features['is_gkp'] = (position_id == 1).astype(int)
            features['is_def'] = (position_id == 2).astype(int)
            features['is_mid'] = (position_id == 3).astype(int)
            features['is_fwd'] = (position_id == 4).astype(int)

            # Price tier encoding
            price = player_data.get('now_cost', 50) / 10
            features['is_budget'] = (price < 5.0).astype(int)
            features['is_mid_price'] = ((price >= 5.0) & (price < 8.0)).astype(int)
            features['is_premium'] = (price >= 8.0).astype(int)

            # Team context (approximate)
            features['team_id_norm'] = self._normalize_feature(player_data.get('team', 1))

            # Ownership and transfers
            features['selected_norm'] = self._normalize_feature(player_data.get('selected', 0))
            features['transfers_in_norm'] = self._normalize_feature(player_data.get('transfers_in_event', 0))
            features['transfers_out_norm'] = self._normalize_feature(player_data.get('transfers_out_event', 0))

            # Fill any NaN values
            features = features.fillna(0)

            # Store feature names
            self.feature_names = list(features.columns)

            return features

        except Exception as e:
            self.logger.error(f"Feature preparation failed: {e}")
            return pd.DataFrame()

    def _normalize_feature(self, values: pd.Series) -> pd.Series:
        """Normalize a feature to 0-1 range"""
        try:
            values = pd.to_numeric(values, errors='coerce').fillna(0)
            min_val = values.min()
            max_val = values.max()

            if max_val == min_val:
                return pd.Series([0.5] * len(values))

            return (values - min_val) / (max_val - min_val)

        except Exception:
            return pd.Series([0.5] * len(values))

    def _store_player_metadata(self, player_data: pd.DataFrame) -> None:
        """Store player metadata for similarity analysis"""
        self.player_metadata = {}

        for _, player in player_data.iterrows():
            player_id = int(player.get('id', 0))
            if player_id > 0:
                self.player_metadata[player_id] = {
                    'name': str(player.get('web_name', '')),
                    'position_id': int(player.get('element_type', 3)),
                    'position': {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}.get(
                        int(player.get('element_type', 3)), 'MID'
                    ),
                    'team_id': int(player.get('team', 1)),
                    'price': float(player.get('now_cost', 50)) / 10,
                    'total_points': int(player.get('total_points', 0)),
                    'form': float(player.get('form', 0)),
                }

    def _create_position_clusters(self) -> None:
        """Create position-specific clusters for better similarity"""
        try:
            self.position_clusters = {}

            for position_id in [1, 2, 3, 4]:
                # Get players for this position
                position_players = [
                    pid for pid, meta in self.player_metadata.items()
                    if meta['position_id'] == position_id
                ]

                if len(position_players) < 3:  # Need minimum players for clustering
                    continue

                # Get embeddings for position players
                position_embeddings = np.array([
                    self.player_embeddings[pid] for pid in position_players
                ])

                # Create clusters
                n_clusters = min(5, len(position_players) // 3)  # 3+ players per cluster
                if n_clusters >= 2:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    cluster_labels = kmeans.fit_predict(position_embeddings)

                    # Store cluster information
                    self.position_clusters[position_id] = {
                        'players': position_players,
                        'labels': cluster_labels,
                        'centers': kmeans.cluster_centers_
                    }

        except Exception as e:
            self.logger.error(f"Clustering failed: {e}")

    def find_similar_players(self, player_id: int, n_similar: int = 5,
                           same_position_only: bool = True) -> List[PlayerSimilarity]:
        """Find most similar players to given player"""
        try:
            if not self.is_fitted or player_id not in self.player_embeddings:
                return []

            target_embedding = self.player_embeddings[player_id]
            target_meta = self.player_metadata.get(player_id, {})

            similarities = []

            for other_id, other_embedding in self.player_embeddings.items():
                if other_id == player_id:
                    continue

                other_meta = self.player_metadata.get(other_id, {})

                # Filter by position if requested
                if same_position_only:
                    if target_meta.get('position_id') != other_meta.get('position_id'):
                        continue

                # Calculate similarity
                similarity = cosine_similarity(
                    target_embedding.reshape(1, -1),
                    other_embedding.reshape(1, -1)
                )[0][0]

                # Identify key similarities
                key_similarities = self._identify_key_similarities(target_meta, other_meta)

                similar_player = PlayerSimilarity(
                    player_id=other_id,
                    player_name=other_meta.get('name', ''),
                    similarity_score=float(similarity),
                    position=other_meta.get('position', ''),
                    team=str(other_meta.get('team_id', '')),
                    price=float(other_meta.get('price', 0)),
                    key_similarities=key_similarities
                )

                similarities.append(similar_player)

            # Sort by similarity and return top N
            similarities.sort(key=lambda x: x.similarity_score, reverse=True)
            return similarities[:n_similar]

        except Exception as e:
            self.logger.error(f"Similarity search failed: {e}")
            return []

    def predict_for_new_player(self, new_player_data: Dict[str, Any],
                             reference_predictions: Dict[int, float]) -> float:
        """Predict performance for new player based on similar players"""
        try:
            # Create temporary embedding for new player
            temp_df = pd.DataFrame([new_player_data])
            temp_features = self._prepare_embedding_features(temp_df)

            if temp_features.empty:
                return 0.0

            # Scale features using existing scaler
            scaled_features = self.scaler.transform(temp_features)
            new_embedding = self.pca.transform(scaled_features)[0]

            # Find similar players
            similarities = []
            position_id = new_player_data.get('element_type', 3)

            for player_id, embedding in self.player_embeddings.items():
                player_meta = self.player_metadata.get(player_id, {})

                # Only compare with same position players
                if player_meta.get('position_id') != position_id:
                    continue

                # Calculate similarity
                similarity = cosine_similarity(
                    new_embedding.reshape(1, -1),
                    embedding.reshape(1, -1)
                )[0][0]

                if player_id in reference_predictions:
                    similarities.append({
                        'player_id': player_id,
                        'similarity': similarity,
                        'prediction': reference_predictions[player_id]
                    })

            if not similarities:
                return 0.0

            # Weight predictions by similarity
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            top_similar = similarities[:5]  # Top 5 most similar

            weighted_prediction = 0.0
            total_weight = 0.0

            for sim in top_similar:
                weight = sim['similarity']
                weighted_prediction += weight * sim['prediction']
                total_weight += weight

            if total_weight > 0:
                return weighted_prediction / total_weight
            else:
                return 0.0

        except Exception as e:
            self.logger.error(f"New player prediction failed: {e}")
            return 0.0

    def _identify_key_similarities(self, player1_meta: Dict, player2_meta: Dict) -> List[str]:
        """Identify key similarities between two players"""
        similarities = []

        # Price similarity
        price_diff = abs(player1_meta.get('price', 0) - player2_meta.get('price', 0))
        if price_diff < 0.5:
            similarities.append("Similar price range")

        # Performance similarity
        points_diff = abs(player1_meta.get('total_points', 0) - player2_meta.get('total_points', 0))
        if points_diff < 20:
            similarities.append("Similar points total")

        # Form similarity
        form_diff = abs(player1_meta.get('form', 0) - player2_meta.get('form', 0))
        if form_diff < 1.0:
            similarities.append("Similar form")

        # Same team
        if player1_meta.get('team_id') == player2_meta.get('team_id'):
            similarities.append("Same team")

        return similarities[:3]  # Return top 3 similarities

    def get_position_archetypes(self, position_id: int) -> Dict[str, List[int]]:
        """Get player archetypes for a specific position"""
        try:
            if position_id not in self.position_clusters:
                return {}

            cluster_info = self.position_clusters[position_id]
            archetypes = {}

            for cluster_id in range(len(cluster_info['centers'])):
                cluster_players = [
                    player_id for i, player_id in enumerate(cluster_info['players'])
                    if cluster_info['labels'][i] == cluster_id
                ]

                # Give archetype a descriptive name based on cluster center
                archetype_name = f"Archetype_{cluster_id + 1}"
                archetypes[archetype_name] = cluster_players

            return archetypes

        except Exception as e:
            self.logger.error(f"Archetype extraction failed: {e}")
            return {}

    def analyze_player_profile(self, player_id: int) -> Dict[str, Any]:
        """Analyze a player's profile and provide insights"""
        try:
            if player_id not in self.player_embeddings:
                return {}

            player_meta = self.player_metadata.get(player_id, {})

            # Find similar players
            similar_players = self.find_similar_players(player_id, n_similar=5)

            # Determine archetype
            archetype = "Unknown"
            position_id = player_meta.get('position_id', 3)

            if position_id in self.position_clusters:
                player_embedding = self.player_embeddings[player_id]
                cluster_centers = self.position_clusters[position_id]['centers']

                # Find closest cluster center
                distances = [
                    np.linalg.norm(player_embedding - center)
                    for center in cluster_centers
                ]
                closest_cluster = np.argmin(distances)
                archetype = f"Type_{closest_cluster + 1}_{player_meta.get('position', 'Unknown')}"

            analysis = {
                'player_name': player_meta.get('name', ''),
                'position': player_meta.get('position', ''),
                'archetype': archetype,
                'similar_players': [
                    {
                        'name': sim.player_name,
                        'similarity': round(sim.similarity_score, 3),
                        'key_factors': sim.key_similarities
                    }
                    for sim in similar_players
                ],
                'unique_traits': self._identify_unique_traits(player_id),
                'transfer_alternatives': [sim.player_name for sim in similar_players[:3]]
            }

            return analysis

        except Exception as e:
            self.logger.error(f"Profile analysis failed: {e}")
            return {}

    def _identify_unique_traits(self, player_id: int) -> List[str]:
        """Identify unique traits of a player"""
        try:
            player_meta = self.player_metadata.get(player_id, {})
            traits = []

            # Price-based traits
            price = player_meta.get('price', 0)
            if price >= 10:
                traits.append("Premium player")
            elif price <= 4.5:
                traits.append("Budget option")

            # Performance traits
            total_points = player_meta.get('total_points', 0)
            form = player_meta.get('form', 0)

            if total_points > 150:
                traits.append("High scorer")
            if form > 6:
                traits.append("Excellent form")
            elif form < 3:
                traits.append("Poor form")

            return traits

        except Exception:
            return []

    def _save_embeddings(self) -> None:
        """Save embeddings and related data"""
        try:
            # Save embedding components
            joblib.dump(self.scaler, "models/embeddings/scaler.pkl")
            joblib.dump(self.pca, "models/embeddings/pca.pkl")

            # Save embeddings and metadata
            np.save("data/embeddings/player_embeddings.npy",
                   {pid: emb for pid, emb in self.player_embeddings.items()})

            joblib.dump(self.player_metadata, "data/embeddings/player_metadata.pkl")
            joblib.dump(self.position_clusters, "data/embeddings/position_clusters.pkl")

            # Save configuration
            config = {
                'embedding_dim': self.embedding_dim,
                'feature_names': self.feature_names,
                'is_fitted': self.is_fitted
            }
            joblib.dump(config, "models/embeddings/config.pkl")

            self.logger.info("Embeddings saved successfully")

        except Exception as e:
            self.logger.error(f"Failed to save embeddings: {e}")

    def _load_embeddings(self) -> bool:
        """Load existing embeddings"""
        try:
            # Check if files exist
            required_files = [
                "models/embeddings/scaler.pkl",
                "models/embeddings/pca.pkl",
                "data/embeddings/player_embeddings.npy",
                "data/embeddings/player_metadata.pkl",
                "models/embeddings/config.pkl"
            ]

            if not all(Path(f).exists() for f in required_files):
                return False

            # Load components
            self.scaler = joblib.load("models/embeddings/scaler.pkl")
            self.pca = joblib.load("models/embeddings/pca.pkl")

            # Load embeddings and metadata
            self.player_embeddings = np.load("data/embeddings/player_embeddings.npy",
                                           allow_pickle=True).item()
            self.player_metadata = joblib.load("data/embeddings/player_metadata.pkl")

            if Path("data/embeddings/position_clusters.pkl").exists():
                self.position_clusters = joblib.load("data/embeddings/position_clusters.pkl")

            # Load configuration
            config = joblib.load("models/embeddings/config.pkl")
            self.embedding_dim = config.get('embedding_dim', 32)
            self.feature_names = config.get('feature_names', [])
            self.is_fitted = config.get('is_fitted', False)

            return True

        except Exception as e:
            self.logger.error(f"Failed to load embeddings: {e}")
            return False