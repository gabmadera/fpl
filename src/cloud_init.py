"""
Cloud Initialization System for Render Deployment
Bootstraps data, models, and database on first startup
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pickle
import numpy as np
from datetime import datetime, timedelta


class CloudInitializer:
    """Initialize FPL AI system for cloud deployment"""

    def __init__(self):
        self.logger = logging.getLogger("CloudInit")
        self.is_cloud = os.getenv('RENDER', False) or os.getenv('PORT', False)
        self.data_dir = Path("data")
        self.models_dir = Path("models")

    def initialize_system(self) -> Dict[str, Any]:
        """Initialize the complete system for cloud deployment"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "is_cloud": self.is_cloud,
            "steps_completed": [],
            "errors": []
        }

        try:
            # Step 1: Create directory structure
            self._create_directories()
            results["steps_completed"].append("directories_created")

            # Step 2: Initialize minimal database
            self._initialize_database()
            results["steps_completed"].append("database_initialized")

            # Step 3: Create bootstrap data
            self._create_bootstrap_data()
            results["steps_completed"].append("bootstrap_data_created")

            # Step 4: Initialize minimal models
            self._initialize_minimal_models()
            results["steps_completed"].append("minimal_models_created")

            # Step 5: Setup cache and state
            self._setup_cache_and_state()
            results["steps_completed"].append("cache_setup_complete")

            # Step 6: Initialize accuracy tracking
            self._initialize_accuracy_tracking()
            results["steps_completed"].append("accuracy_tracking_initialized")

            results["status"] = "success"
            results["message"] = "Cloud initialization completed successfully"

        except Exception as e:
            results["status"] = "error"
            results["message"] = f"Initialization failed: {str(e)}"
            results["errors"].append(str(e))
            self.logger.error(f"Cloud initialization failed: {e}")

        return results

    def _create_directories(self):
        """Create all necessary directories"""
        directories = [
            "data/accuracy_tracking",
            "data/cache",
            "data/processed",
            "data/raw",
            "data/scheduler",
            "data/automated_retraining",
            "models/2025_26",
            "models/maximum_accuracy",
            "logs/retraining",
            "reports"
        ]

        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created directory: {dir_path}")

    def _initialize_database(self):
        """Initialize database structure - PostgreSQL for cloud, SQLite for local"""
        if self.is_cloud:
            self._initialize_postgresql()
        else:
            self._initialize_sqlite()

    def _initialize_postgresql(self):
        """Initialize PostgreSQL database for cloud deployment"""
        try:
            import psycopg2
            from psycopg2 import sql

            # Get database URL from environment
            database_url = os.getenv('DATABASE_URL')
            if not database_url:
                self.logger.warning("DATABASE_URL not found, falling back to file-based storage")
                self._initialize_file_storage()
                return

            # Create connection
            conn = psycopg2.connect(database_url)
            cursor = conn.cursor()

            # Create essential tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS gameweeks (
                    id INTEGER PRIMARY KEY,
                    name VARCHAR(50),
                    is_current BOOLEAN DEFAULT FALSE,
                    is_next BOOLEAN DEFAULT FALSE,
                    finished BOOLEAN DEFAULT FALSE,
                    deadline_time TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS teams (
                    id INTEGER PRIMARY KEY,
                    name VARCHAR(100),
                    short_name VARCHAR(10),
                    strength INTEGER DEFAULT 3,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_state (
                    key VARCHAR(100) PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Insert initial data
            cursor.execute("""
                INSERT INTO gameweeks (id, name, is_current, finished, deadline_time)
                VALUES (6, 'Gameweek 6', TRUE, FALSE, '2024-09-28T10:30:00Z')
                ON CONFLICT (id) DO NOTHING;
            """)

            cursor.execute("""
                INSERT INTO system_state (key, value)
                VALUES ('initialized', 'true'), ('current_gameweek', '6')
                ON CONFLICT (key) DO UPDATE SET
                value = EXCLUDED.value, updated_at = CURRENT_TIMESTAMP;
            """)

            conn.commit()
            cursor.close()
            conn.close()

            self.logger.info("PostgreSQL database initialized successfully")

        except Exception as e:
            self.logger.error(f"PostgreSQL initialization failed: {e}")
            # Fallback to file storage
            self._initialize_file_storage()

    def _initialize_sqlite(self):
        """Initialize SQLite database for local development"""
        try:
            import sqlite3

            # Create SQLite database
            db_file = self.data_dir / "fpl.db"
            conn = sqlite3.connect(str(db_file))
            cursor = conn.cursor()

            # Create tables (same structure as PostgreSQL)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS gameweeks (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    is_current BOOLEAN DEFAULT 0,
                    is_next BOOLEAN DEFAULT 0,
                    finished BOOLEAN DEFAULT 0,
                    deadline_time TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS teams (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    short_name TEXT,
                    strength INTEGER DEFAULT 3,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_state (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Insert initial data
            cursor.execute("""
                INSERT OR IGNORE INTO gameweeks (id, name, is_current, finished, deadline_time)
                VALUES (6, 'Gameweek 6', 1, 0, '2024-09-28T10:30:00Z');
            """)

            cursor.execute("""
                INSERT OR REPLACE INTO system_state (key, value)
                VALUES ('initialized', 'true'), ('current_gameweek', '6');
            """)

            conn.commit()
            conn.close()

            self.logger.info("SQLite database initialized successfully")

        except Exception as e:
            self.logger.error(f"SQLite initialization failed: {e}")
            # Fallback to file storage
            self._initialize_file_storage()

    def _initialize_file_storage(self):
        """Fallback file-based storage when database is unavailable"""
        db_data = {
            "gameweeks": {
                "5": {"finished": True, "current": False},
                "6": {"finished": False, "current": True}
            },
            "players": {},
            "teams": {},
            "last_updated": datetime.now().isoformat()
        }

        db_file = self.data_dir / "bootstrap" / "database.json"
        db_file.parent.mkdir(parents=True, exist_ok=True)

        with open(db_file, 'w') as f:
            json.dump(db_data, f, indent=2)

        self.logger.info("File-based storage initialized as fallback")

    def _create_bootstrap_data(self):
        """Create essential bootstrap data"""
        # Essential gameweek data
        gameweek_data = {
            "current_gameweek": 6,
            "last_finished": 5,
            "events": [
                {
                    "id": 5,
                    "name": "Gameweek 5",
                    "is_current": False,
                    "is_next": False,
                    "finished": True,
                    "deadline_time": "2024-09-21T10:30:00Z"
                },
                {
                    "id": 6,
                    "name": "Gameweek 6",
                    "is_current": True,
                    "is_next": False,
                    "finished": False,
                    "deadline_time": "2024-09-28T10:30:00Z"
                }
            ]
        }

        # Essential team data
        team_data = {
            "teams": [
                {"id": 1, "name": "Arsenal", "short_name": "ARS"},
                {"id": 2, "name": "Aston Villa", "short_name": "AVL"},
                {"id": 3, "name": "Bournemouth", "short_name": "BOU"},
                {"id": 4, "name": "Brentford", "short_name": "BRE"},
                {"id": 5, "name": "Brighton", "short_name": "BHA"},
                {"id": 6, "name": "Burnley", "short_name": "BUR"},
                {"id": 7, "name": "Chelsea", "short_name": "CHE"},
                {"id": 8, "name": "Crystal Palace", "short_name": "CRY"},
                {"id": 9, "name": "Everton", "short_name": "EVE"},
                {"id": 10, "name": "Fulham", "short_name": "FUL"},
                {"id": 11, "name": "Liverpool", "short_name": "LIV"},
                {"id": 12, "name": "Luton", "short_name": "LUT"},
                {"id": 13, "name": "Man City", "short_name": "MCI"},
                {"id": 14, "name": "Man Utd", "short_name": "MUN"},
                {"id": 15, "name": "Newcastle", "short_name": "NEW"},
                {"id": 16, "name": "Nott'm Forest", "short_name": "NFO"},
                {"id": 17, "name": "Sheffield Utd", "short_name": "SHU"},
                {"id": 18, "name": "Tottenham", "short_name": "TOT"},
                {"id": 19, "name": "West Ham", "short_name": "WHU"},
                {"id": 20, "name": "Wolves", "short_name": "WOL"}
            ]
        }

        # Save bootstrap data
        bootstrap_dir = self.data_dir / "bootstrap"
        bootstrap_dir.mkdir(parents=True, exist_ok=True)

        with open(bootstrap_dir / "gameweeks.json", 'w') as f:
            json.dump(gameweek_data, f, indent=2)

        with open(bootstrap_dir / "teams.json", 'w') as f:
            json.dump(team_data, f, indent=2)

        self.logger.info("Bootstrap data created")

    def _initialize_minimal_models(self):
        """Create minimal ML models for immediate use"""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        import joblib

        # Create a simple random forest model
        model = RandomForestRegressor(n_estimators=10, random_state=42)

        # Create dummy training data
        X_dummy = np.random.rand(100, 10)  # 100 samples, 10 features
        y_dummy = np.random.rand(100) * 10  # Points between 0-10

        # Train the model
        model.fit(X_dummy, y_dummy)

        # Save the model
        model_dir = self.models_dir / "minimal"
        model_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, model_dir / "points_predictor.pkl")

        # Create model metadata
        metadata = {
            "model_type": "RandomForestRegressor",
            "created_at": datetime.now().isoformat(),
            "features": [f"feature_{i}" for i in range(10)],
            "target": "total_points",
            "accuracy": 0.65,  # Placeholder
            "status": "minimal_bootstrap"
        }

        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info("Minimal models initialized")

    def _setup_cache_and_state(self):
        """Setup cache and system state"""
        cache_dir = self.data_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Create initial cache state
        cache_state = {
            "last_api_call": datetime.now().isoformat(),
            "api_status": "unknown",
            "cached_endpoints": [],
            "cache_size": 0
        }

        with open(cache_dir / "state.json", 'w') as f:
            json.dump(cache_state, f, indent=2)

        # System state
        state_dir = self.data_dir / "system_state"
        state_dir.mkdir(parents=True, exist_ok=True)

        system_state = {
            "initialized_at": datetime.now().isoformat(),
            "version": "2.0.0",
            "cloud_deployment": self.is_cloud,
            "last_training": None,
            "model_status": "minimal"
        }

        with open(state_dir / "system.json", 'w') as f:
            json.dump(system_state, f, indent=2)

        self.logger.info("Cache and state setup complete")

    def _initialize_accuracy_tracking(self):
        """Initialize accuracy tracking system"""
        accuracy_dir = self.data_dir / "accuracy_tracking"
        accuracy_dir.mkdir(parents=True, exist_ok=True)

        # Create initial accuracy data
        accuracy_data = {
            "overall_accuracy": 0.0,
            "gameweek_results": [],
            "target_accuracy": 75.0,
            "last_updated": datetime.now().isoformat(),
            "improvement_trend": []
        }

        with open(accuracy_dir / "current.json", 'w') as f:
            json.dump(accuracy_data, f, indent=2)

        # Create scheduler state
        scheduler_dir = self.data_dir / "scheduler"
        scheduler_dir.mkdir(parents=True, exist_ok=True)

        scheduler_state = {
            "is_running": False,
            "last_processed_gameweek": 5,
            "current_gameweek": 6,
            "next_check": (datetime.now() + timedelta(minutes=5)).isoformat(),
            "status": "initialized"
        }

        with open(scheduler_dir / "retraining_status.json", 'w') as f:
            json.dump(scheduler_state, f, indent=2)

        self.logger.info("Accuracy tracking initialized")

    def is_initialized(self) -> bool:
        """Check if system is already initialized"""
        state_file = self.data_dir / "system_state" / "system.json"
        return state_file.exists()

    def get_initialization_status(self) -> Dict[str, Any]:
        """Get current initialization status"""
        return {
            "is_initialized": self.is_initialized(),
            "is_cloud": self.is_cloud,
            "data_dir_exists": self.data_dir.exists(),
            "models_dir_exists": self.models_dir.exists(),
            "essential_files": {
                "gameweeks": (self.data_dir / "bootstrap" / "gameweeks.json").exists(),
                "teams": (self.data_dir / "bootstrap" / "teams.json").exists(),
                "minimal_model": (self.models_dir / "minimal" / "points_predictor.pkl").exists(),
                "system_state": (self.data_dir / "system_state" / "system.json").exists()
            }
        }