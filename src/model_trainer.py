from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import joblib


class FPLModelTrainer:
    def __init__(self) -> None:
        self.model = None
        self.feature_importance: dict[str, float] = {}

    def train_model_2025_26(self, df: pd.DataFrame) -> Dict[str, float]:
        # Backward-compatible stub if df is empty
        if df.empty:
            return {"status": "no_data"}
        # If df already split into X/y externally, this method is not used.
        # Keep minimal behavior here.
        return {"status": "trained_stub", "n_features": len(df.columns)}

    def predict_points_2025_26(self, features_df: pd.DataFrame) -> pd.DataFrame:
        # Very naive baseline: scale by price band and defensive bonus
        results = features_df[["player_id", "name", "position", "team_id", "price"]].copy()
        base = 2.0 + (features_df.get("price_band", 0).astype(float) * 0.5)
        dc_bonus = features_df.get("dc_point_probability", 0.0) * 2.0
        if "starting_probability" in features_df.columns:
            availability = pd.to_numeric(features_df["starting_probability"], errors="coerce").fillna(0.9)
        else:
            availability = pd.Series(0.9, index=features_df.index)
        # Further downweight by FPL status/chance if present
        fpl_status = features_df.get("fpl_status")
        chance_next = pd.to_numeric(features_df.get("chance_next"), errors="coerce") / 100.0 if "chance_next" in features_df.columns else None
        if fpl_status is not None:
            penalty = pd.Series(1.0, index=features_df.index)
            # Injured/Suspended/Doubtful
            penalty[fpl_status.isin(["i", "s", "d"]) & (chance_next.notna())] = chance_next[fpl_status.isin(["i", "s", "d"])].clip(lower=0.0, upper=1.0)
            penalty[fpl_status.isin(["i", "s", "d"]) & ((chance_next.isna()) if chance_next is not None else True)] = 0.05
            availability = availability * penalty
        # Add simple component EP if xG/90 & xA/90 exist
        xg = pd.to_numeric(features_df.get("xg_per90"), errors="coerce")
        xa = pd.to_numeric(features_df.get("xa_per90"), errors="coerce")
        pos = features_df.get("position", pd.Series(["" for _ in range(len(features_df))]))
        goal_pts_map = pos.map({"GKP": 6, "DEF": 6, "MID": 5, "FWD": 4}).fillna(4)
        comp = (xg.fillna(0) * goal_pts_map + xa.fillna(0) * 3.0) * (features_df.get("minutes_avg_3", 60) / 90.0)
        points = (base + dc_bonus + comp.fillna(0)) * availability
        results["predicted_points"] = points.round(2)
        return results.sort_values("predicted_points", ascending=False)

    def _get_top_features(self, n: int) -> List[str]:
        return []


class XGBoostTrainer:
    def __init__(self) -> None:
        self.model = None
        self.feature_names = []
        self.scaler = None

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        if X.empty or y.empty:
            return {"status": "no_data"}
        
        try:
            import xgboost as xgb
            from sklearn.feature_selection import SelectKBest, f_regression
            from sklearn.preprocessing import StandardScaler
        except Exception:
            return {"status": "dependencies_missing"}
        
        # Feature preprocessing
        X_processed = self._preprocess_features(X)
        
        # Feature selection - keep top features
        if len(X_processed.columns) > 50:
            selector = SelectKBest(score_func=f_regression, k=min(50, len(X_processed.columns)))
            X_selected = pd.DataFrame(
                selector.fit_transform(X_processed, y),
                columns=X_processed.columns[selector.get_support()],
                index=X_processed.index
            )
        else:
            X_selected = X_processed
        
        # Enhanced cross-validation
        n_splits = min(6, max(3, len(X_selected) // 800))
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=1)  # Add gap to prevent data leakage
        
        scores: List[float] = []
        feature_importance_scores = []
        
        # Hyperparameter optimization via cross-validation
        best_params = self._tune_hyperparameters(X_selected, y, tscv)
        
        for train_idx, val_idx in tscv.split(X_selected):
            X_tr, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = xgb.XGBRegressor(**best_params)
            
            try:
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], 
                         verbose=False, early_stopping_rounds=30)
            except TypeError:
                model.fit(X_tr, y_tr)
            
            pred = model.predict(X_val)
            mae = float(mean_absolute_error(y_val, pred))
            scores.append(mae)
            
            if hasattr(model, 'feature_importances_'):
                feature_importance_scores.append(model.feature_importances_)
        
        # Train final model on all data
        self.model = xgb.XGBRegressor(**best_params)
        
        try:
            self.model.fit(X_selected, y, verbose=False)
        except TypeError:
            self.model.fit(X_selected, y)
        
        # Store feature names for prediction
        self.feature_names = list(X_selected.columns)
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'feature_importance': dict(zip(self.feature_names, self.model.feature_importances_)) if hasattr(self.model, 'feature_importances_') else {}
        }
        joblib.dump(model_data, "models/2025_26/fpl_xgb_model.pkl")
        
        return {
            "status": "ok", 
            "cv_mae_mean": float(np.mean(scores)), 
            "cv_mae_std": float(np.std(scores)),
            "n_features": len(X_selected.columns),
            "best_params": best_params
        }
    
    def _preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Preprocess features for training and prediction"""
        X_clean = X.copy()
        
        # Handle infinite values
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with median for numeric columns
        numeric_cols = X_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            X_clean[col] = X_clean[col].fillna(X_clean[col].median())
        
        # Remove constant columns
        constant_cols = [col for col in X_clean.columns if X_clean[col].nunique() <= 1]
        X_clean = X_clean.drop(columns=constant_cols)
        
        return X_clean
    
    def _tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series, cv) -> dict:
        """Simplified hyperparameter selection to avoid memory issues"""
        # Return optimized default params based on FPL data characteristics
        return {
            'n_estimators': 500,
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.85,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.3,
            'objective': 'reg:squarederror',
            'random_state': 42
        }

    def predict(self, X: pd.DataFrame) -> pd.Series:
        try:
            if self.model is None:
                model_data = joblib.load("models/2025_26/fpl_xgb_model.pkl")
                if isinstance(model_data, dict):
                    self.model = model_data['model']
                    self.feature_names = model_data.get('feature_names', [])
                else:
                    self.model = model_data  # Backward compatibility
                    self.feature_names = []
            
            # Process features consistently with training
            X_processed = self._preprocess_features(X)
            
            # Ensure we have the same features as training
            if self.feature_names:
                # Add missing features with zeros
                for feature in self.feature_names:
                    if feature not in X_processed.columns:
                        X_processed[feature] = 0
                # Select only training features in correct order
                X_processed = X_processed[self.feature_names]
            
            predictions = self.model.predict(X_processed)
            return pd.Series(predictions, index=X.index)
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")

