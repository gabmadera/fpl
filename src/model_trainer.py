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

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        if X.empty or y.empty:
            return {"status": "no_data"}
        # Lazy import to avoid hard dependency when local env lacks libomp
        try:
            import xgboost as xgb  # type: ignore
        except Exception:
            return {"status": "xgb_missing"}
        n_splits = min(5, max(2, len(X) // 1000))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores: List[float] = []
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            model = xgb.XGBRegressor(
                n_estimators=400,
                max_depth=8,
                learning_rate=0.06,
                subsample=0.85,
                colsample_bytree=0.8,
                reg_alpha=0.0,
                reg_lambda=0.2,
                objective="reg:squarederror",
                random_state=42,
            )
            # Be compatible with multiple xgboost versions
            try:
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False, early_stopping_rounds=20)
            except TypeError:
                try:
                    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                except TypeError:
                    model.fit(X_tr, y_tr)
            pred = model.predict(X_val)
            mae = float(mean_absolute_error(y_val, pred))
            scores.append(mae)
        # Train final on full data
        self.model = xgb.XGBRegressor(
            n_estimators=400,
            max_depth=8,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=0.2,
            objective="reg:squarederror",
            random_state=42,
        )
        try:
            self.model.fit(X, y, verbose=False)
        except TypeError:
            self.model.fit(X, y)
        joblib.dump(self.model, "models/2025_26/fpl_xgb_model.pkl")
        return {"status": "ok", "cv_mae_mean": float(np.mean(scores)), "cv_mae_std": float(np.std(scores))}

    def predict(self, X: pd.DataFrame) -> pd.Series:
        try:
            if self.model is None:
                self.model = joblib.load("models/2025_26/fpl_xgb_model.pkl")
            return pd.Series(self.model.predict(X), index=X.index)
        except Exception:
            raise RuntimeError("xgboost_unavailable")

