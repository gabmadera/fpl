from __future__ import annotations

import pandas as pd
from typing import Dict
import pulp
from .settings import load_exclusions


class TeamOptimizer:
    def optimize_team(self, predictions: pd.DataFrame, budget_cap: float = 100.0) -> Dict:
        """LP optimizer: select 11 players under constraints with captain choice.

        Constraints:
        - Exactly 11 starters
        - Positions: 1 GKP, 3 DEF, 4 MID, 3 FWD
        - Max 3 from the same team
        - Total price <= budget_cap
        Objective:
        - Maximize predicted_points + captain bonus (captain doubles points)
        """
        if predictions is None or predictions.empty:
            return {
                "transfers_in": [],
                "transfers_out": [],
                "net_gain": 0,
                "captain": None,
                "vice_captain": None,
                "confidence": "Low",
            }
        df = predictions.copy()
        # Sanitize key numeric fields
        df["predicted_points"] = pd.to_numeric(df.get("predicted_points"), errors="coerce")
        df["price"] = pd.to_numeric(df.get("price"), errors="coerce")
        # Replace non-finite predicted points with 0; price NaN with very high so they won't be chosen
        df["predicted_points"] = df["predicted_points"].replace([float("inf"), float("-inf")], pd.NA).fillna(0.0).clip(lower=0.0)
        df["price"] = df["price"].replace([float("inf"), float("-inf")], pd.NA).fillna(1000.0)
        # Exclude manually flagged players (injuries/suspensions/news-driven)
        excl = load_exclusions()
        if "player_id" in df.columns and excl:
            df = df[~df["player_id"].isin(list(excl))]
        # Availability filter: drop suspended/injured or very low chance
        if "fpl_status" in df.columns:
            chance = pd.to_numeric(df.get("chance_next"), errors="coerce")
            df = df[~df["fpl_status"].isin(["s", "i"])]
            df = df[~((df["fpl_status"] == "d") & (chance.notna()) & (chance < 50))]
        for c in ["player_id", "position", "team_id", "predicted_points", "name", "price"]:
            if c not in df.columns:
                return {"error": f"Missing column {c}", "confidence": "Low"}

        # Ensure enough players per position exist; else fallback greedy
        pos_limits = {"GKP": 1, "DEF": 3, "MID": 4, "FWD": 3}
        for pos, limit in pos_limits.items():
            if (df["position"] == pos).sum() < limit:
                return self._greedy_fallback(df)

        # Decision variables
        idx = list(df.index)
        x = pulp.LpVariable.dicts("x", idx, lowBound=0, upBound=1, cat=pulp.LpBinary)  # selected
        y = pulp.LpVariable.dicts("y", idx, lowBound=0, upBound=1, cat=pulp.LpBinary)  # captain

        # Problem
        prob = pulp.LpProblem("FPL_Optimizer", pulp.LpMaximize)

        # Objective: sum points for starters + extra for captain (captain adds one more unit)
        prob += (
            pulp.lpSum([x[i] * float(df.at[i, "predicted_points"]) for i in idx])
            + pulp.lpSum([y[i] * float(df.at[i, "predicted_points"]) for i in idx])
        )

        # Exactly 11 starters
        prob += pulp.lpSum([x[i] for i in idx]) == 11

        # Captain exactly one and must be selected
        prob += pulp.lpSum([y[i] for i in idx]) == 1
        for i in idx:
            prob += y[i] <= x[i]

        # Position constraints
        for pos, limit in pos_limits.items():
            prob += pulp.lpSum([x[i] for i in idx if df.at[i, "position"] == pos]) == limit

        # Team constraints: <= 3 per team
        for team_id in df["team_id"].dropna().unique():
            prob += pulp.lpSum([x[i] for i in idx if df.at[i, "team_id"] == team_id]) <= 3

        # Budget constraint
        prob += pulp.lpSum([x[i] * float(df.at[i, "price"]) for i in idx]) <= float(budget_cap)

        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        if prob.status != pulp.LpStatusOptimal:
            # Fallback to greedy if infeasible
            return self._greedy_fallback(df)

        chosen = [i for i in idx if pulp.value(x[i]) >= 0.5]
        captain_idx = next((i for i in idx if pulp.value(y[i]) >= 0.5), None)
        sel_df = df.loc[chosen].copy().sort_values("predicted_points", ascending=False)
        captain = df.at[captain_idx, "name"] if captain_idx is not None else None
        # Vice: best remaining not the captain
        if captain is not None and not sel_df.empty:
            rem = sel_df[sel_df["name"] != captain]
            vice = rem.iloc[0]["name"] if not rem.empty else None
        else:
            vice = sel_df.iloc[1]["name"] if len(sel_df) > 1 else None
        total_points = float(sel_df["predicted_points"].sum()) + (float(df.at[captain_idx, "predicted_points"]) if captain_idx is not None else 0.0)
        return {
            "selected": sel_df[["player_id", "name", "position", "team_id", "price", "predicted_points"]].to_dict(orient="records"),
            "captain": captain,
            "vice_captain": vice,
            "net_gain": total_points,
            "confidence": "Medium",
        }

    def _greedy_fallback(self, df: pd.DataFrame) -> Dict:
        # Sanitize first
        df = df.copy()
        df["predicted_points"] = pd.to_numeric(df.get("predicted_points"), errors="coerce").fillna(0.0).clip(lower=0.0)
        df["price"] = pd.to_numeric(df.get("price"), errors="coerce").fillna(1000.0)
        selected = []
        team_counts: Dict[int, int] = {}
        pos_limits = {"GKP": 1, "DEF": 3, "MID": 4, "FWD": 3}
        for _, row in df.sort_values("predicted_points", ascending=False).iterrows():
            if len(selected) >= 11:
                break
            pos = row["position"]
            tid = int(row["team_id"]) if pd.notna(row["team_id"]) else -1
            if sum(1 for r in selected if r["position"] == pos) >= pos_limits.get(pos, 0):
                continue
            if tid != -1 and team_counts.get(tid, 0) >= 3:
                continue
            selected.append(row)
            if tid != -1:
                team_counts[tid] = team_counts.get(tid, 0) + 1
        sel_df = pd.DataFrame(selected)
        captain = sel_df.iloc[0]["name"] if not sel_df.empty else None
        vice = sel_df.iloc[1]["name"] if len(sel_df) > 1 else None
        return {
            "selected": sel_df[["player_id", "name", "position", "team_id", "price", "predicted_points"]].to_dict(orient="records") if not sel_df.empty else [],
            "captain": captain,
            "vice_captain": vice,
            "net_gain": float(sel_df["predicted_points"].sum()) if not sel_df.empty else 0,
            "confidence": "Low",
        }

