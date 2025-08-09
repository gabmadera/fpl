from __future__ import annotations

import pandas as pd
from .data_collector import DataCollector
from .utils import ensure_dirs


def build_player_index() -> str:
    collector = DataCollector()
    data = collector.collect_all_data()
    players = data["players"]
    teams = data.get("teams", pd.DataFrame())

    if not teams.empty:
        teams = teams.rename(columns={"id": "team_id", "name": "team_name"})
        players = players.merge(teams[["team_id", "team_name"]], on="team_id", how="left")
    else:
        players["team_name"] = None

    # Normalize column order
    cols = [
        "player_id",
        "name",
        "position",
        "team_id",
        "team_name",
        "price",
    ]
    index_df = players[cols].copy()
    ensure_dirs("data/processed")
    out_path = "data/processed/player_index.csv"
    index_df.to_csv(out_path, index=False)
    return out_path

