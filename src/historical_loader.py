from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import json


class HistoricalDataLoader:
    """Skeleton to stage historical FPL datasets for ML training.

    Sources to consider:
    - https://github.com/vaastav/Fantasy-Premier-League
    - https://github.com/wiscostret/fpldata
    - Kaggle FPL datasets (manual download)
    """

    def __init__(self) -> None:
        Path("data/historical").mkdir(parents=True, exist_ok=True)

    def list_required(self) -> List[str]:
        return ["2021-22", "2022-23", "2023-24", "2024-25"]

    def stage_placeholders(self) -> str:
        path = "data/historical/REQUIRED.json"
        payload: Dict[str, List[str]] = {"seasons": self.list_required()}
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        return path

    def integrate_from_local_csvs(self, folder: str) -> Dict[str, int]:
        """Placeholder for merging locally downloaded CSVs into a single training set."""
        # TODO: Implement merging logic once CSVs are present
        return {"merged_rows": 0}

