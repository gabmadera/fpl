from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import pandas as pd

from .fpl_client import FPLClient
from .fbref_scraper import FBRefScraper
# News scraper removed - functionality disabled


class DataPipeline:
    def __init__(self) -> None:
        self.fpl = FPLClient()
        self.fbref = FBRefScraper()
        # News scraper removed - functionality disabled
        Path("data/raw").mkdir(parents=True, exist_ok=True)

    def collect_fpl_snapshots(self) -> dict[str, str]:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        files: dict[str, str] = {}
        bootstrap = self.fpl.bootstrap_static()
        fixtures = self.fpl.fixtures()
        files["bootstrap"] = self._write_json(f"data/raw/bootstrap_{ts}.json", bootstrap)
        files["fixtures"] = self._write_json(f"data/raw/fixtures_{ts}.json", fixtures)
        return files

    def scrape_fbref(self) -> dict[str, str | int]:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        x_stats = self.fbref.get_player_expected_stats()
        def_stats = self.fbref.get_defensive_contribution_stats()
        x_path = f"data/raw/fbref_expected_{ts}.csv"
        d_path = f"data/raw/fbref_defensive_{ts}.csv"
        if not x_stats.empty:
            x_stats.to_csv(x_path, index=False)
        if not def_stats.empty:
            def_stats.to_csv(d_path, index=False)
        return {"x_rows": 0 if x_stats is None else len(x_stats), "d_rows": 0 if def_stats is None else len(def_stats), "x_path": x_path, "d_path": d_path}

    def scrape_news(self) -> str:
        """News scraping disabled - returns empty result"""
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        empty_news = {"articles": [], "timestamp": ts, "status": "disabled"}
        return self._write_json(f"data/raw/news_{ts}.json", empty_news)

    def _write_json(self, path: str, data) -> str:
        with open(path, "w") as f:
            json.dump(data, f)
        return path

