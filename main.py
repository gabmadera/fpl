#!/usr/bin/env python3
"""
FPL AI System - 2025-26 Season
Main CLI Entry Point

Commands:
  validate  - Validate environment and API health
  setup     - Initial setup and sample data collection
  analyze   - Run weekly analysis pipeline (stub)
  schedule  - Start automated scheduler
  chips     - Print chip strategy (stub)
  afcon     - Print AFCON impact (stub)
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.security import SecurityManager
from src.data_collector import DataCollector
from src.scheduler import FPLScheduler
from src.chip_strategy import ChipStrategyManager
from src.db import init_db
from src.pipeline import DataPipeline
from src.historical_loader import HistoricalDataLoader
from src.ml_pipeline import MLPipeline
from src.player_index import build_player_index
from src.web_app import app as web_app  # for uvicorn


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        return

    command = sys.argv[1].lower()
    security = SecurityManager()

    if command == "validate":
        validate_system(security)
    elif command == "setup":
        setup_system(security)
    elif command == "analyze":
        run_weekly_analysis()
    elif command == "schedule":
        start_scheduler()
    elif command == "dbinit":
        init_db_command()
    elif command == "collect":
        collect_command()
    elif command == "scrape":
        scrape_command()
    elif command == "historical":
        historical_command()
    elif command == "train":
        train_command()
    elif command == "predict":
        predict_command()
    elif command == "build_index":
        build_index_command()
    elif command == "web":
        start_web()
    elif command == "chips":
        analyze_chip_strategy()
    elif command == "afcon":
        check_afcon_impact()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)


def validate_system(security: SecurityManager) -> bool:
    print("Validating FPL AI system for 2025-26...")
    if not security.validate_environment():
        print("Environment validation failed")
        return False

    collector = DataCollector()
    try:
        bootstrap = collector.fetch_fpl_data("bootstrap-static/")
        if not bootstrap or not bootstrap.get("elements"):
            print("FPL API not accessible or returned no elements")
            return False
        print("FPL API accessible")
    except Exception as exc:  # noqa: BLE001
        print(f"FPL API error: {exc}")
        return False

    print("System validation complete - ready for 2025-26!")
    return True


def setup_system(security: SecurityManager) -> None:
    print("Setting up FPL AI system for 2025-26...")
    if not validate_system(security):
        print("Setup failed - fix validation errors first")
        return

    # Create folders commonly used at runtime
    for folder in ["data/raw", "data/processed", "data/historical", "data/afcon", "models/2025_26", "models/archived", "logs", "reports"]:
        Path(folder).mkdir(parents=True, exist_ok=True)

    # Collect one sample of data
    collector = DataCollector()
    _ = collector.collect_all_data()

    print("System setup complete for 2025-26!")
    print("Next:")
    print("  - python main.py analyze")
    print("  - python main.py schedule")


def analyze_chip_strategy() -> None:
    print("Analyzing chip strategy (stub)...")
    collector = DataCollector()
    datasets = collector.collect_all_data()
    chip_manager = ChipStrategyManager()
    current_gw = 1  # TODO: fetch from API
    recs = chip_manager.recommend_chips(datasets.get("players"), current_gw, {})
    if not recs:
        print("Hold chips - no clear opportunities")
    for rec in recs:
        print(f"- {rec.chip_type} (Set {rec.chip_set}) GW{rec.gameweek}: {rec.reasoning} | +{rec.expected_gain:.1f} pts | conf {rec.confidence:.0%}")


def check_afcon_impact() -> None:
    print("AFCON impact (stub)...")
    collector = DataCollector()
    _ = collector.collect_all_data()
    print("AFCON analysis completed (stub)")


def run_weekly_analysis() -> None:
    scheduler = FPLScheduler()
    scheduler.weekly_analysis_pipeline()


def start_scheduler() -> None:
    scheduler = FPLScheduler()
    scheduler.start_scheduler()


def init_db_command() -> None:
    init_db()
    print("Database initialized/migrated (basic metadata created)")


def collect_command() -> None:
    pipe = DataPipeline()
    out = pipe.collect_fpl_snapshots()
    print(f"Saved FPL snapshots: {out}")


def scrape_command() -> None:
    pipe = DataPipeline()
    fb = pipe.scrape_fbref()
    news = pipe.scrape_news()
    print(f"Saved FBref: {fb}\nSaved News: {news}")


def historical_command() -> None:
    loader = HistoricalDataLoader()
    req = loader.stage_placeholders()
    print(f"Historical requirements staged at {req}")


def train_command() -> None:
    ml = MLPipeline()
    res = ml.train()
    print(f"Training complete: {res}")


def predict_command() -> None:
    ml = MLPipeline()
    preds = ml.predict_current()
    print(f"Predictions saved: data/processed/predictions_current.csv ({len(preds)} rows)")


def build_index_command() -> None:
    path = build_player_index()
    print(f"Player index saved: {path}")


def start_web() -> None:
    try:
        import uvicorn
    except Exception:
        print("uvicorn not installed; run: pip install uvicorn")
        return
    uvicorn.run("src.web_app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()

