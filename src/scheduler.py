from __future__ import annotations

import schedule
import time
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
from typing import List
import pandas as pd

from .data_collector import DataCollector
from .feature_engineer import FeatureEngineer
from .model_trainer import FPLModelTrainer
from .team_optimizer import TeamOptimizer
from .chip_strategy import ChipStrategyManager
from .config import config
from .fpl_client import FPLClient


class FPLScheduler:
    def __init__(self) -> None:
        self.collector = DataCollector()
        self.engineer = FeatureEngineer()
        self.trainer = FPLModelTrainer()
        self.optimizer = TeamOptimizer()
        self.chip_manager = ChipStrategyManager()
        self.fpl = FPLClient()

    def weekly_analysis_pipeline(self) -> None:
        try:
            print(f"Starting FPL analysis at {datetime.now()}")

            datasets = self.collector.collect_all_data()
            features = self.engineer.create_features(datasets["players"], datasets["fixtures"])  # type: ignore[index]
            predictions = self.trainer.predict_points_2025_26(features)

            team_rec = self.optimizer.optimize_team(predictions)

            current_gw = self.fpl.current_gameweek()
            # Provide fixture info to chip strategy
            fixture_context = {"fixtures": datasets.get("fixtures", pd.DataFrame())}
            chip_recs = self.chip_manager.recommend_chips(predictions, current_gw, fixture_context)

            report = self._generate_report(team_rec, predictions, chip_recs)
            self._send_alert_email(report)
            print("Weekly analysis complete")
        except Exception as exc:  # noqa: BLE001
            error_msg = f"FPL Analysis failed: {exc}"
            print(error_msg)

    def _generate_report(self, team_rec: dict, predictions: pd.DataFrame, chip_recs: List) -> str:
        current_gw = 1
        report = (
            f"FPL AI WEEKLY REPORT - GAMEWEEK {current_gw}\n"
            f"Generated: {datetime.now():%Y-%m-%d %H:%M}\n\n"
            f"Top Predicted Performers:\n"
            + predictions.head(10)[["name", "position", "team_id", "predicted_points"]].to_string(index=False)
            + "\n\n"
            f"Recommended Captain: {team_rec.get('captain', 'N/A')}\n"
            f"Recommended Vice: {team_rec.get('vice_captain', 'N/A')}\n\n"
            f"Chip Recommendations:\n"
            + ("\n".join([f"• {r.chip_type.upper()} (Set {r.chip_set}): {r.reasoning}" for r in chip_recs]) or "Hold chips - no clear opportunities")
        )
        # Persist a copy
        with open("reports/weekly_report.txt", "w") as f:
            f.write(report)
        return report

    def _send_alert_email(self, body: str, subject: str | None = None) -> None:
        subject = subject or "FPL AI Weekly Report"
        if not (config.EMAIL_USER and config.EMAIL_PASS and config.ALERT_EMAIL):
            print("Email not configured, skipping send.")
            return
        msg = MIMEMultipart()
        msg["From"] = config.EMAIL_USER
        msg["To"] = config.ALERT_EMAIL
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))
        with smtplib.SMTP(config.SMTP_SERVER, config.SMTP_PORT) as server:
            server.starttls()
            server.login(config.EMAIL_USER, config.EMAIL_PASS)
            server.send_message(msg)

    def start_scheduler(self) -> None:
        # Default weekly cadence
        schedule.every().tuesday.at("14:00").do(self.weekly_analysis_pipeline)
        schedule.every().friday.at("14:00").do(self.weekly_analysis_pipeline)
        # Deadline-based run ~4 hours before next deadline if available
        schedule.every(60).minutes.do(self._deadline_check_and_schedule)
        print("Scheduler started (Tue/Fri 14:00 + deadline-based checks)")
        while True:
            schedule.run_pending()
            time.sleep(60)

    def _deadline_check_and_schedule(self) -> None:
        try:
            data = self.fpl.bootstrap_static()
            events = data.get("events", [])
            next_ev = next((e for e in events if e.get("is_next")), None)
            if not next_ev:
                return
            deadline_iso = next_ev.get("deadline_time")  # e.g., 2025-08-14T16:30:00Z
            if not deadline_iso:
                return
            # If within ~5h before deadline, run once
            from datetime import datetime, timezone
            import dateutil.parser
            dl = dateutil.parser.isoparse(deadline_iso)
            now = datetime.now(timezone.utc)
            delta_hours = (dl - now).total_seconds() / 3600.0
            if 0 < delta_hours <= 5:
                self.weekly_analysis_pipeline()
        except Exception:
            return

