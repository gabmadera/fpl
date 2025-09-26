"""
Automated Weekly Retraining Scheduler for Maximum Accuracy System
Monitors gameweek completion and triggers retraining automatically
"""

from __future__ import annotations

import asyncio
import logging
import schedule
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import json
from dataclasses import dataclass, asdict
from enum import Enum
import threading

from .fpl_client import FPLClient
from .cache_manager import CacheManager


class RetrainingStatus(Enum):
    IDLE = "idle"
    MONITORING = "monitoring"
    TRIGGERED = "triggered"
    TRAINING = "training"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class RetrainingEvent:
    """Record of retraining events"""
    gameweek: int
    trigger_time: datetime
    completion_time: Optional[datetime]
    status: RetrainingStatus
    accuracy_before: Optional[float]
    accuracy_after: Optional[float]
    trigger_reason: str
    error_message: Optional[str] = None
    model_improvements: Optional[Dict[str, Any]] = None


@dataclass
class RetrainingJob:
    """Job record for scheduled retraining"""
    gameweek: int
    scheduled_time: datetime
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class AutomatedRetrainingScheduler:
    """
    Production scheduler for automated weekly retraining
    """

    def __init__(self,
                 max_accuracy_system = None,
                 notification_callback: Optional[Callable] = None):
        self.logger = logging.getLogger(__name__)
        self.fpl = FPLClient()
        self.cache_manager = CacheManager()

        # Initialize maximum accuracy system
        if max_accuracy_system is None:
            from .maximum_accuracy_system import MaximumAccuracySystem
            self.max_accuracy_system = MaximumAccuracySystem()
        else:
            self.max_accuracy_system = max_accuracy_system

        # Notification system
        self.notification_callback = notification_callback

        # Scheduler state
        self.is_running = False
        self.current_status = RetrainingStatus.IDLE
        self.last_processed_gameweek = 0
        self.retraining_history: list[RetrainingEvent] = []
        self.current_gameweek = 1  # Will be updated from FPL API
        self.retraining_jobs: list[RetrainingJob] = []  # Track scheduled retraining jobs
        self.scheduler_thread = None  # Background scheduler thread

        # Configuration
        self.check_interval = 300  # 5 minutes in seconds
        self.check_interval_minutes = 5  # 5 minutes for schedule
        self.gameweek_completion_delay = 120  # 2 hours after gameweek ends (minutes)
        self.retrain_delay_hours = self.gameweek_completion_delay / 60  # Convert to hours
        self.max_retrain_attempts = 3

        # File paths
        self.status_file = Path("data/scheduler/retraining_status.json")
        self.history_file = Path("data/scheduler/retraining_history.jsonl")
        self.status_file.parent.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

        # Create directories
        Path("data/automated_retraining").mkdir(parents=True, exist_ok=True)
        Path("logs/retraining").mkdir(parents=True, exist_ok=True)
        Path("reports").mkdir(parents=True, exist_ok=True)

        # Initialize accuracy tracker (use max_accuracy_system's tracker if available)
        self.accuracy_tracker = getattr(self.max_accuracy_system, 'accuracy_tracker', None)

        # Load previous state
        self._load_scheduler_state()

        # Initialize current gameweek from FPL API
        self._initialize_current_gameweek()

    def _setup_logging(self):
        """Setup logging configuration for the scheduler"""
        log_dir = Path("logs/retraining")
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create scheduler-specific logger
        self.logger = logging.getLogger("AutomatedRetrainingScheduler")
        if not self.logger.handlers:
            handler = logging.FileHandler(log_dir / "automated_retraining.log")
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def start_automated_monitoring(self):
        """Start the automated monitoring and retraining system"""

        if self.is_running:
            self.logger.warning("Scheduler already running")
            return

        self.is_running = True
        self.logger.info("🚀 Starting automated FPL retraining scheduler...")

        # Schedule regular checks for gameweek completion
        schedule.every(self.check_interval_minutes).minutes.do(self._check_gameweek_completion)

        # Schedule daily health checks
        schedule.every().day.at("06:00").do(self._daily_health_check)

        # Schedule weekly performance reports
        schedule.every().sunday.at("20:00").do(self._weekly_performance_report)

        # Start scheduler in background thread
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()

        self.logger.info("✅ Automated retraining scheduler started successfully")

    def stop_automated_monitoring(self):
        """Stop the automated monitoring system"""

        self.is_running = False
        schedule.clear()
        self.logger.info("🛑 Automated retraining scheduler stopped")

    def _run_scheduler(self):
        """Main scheduler loop"""

        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Scheduler error: {e}")
                time.sleep(300)  # Wait 5 minutes on error

    def _check_gameweek_completion(self) -> Dict[str, Any]:
        """Check if a new gameweek has completed and trigger retraining"""

        try:
            # Get current FPL status
            bootstrap = self.fpl.bootstrap_static()
            events = bootstrap.get("events", [])

            # Find current and completed gameweeks
            current_gw = 1
            completed_gameweeks = []

            for event in events:
                gw_id = event.get("id", 0)
                is_current = event.get("is_current", False)
                is_finished = event.get("finished", False)

                if is_current:
                    current_gw = gw_id

                if is_finished and gw_id > self.last_processed_gameweek:
                    completed_gameweeks.append(gw_id)

            # Update current gameweek
            self.current_gameweek = current_gw

            result = {
                "timestamp": datetime.now().isoformat(),
                "current_gameweek": current_gw,
                "completed_gameweeks": completed_gameweeks,
                "last_processed": self.last_processed_gameweek,
                "action": "none"
            }

            # Process any newly completed gameweeks
            for completed_gw in completed_gameweeks:
                if completed_gw > self.last_processed_gameweek:
                    self.logger.info(f"🎯 Detected completed gameweek: GW{completed_gw}")

                    # Schedule immediate retraining
                    retrain_result = self._schedule_immediate_retraining(completed_gw)
                    result[f"retrain_gw{completed_gw}"] = retrain_result
                    result["action"] = "retraining_scheduled"

                    # Update processed gameweek
                    self.last_processed_gameweek = completed_gw

            # Save state
            self._save_scheduler_state()

            return result

        except Exception as e:
            error_msg = f"Gameweek completion check failed: {e}"
            self.logger.error(error_msg)
            return {"error": error_msg, "timestamp": datetime.now().isoformat()}

    def _schedule_immediate_retraining(self, completed_gameweek: int) -> Dict[str, Any]:
        """Schedule immediate retraining for a completed gameweek"""

        try:
            # Check if already scheduled
            existing_job = next(
                (job for job in self.retraining_jobs if job.gameweek == completed_gameweek),
                None
            )

            if existing_job and existing_job.status in ["running", "completed"]:
                return {"status": "already_scheduled", "gameweek": completed_gameweek}

            # Create retraining job
            scheduled_time = datetime.now() + timedelta(hours=self.retrain_delay_hours)

            job = RetrainingJob(
                gameweek=completed_gameweek,
                scheduled_time=scheduled_time,
                status="pending"
            )

            self.retraining_jobs.append(job)

            # Schedule the actual retraining
            schedule.every().minute.do(
                self._execute_retraining_if_ready,
                completed_gameweek
            ).tag(f"retrain_gw{completed_gameweek}")

            self.logger.info(f"📅 Retraining scheduled for GW{completed_gameweek} at {scheduled_time}")

            return {
                "status": "scheduled",
                "gameweek": completed_gameweek,
                "scheduled_time": scheduled_time.isoformat(),
                "delay_hours": self.retrain_delay_hours
            }

        except Exception as e:
            error_msg = f"Failed to schedule retraining for GW{completed_gameweek}: {e}"
            self.logger.error(error_msg)
            return {"status": "failed", "error": error_msg}

    def _execute_retraining_if_ready(self, gameweek: int):
        """Execute retraining if the scheduled time has arrived"""

        try:
            # Find the job
            job = next(
                (job for job in self.retraining_jobs if job.gameweek == gameweek),
                None
            )

            if not job or job.status != "pending":
                return

            # Check if it's time to retrain
            if datetime.now() < job.scheduled_time:
                return

            # Update job status
            job.status = "running"
            self.logger.info(f"🚀 Starting aggressive retraining for GW{gameweek}")

            # Execute the actual retraining
            retrain_result = self.max_accuracy_system.immediate_post_gameweek_retrain(gameweek)

            # Update job with results
            if retrain_result.get("status") == "success":
                job.status = "completed"
                job.result = retrain_result
                self.logger.info(f"✅ Retraining completed successfully for GW{gameweek}")

                # Trigger accuracy evaluation
                self._evaluate_retraining_success(gameweek, retrain_result)

            else:
                job.status = "failed"
                job.error = retrain_result.get("error", "Unknown error")
                self.logger.error(f"❌ Retraining failed for GW{gameweek}: {job.error}")

                # Attempt retry if under limit
                self._attempt_retrain_retry(gameweek, job)

            # Clear the scheduled job
            schedule.clear(f"retrain_gw{gameweek}")

            # Save results
            self._save_retraining_results(job)

        except Exception as e:
            self.logger.error(f"Retraining execution failed for GW{gameweek}: {e}")

            # Update job status
            if job:
                job.status = "failed"
                job.error = str(e)

    def _evaluate_retraining_success(self, gameweek: int, retrain_result: Dict[str, Any]):
        """Evaluate the success of retraining and take corrective action if needed"""

        try:
            # Check if new accuracy targets are met
            performance = retrain_result.get("performance_metrics", {})
            accuracy = performance.get("accuracy", 0)

            target_accuracy = getattr(self.max_accuracy_system, 'target_accuracy', 70)
            aggressive_threshold = getattr(self.max_accuracy_system, 'aggressive_threshold', 60)

            if accuracy < target_accuracy:
                self.logger.warning(f"⚠️  GW{gameweek} retraining did not meet accuracy target: {accuracy:.1f}% < {target_accuracy}%")

                # Schedule emergency retraining if accuracy is critically low
                if accuracy < aggressive_threshold:
                    self._schedule_emergency_retraining(gameweek)

            else:
                self.logger.info(f"🎉 GW{gameweek} retraining successful: {accuracy:.1f}% accuracy achieved")

            # Log to accuracy tracking system if available
            if self.accuracy_tracker and hasattr(self.accuracy_tracker, 'collect_gameweek_results'):
                self.accuracy_tracker.collect_gameweek_results(gameweek)

        except Exception as e:
            self.logger.error(f"Retraining evaluation failed for GW{gameweek}: {e}")

    def _attempt_retrain_retry(self, gameweek: int, failed_job: RetrainingJob):
        """Attempt to retry failed retraining"""

        # Count previous attempts
        retry_count = len([j for j in self.retraining_jobs
                          if j.gameweek == gameweek and j.status == "failed"])

        if retry_count < self.max_retrain_attempts:
            self.logger.info(f"🔄 Attempting retry {retry_count + 1} for GW{gameweek} retraining")

            # Schedule retry with increased delay
            retry_delay = timedelta(hours=1 * (retry_count + 1))
            retry_time = datetime.now() + retry_delay

            retry_job = RetrainingJob(
                gameweek=gameweek,
                scheduled_time=retry_time,
                status="pending"
            )

            self.retraining_jobs.append(retry_job)

            schedule.every().minute.do(
                self._execute_retraining_if_ready,
                gameweek
            ).tag(f"retrain_gw{gameweek}_retry{retry_count + 1}")

        else:
            self.logger.error(f"❌ Maximum retry attempts exceeded for GW{gameweek}")

    def _schedule_emergency_retraining(self, gameweek: int):
        """Schedule emergency retraining for critically low accuracy"""

        self.logger.warning(f"🚨 Scheduling emergency retraining for GW{gameweek}")

        # Create emergency job with immediate execution
        emergency_job = RetrainingJob(
            gameweek=gameweek,
            scheduled_time=datetime.now() + timedelta(minutes=5),
            status="pending"
        )

        self.retraining_jobs.append(emergency_job)

        # Schedule immediate emergency retraining
        schedule.every().minute.do(
            self._execute_emergency_retraining,
            gameweek
        ).tag(f"emergency_retrain_gw{gameweek}")

    def _execute_emergency_retraining(self, gameweek: int):
        """Execute emergency retraining with maximum intensity"""

        try:
            # Force emergency retrain mode
            original_threshold = getattr(self.max_accuracy_system, 'aggressive_threshold', 60)
            if hasattr(self.max_accuracy_system, 'aggressive_threshold'):
                self.max_accuracy_system.aggressive_threshold = 100  # Force emergency mode

            # Execute retraining
            retrain_result = self.max_accuracy_system.immediate_post_gameweek_retrain(gameweek)

            # Restore original threshold
            if hasattr(self.max_accuracy_system, 'aggressive_threshold'):
                self.max_accuracy_system.aggressive_threshold = original_threshold

            self.logger.info(f"🆘 Emergency retraining completed for GW{gameweek}")

            # Clear emergency job
            schedule.clear(f"emergency_retrain_gw{gameweek}")

        except Exception as e:
            self.logger.error(f"Emergency retraining failed for GW{gameweek}: {e}")

    def _daily_health_check(self) -> Dict[str, Any]:
        """Perform daily health check of the retraining system"""

        try:
            health_status = {
                "timestamp": datetime.now().isoformat(),
                "scheduler_running": self.is_running,
                "current_gameweek": self.current_gameweek,
                "last_processed_gameweek": self.last_processed_gameweek,
                "pending_jobs": len([j for j in self.retraining_jobs if j.status == "pending"]),
                "failed_jobs": len([j for j in self.retraining_jobs if j.status == "failed"]),
                "completed_jobs": len([j for j in self.retraining_jobs if j.status == "completed"])
            }

            # Get accuracy dashboard
            accuracy_dashboard = {}
            if hasattr(self.max_accuracy_system, 'get_accuracy_dashboard'):
                accuracy_dashboard = self.max_accuracy_system.get_accuracy_dashboard()
            health_status["accuracy_status"] = accuracy_dashboard

            # Check for issues
            issues = []
            if health_status["failed_jobs"] > 2:
                issues.append("Multiple retraining failures detected")

            if health_status["pending_jobs"] > 3:
                issues.append("Too many pending retraining jobs")

            recent_accuracy = accuracy_dashboard.get("current_performance", {}).get("accuracy", 0)
            if recent_accuracy < 50:
                issues.append("Critical accuracy issue detected")

            health_status["issues"] = issues
            health_status["health_score"] = self._calculate_health_score(health_status)

            # Save health report
            health_file = f"logs/retraining/daily_health_{datetime.now().strftime('%Y%m%d')}.json"
            with open(health_file, 'w') as f:
                json.dump(health_status, f, indent=2)

            if issues:
                self.logger.warning(f"🏥 Daily health check found issues: {issues}")
            else:
                self.logger.info(f"✅ Daily health check passed - Health score: {health_status['health_score']}")

            return health_status

        except Exception as e:
            error_msg = f"Daily health check failed: {e}"
            self.logger.error(error_msg)
            return {"error": error_msg}

    def _weekly_performance_report(self) -> Dict[str, Any]:
        """Generate weekly performance report"""

        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "week_ending": datetime.now().strftime('%Y-%m-%d'),
                "gameweeks_processed": []
            }

            # Get recent gameweek results
            recent_jobs = [j for j in self.retraining_jobs if j.status == "completed"][-7:]  # Last 7 jobs

            for job in recent_jobs:
                if job.result:
                    gw_summary = {
                        "gameweek": job.gameweek,
                        "retrain_intensity": job.result.get("retrain_intensity", "unknown"),
                        "accuracy": job.result.get("performance_metrics", {}).get("accuracy", 0),
                        "captain_success": job.result.get("performance_metrics", {}).get("captain_success_rate", 0)
                    }
                    report["gameweeks_processed"].append(gw_summary)

            # Calculate weekly averages
            if report["gameweeks_processed"]:
                accuracies = [gw["accuracy"] for gw in report["gameweeks_processed"]]
                captain_rates = [gw["captain_success"] for gw in report["gameweeks_processed"]]

                report["weekly_averages"] = {
                    "accuracy": sum(accuracies) / len(accuracies),
                    "captain_success": sum(captain_rates) / len(captain_rates),
                    "gameweeks_count": len(accuracies)
                }

            # Performance trend
            if len(accuracies) >= 2:
                trend = accuracies[-1] - accuracies[0]
                report["accuracy_trend"] = "improving" if trend > 2 else "declining" if trend < -2 else "stable"

            # Save weekly report
            report_file = f"reports/weekly_performance_{datetime.now().strftime('%Y%m%d')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)

            self.logger.info(f"📊 Weekly performance report generated: {report.get('weekly_averages', {})}")

            return report

        except Exception as e:
            error_msg = f"Weekly performance report failed: {e}"
            self.logger.error(error_msg)
            return {"error": error_msg}

    def _calculate_health_score(self, health_status: Dict[str, Any]) -> float:
        """Calculate overall system health score (0-100)"""

        score = 100.0

        # Deduct for failed jobs
        failed_jobs = health_status.get("failed_jobs", 0)
        score -= failed_jobs * 10

        # Deduct for accuracy issues
        accuracy = health_status.get("accuracy_status", {}).get("current_performance", {}).get("accuracy", 0)
        if accuracy < 50:
            score -= 30
        elif accuracy < 60:
            score -= 20
        elif accuracy < 70:
            score -= 10

        # Deduct for system issues
        issues = health_status.get("issues", [])
        score -= len(issues) * 5

        return max(0, min(100, score))

    def _save_scheduler_state(self):
        """Save scheduler state to disk"""

        state = {
            "current_gameweek": self.current_gameweek,
            "last_processed_gameweek": self.last_processed_gameweek,
            "is_running": self.is_running,
            "jobs_count": len(self.retraining_jobs),
            "timestamp": datetime.now().isoformat()
        }

        state_file = "data/automated_retraining/scheduler_state.json"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def _load_scheduler_state(self):
        """Load scheduler state from disk"""

        state_file = "data/automated_retraining/scheduler_state.json"
        if Path(state_file).exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)

                self.current_gameweek = state.get("current_gameweek", 1)
                self.last_processed_gameweek = state.get("last_processed_gameweek", 0)

                self.logger.info(f"📚 Loaded scheduler state: GW{self.current_gameweek}, last processed: GW{self.last_processed_gameweek}")

            except Exception as e:
                self.logger.warning(f"Failed to load scheduler state: {e}")

    def _initialize_current_gameweek(self):
        """Initialize current gameweek from FPL API"""
        try:
            bootstrap = self.fpl.bootstrap_static()
            events = bootstrap.get("events", [])

            for event in events:
                if event.get("is_current", False):
                    self.current_gameweek = event.get("id", 1)
                    break

            self.logger.info(f"Initialized current gameweek: GW{self.current_gameweek}")

        except Exception as e:
            self.logger.warning(f"Failed to initialize current gameweek: {e}")
            self.current_gameweek = 1

    def _save_retraining_results(self, job: RetrainingJob):
        """Save detailed retraining results"""

        result_data = {
            "gameweek": job.gameweek,
            "scheduled_time": job.scheduled_time.isoformat(),
            "status": job.status,
            "result": job.result,
            "error": job.error,
            "timestamp": datetime.now().isoformat()
        }

        result_file = f"data/automated_retraining/gw{job.gameweek}_retrain_result.json"
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2)

    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get current scheduler status for monitoring"""

        return {
            "is_running": self.is_running,
            "current_gameweek": self.current_gameweek,
            "last_processed_gameweek": self.last_processed_gameweek,
            "pending_jobs": [
                {
                    "gameweek": job.gameweek,
                    "scheduled_time": job.scheduled_time.isoformat(),
                    "status": job.status
                }
                for job in self.retraining_jobs if job.status == "pending"
            ],
            "recent_completions": [
                {
                    "gameweek": job.gameweek,
                    "status": job.status,
                    "accuracy": job.result.get("performance_metrics", {}).get("accuracy", 0) if job.result else 0
                }
                for job in self.retraining_jobs[-5:] if job.status in ["completed", "failed"]
            ],
            "next_check": datetime.now() + timedelta(minutes=self.check_interval_minutes),
            "health_score": self._calculate_health_score({"failed_jobs": 0, "issues": [], "accuracy_status": {}})
        }

    def force_manual_retrain(self, gameweek: int) -> Dict[str, Any]:
        """Manually trigger retraining for a specific gameweek"""

        self.logger.info(f"🔧 Manual retraining triggered for GW{gameweek}")

        try:
            result = self.max_accuracy_system.immediate_post_gameweek_retrain(gameweek)

            # Create job record
            manual_job = RetrainingJob(
                gameweek=gameweek,
                scheduled_time=datetime.now(),
                status="completed" if result.get("status") == "success" else "failed",
                result=result if result.get("status") == "success" else None,
                error=result.get("error") if result.get("status") != "success" else None
            )

            self.retraining_jobs.append(manual_job)
            self._save_retraining_results(manual_job)

            return result

        except Exception as e:
            error_msg = f"Manual retraining failed for GW{gameweek}: {e}"
            self.logger.error(error_msg)
            return {"status": "failed", "error": error_msg}


# Global scheduler instance
_scheduler_instance = None

def get_retraining_scheduler() -> AutomatedRetrainingScheduler:
    """Get or create the global retraining scheduler instance"""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = AutomatedRetrainingScheduler()
    return _scheduler_instance

def start_automated_retraining():
    """Start the automated retraining system"""
    scheduler = get_retraining_scheduler()
    scheduler.start_automated_monitoring()
    return scheduler

def stop_automated_retraining():
    """Stop the automated retraining system"""
    scheduler = get_retraining_scheduler()
    scheduler.stop_automated_monitoring()

def manual_retrain_gameweek(gameweek: int) -> Dict[str, Any]:
    """Manually trigger retraining for a specific gameweek"""
    scheduler = get_retraining_scheduler()
    return scheduler.force_manual_retrain(gameweek)