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

        # Configuration
        self.check_interval = 300  # 5 minutes
        self.gameweek_completion_delay = 120  # 2 hours after gameweek ends
        self.max_retrain_attempts = 3

        # File paths
        self.status_file = Path("data/scheduler/retraining_status.json")
        self.history_file = Path("data/scheduler/retraining_history.jsonl")
        self.status_file.parent.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

        # Load previous state
        self._load_state()

    def _setup_logging(self):
        """Setup dedicated logging for scheduler"""
        log_file = Path("logs/automated_retraining.log")
        log_file.parent.mkdir(exist_ok=True)

        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def _load_state(self):
        """Load previous scheduler state"""
        try:
            if self.status_file.exists():
                with open(self.status_file, 'r') as f:
                    state = json.load(f)
                    self.last_processed_gameweek = state.get('last_processed_gameweek', 0)
                    self.logger.info(f"Loaded state: last processed GW {self.last_processed_gameweek}")

            # Load history
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    for line in f:
                        try:
                            event_data = json.loads(line)
                            # Convert string dates back to datetime
                            event_data['trigger_time'] = datetime.fromisoformat(event_data['trigger_time'])
                            if event_data.get('completion_time'):
                                event_data['completion_time'] = datetime.fromisoformat(event_data['completion_time'])
                            event_data['status'] = RetrainingStatus(event_data['status'])

                            event = RetrainingEvent(**event_data)
                            self.retraining_history.append(event)
                        except Exception as e:
                            self.logger.warning(f"Could not load history event: {e}")

        except Exception as e:
            self.logger.warning(f"Could not load previous state: {e}")

    def _save_state(self):
        """Save current scheduler state"""
        try:
            state = {
                'last_processed_gameweek': self.last_processed_gameweek,
                'current_status': self.current_status.value,
                'last_updated': datetime.now().isoformat()
            }

            with open(self.status_file, 'w') as f:
                json.dump(state, f, indent=2)

        except Exception as e:
            self.logger.error(f"Could not save state: {e}")

    def _save_retraining_event(self, event: RetrainingEvent):
        """Save retraining event to history"""
        try:
            with open(self.history_file, 'a') as f:
                event_dict = asdict(event)
                # Convert datetime objects to strings
                event_dict['trigger_time'] = event.trigger_time.isoformat()
                if event.completion_time:
                    event_dict['completion_time'] = event.completion_time.isoformat()
                event_dict['status'] = event.status.value

                f.write(json.dumps(event_dict) + '\n')

        except Exception as e:
            self.logger.error(f"Could not save retraining event: {e}")

    def check_gameweek_completion(self) -> Dict[str, Any]:
        """Check if any gameweeks have completed and need retraining"""
        try:
            # Get current FPL status
            bs = self.fpl.bootstrap_static()
            if not bs:
                return {'status': 'error', 'message': 'Could not fetch FPL data'}

            events = bs.get('events', [])
            finished_gameweeks = [e['id'] for e in events if e.get('finished', False)]

            if not finished_gameweeks:
                return {'status': 'no_completed_gameweeks', 'finished': []}

            latest_finished = max(finished_gameweeks)

            # Check if we need to process this gameweek
            if latest_finished > self.last_processed_gameweek:
                # Check if enough time has passed since gameweek completion
                latest_event = next((e for e in events if e['id'] == latest_finished), None)
                if latest_event:
                    deadline_time = datetime.fromisoformat(latest_event['deadline_time'].replace('Z', '+00:00'))
                    time_since_deadline = datetime.now() - deadline_time.replace(tzinfo=None)

                    if time_since_deadline.total_seconds() >= self.gameweek_completion_delay * 60:
                        return {
                            'status': 'retraining_needed',
                            'gameweek': latest_finished,
                            'trigger_reason': f'GW {latest_finished} completed {time_since_deadline} ago',
                            'finished_gameweeks': finished_gameweeks
                        }
                    else:
                        return {
                            'status': 'waiting_for_delay',
                            'gameweek': latest_finished,
                            'time_remaining': self.gameweek_completion_delay * 60 - time_since_deadline.total_seconds(),
                            'finished_gameweeks': finished_gameweeks
                        }

            return {
                'status': 'up_to_date',
                'last_processed': self.last_processed_gameweek,
                'latest_finished': latest_finished,
                'finished_gameweeks': finished_gameweeks
            }

        except Exception as e:
            self.logger.error(f"Error checking gameweek completion: {e}")
            return {'status': 'error', 'message': str(e)}

    def trigger_retraining(self, gameweek: int, reason: str) -> RetrainingEvent:
        """Trigger retraining for specified gameweek"""
        self.logger.info(f"Triggering retraining for GW {gameweek}: {reason}")

        event = RetrainingEvent(
            gameweek=gameweek,
            trigger_time=datetime.now(),
            completion_time=None,
            status=RetrainingStatus.TRIGGERED,
            accuracy_before=None,
            accuracy_after=None,
            trigger_reason=reason
        )

        try:
            self.current_status = RetrainingStatus.TRAINING

            # Get accuracy before retraining
            if self.max_accuracy_system.accuracy_history:
                event.accuracy_before = self.max_accuracy_system.accuracy_history[-1].overall_accuracy

            # Send notification
            if self.notification_callback:
                self.notification_callback(f"🔄 Starting automated retraining for GW {gameweek}")

            # Perform retraining
            retrain_result = self.max_accuracy_system.automatic_retrain(gameweek)

            if retrain_result['status'] == 'success':
                event.status = RetrainingStatus.COMPLETED
                event.completion_time = datetime.now()
                event.model_improvements = {
                    'ensemble_weights': retrain_result['ensemble_weights'],
                    'models_trained': retrain_result['models_trained']
                }

                # Get accuracy after retraining (if available)
                if self.max_accuracy_system.accuracy_history:
                    event.accuracy_after = self.max_accuracy_system.accuracy_history[-1].overall_accuracy

                # Update processed gameweek
                self.last_processed_gameweek = gameweek
                self.current_status = RetrainingStatus.COMPLETED

                self.logger.info(f"Retraining completed successfully for GW {gameweek}")

                if self.notification_callback:
                    accuracy_msg = ""
                    if event.accuracy_before and event.accuracy_after:
                        change = event.accuracy_after - event.accuracy_before
                        accuracy_msg = f" (Accuracy: {event.accuracy_before:.1%} → {event.accuracy_after:.1%}, {change:+.1%})"

                    self.notification_callback(f"✅ Retraining completed for GW {gameweek}{accuracy_msg}")

            else:
                event.status = RetrainingStatus.ERROR
                event.error_message = retrain_result.get('error', 'Unknown error')
                event.completion_time = datetime.now()
                self.current_status = RetrainingStatus.ERROR

                self.logger.error(f"Retraining failed for GW {gameweek}: {event.error_message}")

                if self.notification_callback:
                    self.notification_callback(f"❌ Retraining failed for GW {gameweek}: {event.error_message}")

        except Exception as e:
            event.status = RetrainingStatus.ERROR
            event.error_message = str(e)
            event.completion_time = datetime.now()
            self.current_status = RetrainingStatus.ERROR

            self.logger.error(f"Exception during retraining for GW {gameweek}: {e}")

            if self.notification_callback:
                self.notification_callback(f"💥 Retraining crashed for GW {gameweek}: {str(e)}")

        # Save event and state
        self.retraining_history.append(event)
        self._save_retraining_event(event)
        self._save_state()

        return event

    def monitoring_loop(self):
        """Main monitoring loop"""
        self.logger.info("Starting automated retraining monitoring loop")
        self.is_running = True
        self.current_status = RetrainingStatus.MONITORING

        while self.is_running:
            try:
                # Check for gameweek completion
                check_result = self.check_gameweek_completion()

                if check_result['status'] == 'retraining_needed':
                    gameweek = check_result['gameweek']
                    reason = check_result['trigger_reason']

                    # Trigger retraining
                    self.trigger_retraining(gameweek, reason)

                elif check_result['status'] == 'error':
                    self.logger.warning(f"Gameweek check failed: {check_result['message']}")

                # Wait before next check
                time.sleep(self.check_interval)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.check_interval)

        self.logger.info("Monitoring loop stopped")
        self.current_status = RetrainingStatus.IDLE

    def start_monitoring(self):
        """Start monitoring in background thread"""
        if self.is_running:
            self.logger.warning("Monitoring is already running")
            return

        self.logger.info("Starting automated retraining monitoring")

        # Start monitoring in background thread
        monitor_thread = threading.Thread(target=self.monitoring_loop, daemon=True)
        monitor_thread.start()

        if self.notification_callback:
            self.notification_callback("🤖 Automated retraining monitoring started")

    def stop_monitoring(self):
        """Stop monitoring"""
        if not self.is_running:
            self.logger.warning("Monitoring is not running")
            return

        self.logger.info("Stopping automated retraining monitoring")
        self.is_running = False
        self.current_status = RetrainingStatus.IDLE

        if self.notification_callback:
            self.notification_callback("⏹️ Automated retraining monitoring stopped")

    def force_retrain(self, gameweek: Optional[int] = None) -> RetrainingEvent:
        """Force retraining for current or specified gameweek"""
        if gameweek is None:
            # Get current gameweek
            try:
                bs = self.fpl.bootstrap_static()
                events = bs.get('events', [])
                current_gw = next((e['id'] for e in events if e.get('is_current')), 1)
                gameweek = current_gw
            except Exception:
                gameweek = 1

        return self.trigger_retraining(gameweek, "Manual force retrain")

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive scheduler status"""
        recent_events = self.retraining_history[-5:] if self.retraining_history else []

        return {
            'is_running': self.is_running,
            'current_status': self.current_status.value,
            'last_processed_gameweek': self.last_processed_gameweek,
            'total_retraining_events': len(self.retraining_history),
            'recent_events': [
                {
                    'gameweek': e.gameweek,
                    'trigger_time': e.trigger_time.isoformat(),
                    'status': e.status.value,
                    'trigger_reason': e.trigger_reason,
                    'accuracy_improvement': (e.accuracy_after - e.accuracy_before)
                                          if e.accuracy_before and e.accuracy_after else None
                }
                for e in recent_events
            ],
            'configuration': {
                'check_interval': self.check_interval,
                'gameweek_completion_delay': self.gameweek_completion_delay,
                'max_retrain_attempts': self.max_retrain_attempts
            },
            'last_check': datetime.now().isoformat()
        }

    def get_retraining_history(self, limit: int = 20) -> list[Dict[str, Any]]:
        """Get detailed retraining history"""
        recent_events = self.retraining_history[-limit:] if self.retraining_history else []

        return [
            {
                'gameweek': e.gameweek,
                'trigger_time': e.trigger_time.isoformat(),
                'completion_time': e.completion_time.isoformat() if e.completion_time else None,
                'status': e.status.value,
                'trigger_reason': e.trigger_reason,
                'accuracy_before': e.accuracy_before,
                'accuracy_after': e.accuracy_after,
                'accuracy_improvement': (e.accuracy_after - e.accuracy_before)
                                      if e.accuracy_before and e.accuracy_after else None,
                'error_message': e.error_message,
                'model_improvements': e.model_improvements
            }
            for e in reversed(recent_events)
        ]


# Global scheduler instance
_scheduler_instance: Optional[AutomatedRetrainingScheduler] = None


def get_retraining_scheduler() -> AutomatedRetrainingScheduler:
    """Get global scheduler instance"""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = AutomatedRetrainingScheduler()
    return _scheduler_instance


def start_automated_retraining(notification_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """Start automated retraining system"""
    try:
        scheduler = get_retraining_scheduler()

        if notification_callback:
            scheduler.notification_callback = notification_callback

        scheduler.start_monitoring()

        return {
            'status': 'success',
            'message': 'Automated retraining started',
            'scheduler_status': scheduler.get_status()
        }

    except Exception as e:
        logging.error(f"Failed to start automated retraining: {e}")
        return {
            'status': 'error',
            'message': str(e)
        }


def stop_automated_retraining() -> Dict[str, Any]:
    """Stop automated retraining system"""
    try:
        scheduler = get_retraining_scheduler()
        scheduler.stop_monitoring()

        return {
            'status': 'success',
            'message': 'Automated retraining stopped'
        }

    except Exception as e:
        logging.error(f"Failed to stop automated retraining: {e}")
        return {
            'status': 'error',
            'message': str(e)
        }