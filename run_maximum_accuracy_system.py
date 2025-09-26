#!/usr/bin/env python3
"""
Maximum Accuracy FPL System Launcher
Complete production-ready system for 70-75%+ FPL prediction accuracy

Usage:
    python run_maximum_accuracy_system.py [--mode MODE] [--strategy STRATEGY] [--port PORT]

Modes:
    - serve: Start web server with dashboard (default)
    - predict: Generate predictions only
    - retrain: Force model retraining
    - monitor: Start monitoring only
    - full: Start all components

Strategies:
    - conservative: Lower risk, steady gains (55-60% target)
    - balanced: Moderate risk/reward (60-65% target)
    - aggressive: Higher risk, maximum returns (65-70% target)
    - maximum: All-out attack on accuracy (70-75%+ target)
"""

import asyncio
import logging
import argparse
import sys
import signal
import threading
from pathlib import Path
from datetime import datetime
import uvicorn
from fastapi import FastAPI

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.maximum_accuracy_system import MaximumAccuracySystem, AccuracyTarget, ModelConfig
from src.automated_retraining_scheduler_v2 import AutomatedRetrainingScheduler, start_automated_retraining
from src.aggressive_accuracy_optimizer import AggressiveAccuracyOptimizer, OptimizationStrategy
from src.accuracy_monitoring_dashboard import AccuracyMonitoringDashboard
from src.maximum_accuracy_endpoints import app as max_accuracy_app


class MaximumAccuracySystemRunner:
    """Main system runner for maximum accuracy FPL system"""

    def __init__(self, strategy: str = "maximum", mode: str = "serve"):
        self.strategy = self._parse_strategy(strategy)
        self.mode = mode

        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.logger.info("🚀 Initializing Maximum Accuracy FPL System...")
        self._initialize_components()

        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.running = False

    def _setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # Root logger configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "maximum_accuracy_system.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )

        # Reduce noise from other libraries
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)

    def _parse_strategy(self, strategy: str) -> OptimizationStrategy:
        """Parse strategy string to enum"""
        strategy_map = {
            "conservative": OptimizationStrategy.CONSERVATIVE,
            "balanced": OptimizationStrategy.BALANCED,
            "aggressive": OptimizationStrategy.AGGRESSIVE,
            "maximum": OptimizationStrategy.MAXIMUM
        }

        if strategy not in strategy_map:
            raise ValueError(f"Invalid strategy: {strategy}. Must be one of: {list(strategy_map.keys())}")

        return strategy_map[strategy]

    def _initialize_components(self):
        """Initialize all system components"""
        try:
            # Map strategy to accuracy target
            target_map = {
                OptimizationStrategy.CONSERVATIVE: AccuracyTarget.CONSERVATIVE,
                OptimizationStrategy.BALANCED: AccuracyTarget.BALANCED,
                OptimizationStrategy.AGGRESSIVE: AccuracyTarget.AGGRESSIVE,
                OptimizationStrategy.MAXIMUM: AccuracyTarget.MAXIMUM
            }

            # Initialize model configuration
            model_config = ModelConfig(target_accuracy=target_map[self.strategy])

            # Initialize core components
            self.max_accuracy_system = MaximumAccuracySystem(model_config)
            self.accuracy_optimizer = AggressiveAccuracyOptimizer(self.strategy)
            self.accuracy_dashboard = AccuracyMonitoringDashboard()
            self.automated_scheduler = AutomatedRetrainingScheduler(
                max_accuracy_system=self.max_accuracy_system,
                notification_callback=self._notification_handler
            )

            self.logger.info(f"✅ System initialized with {self.strategy.value} strategy")
            self.logger.info(f"📊 Target accuracy: {model_config.target_accuracy.value}")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize system: {e}")
            raise

    def _notification_handler(self, message: str):
        """Handle system notifications"""
        self.logger.info(f"📢 NOTIFICATION: {message}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"🛑 Received signal {signum}, shutting down gracefully...")
        self.running = False

    def run_predict_mode(self):
        """Run prediction generation only"""
        try:
            self.logger.info("🔮 Generating maximum accuracy predictions...")

            # Generate predictions
            predictions_df = self.max_accuracy_system.generate_maximum_accuracy_predictions()

            # Apply optimization
            optimized_predictions = self.accuracy_optimizer.optimize_predictions(predictions_df)

            # Save results
            output_file = Path("data/predictions/maximum_accuracy_predictions.csv")
            output_file.parent.mkdir(parents=True, exist_ok=True)
            optimized_predictions.to_csv(output_file, index=False)

            # Display top predictions
            top_predictions = optimized_predictions.nlargest(20, 'predicted_points')

            print("\n" + "="*80)
            print("🎯 TOP 20 MAXIMUM ACCURACY PREDICTIONS")
            print("="*80)
            print(f"{'Player':<25} {'Pos':<5} {'Team':<8} {'Price':<7} {'Pred Pts':<9} {'Strategy':<12}")
            print("-"*80)

            for _, player in top_predictions.iterrows():
                captain_flag = "👑 " if player.get('captain_candidate', False) else ""
                diff_flag = "🔥 " if player.get('differential_candidate', False) else ""

                print(f"{captain_flag}{diff_flag}{player['name'][:23]:<25} "
                      f"{player['position']:<5} "
                      f"{str(player.get('team_name', 'Unknown'))[:8]:<8} "
                      f"£{player['price']:<6.1f} "
                      f"{player['predicted_points']:<8.1f} "
                      f"{self.strategy.value:<12}")

            print("-"*80)
            print(f"💾 Predictions saved to: {output_file}")
            print(f"📊 Total players: {len(optimized_predictions)}")
            print(f"🎯 Strategy: {self.strategy.value}")
            print(f"👑 Captain candidates: {optimized_predictions.get('captain_candidate', False).sum()}")
            print(f"🔥 Differential picks: {optimized_predictions.get('differential_candidate', False).sum()}")

            # Capture accuracy snapshot
            snapshot = self.accuracy_dashboard.capture_accuracy_snapshot(optimized_predictions)
            print(f"📈 System confidence: {snapshot.model_confidence:.1%}")

        except Exception as e:
            self.logger.error(f"❌ Error in predict mode: {e}")
            raise

    def run_retrain_mode(self):
        """Run model retraining"""
        try:
            self.logger.info("🔄 Starting forced model retraining...")

            # Get current gameweek
            from src.fpl_client import FPLClient
            fpl = FPLClient()
            bs = fpl.bootstrap_static()
            events = bs.get('events', [])
            current_gw = next((e['id'] for e in events if e.get('is_current')), 1)

            # Trigger retraining
            event = self.automated_scheduler.trigger_retraining(current_gw, "Manual force retrain")

            print("\n" + "="*60)
            print("🔄 RETRAINING RESULTS")
            print("="*60)
            print(f"Gameweek: {event.gameweek}")
            print(f"Status: {event.status.value}")
            print(f"Trigger reason: {event.trigger_reason}")

            if event.completion_time:
                duration = event.completion_time - event.trigger_time
                print(f"Duration: {duration.total_seconds():.1f} seconds")

            if event.accuracy_before and event.accuracy_after:
                improvement = event.accuracy_after - event.accuracy_before
                print(f"Accuracy before: {event.accuracy_before:.1%}")
                print(f"Accuracy after: {event.accuracy_after:.1%}")
                print(f"Improvement: {improvement:+.1%}")

            if event.error_message:
                print(f"Error: {event.error_message}")

            print("="*60)

        except Exception as e:
            self.logger.error(f"❌ Error in retrain mode: {e}")
            raise

    def run_monitor_mode(self):
        """Run monitoring only"""
        try:
            self.logger.info("📊 Starting accuracy monitoring...")

            # Start automated scheduler
            self.automated_scheduler.start_monitoring()

            self.running = True
            print("\n" + "="*60)
            print("📊 MAXIMUM ACCURACY MONITORING ACTIVE")
            print("="*60)
            print("Press Ctrl+C to stop monitoring")
            print("="*60)

            # Keep running until interrupted
            while self.running:
                import time
                time.sleep(60)  # Check every minute

                # Display status every 10 minutes
                if int(time.time()) % 600 == 0:
                    status = self.automated_scheduler.get_status()
                    self.logger.info(f"📊 Monitoring status: {status['current_status']} | "
                                   f"Events: {status['total_retraining_events']} | "
                                   f"Last GW: {status['last_processed_gameweek']}")

        except KeyboardInterrupt:
            self.logger.info("🛑 Monitoring stopped by user")
        except Exception as e:
            self.logger.error(f"❌ Error in monitor mode: {e}")
        finally:
            self.automated_scheduler.stop_monitoring()

    def run_serve_mode(self, port: int = 8000):
        """Run web server with dashboard"""
        try:
            self.logger.info(f"🌐 Starting web server on port {port}...")

            # Start automated monitoring in background
            if self.mode == "full":
                self.automated_scheduler.start_monitoring()
                self.logger.info("🤖 Automated monitoring started")

            print("\n" + "="*80)
            print("🌐 MAXIMUM ACCURACY FPL SYSTEM - WEB DASHBOARD")
            print("="*80)
            print(f"🎯 Strategy: {self.strategy.value}")
            print(f"📊 Target Accuracy: {self.max_accuracy_system.config.target_accuracy.value}")
            print(f"🌐 Dashboard URL: http://localhost:{port}")
            print(f"📊 API Docs: http://localhost:{port}/docs")
            print("="*80)
            print("Press Ctrl+C to stop the server")
            print("="*80)

            # Configure and run server
            config = uvicorn.Config(
                max_accuracy_app,
                host="0.0.0.0",
                port=port,
                log_level="info",
                access_log=False
            )

            server = uvicorn.Server(config)
            server.run()

        except KeyboardInterrupt:
            self.logger.info("🛑 Web server stopped by user")
        except Exception as e:
            self.logger.error(f"❌ Error in serve mode: {e}")
            raise
        finally:
            if hasattr(self, 'automated_scheduler'):
                self.automated_scheduler.stop_monitoring()

    def run_full_mode(self, port: int = 8000):
        """Run complete system with all components"""
        try:
            self.logger.info("🚀 Starting complete maximum accuracy system...")

            # Start all components
            self.automated_scheduler.start_monitoring()

            # Run web server
            self.run_serve_mode(port)

        except Exception as e:
            self.logger.error(f"❌ Error in full mode: {e}")
            raise

    def run(self, port: int = 8000):
        """Run the system in the specified mode"""
        try:
            if self.mode == "predict":
                self.run_predict_mode()
            elif self.mode == "retrain":
                self.run_retrain_mode()
            elif self.mode == "monitor":
                self.run_monitor_mode()
            elif self.mode == "serve":
                self.run_serve_mode(port)
            elif self.mode == "full":
                self.run_full_mode(port)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

        except Exception as e:
            self.logger.error(f"❌ System failed: {e}")
            sys.exit(1)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Maximum Accuracy FPL System - Target 70-75%+ prediction accuracy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--mode", "-m",
        choices=["serve", "predict", "retrain", "monitor", "full"],
        default="serve",
        help="System mode (default: serve)"
    )

    parser.add_argument(
        "--strategy", "-s",
        choices=["conservative", "balanced", "aggressive", "maximum"],
        default="maximum",
        help="Optimization strategy (default: maximum)"
    )

    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Web server port (default: 8000)"
    )

    parser.add_argument(
        "--version", "-v",
        action="version",
        version="Maximum Accuracy FPL System v1.0.0"
    )

    args = parser.parse_args()

    # Display banner
    print("\n" + "="*80)
    print("🎯 MAXIMUM ACCURACY FPL SYSTEM v1.0.0")
    print("🚀 Advanced ML-Powered FPL Predictions")
    print("🎯 Target: 70-75%+ Accuracy")
    print("="*80)

    try:
        # Initialize and run system
        runner = MaximumAccuracySystemRunner(strategy=args.strategy, mode=args.mode)
        runner.run(port=args.port)

    except KeyboardInterrupt:
        print("\n🛑 System stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ System failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()