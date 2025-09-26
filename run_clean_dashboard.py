#!/usr/bin/env python3
"""
Enhanced FPL AI Dashboard Startup Script
Professional Fantasy Premier League prediction system with 80%+ accuracy target

USAGE: python run_clean_dashboard.py [--port PORT] [--test]

Features:
- Enhanced ML prediction model with improved scaling
- Football field team visualization
- Historical accuracy tracking with real FPL data
- Model retraining with actual results
- Prediction validation and monitoring
"""

import subprocess
import sys
import os
import argparse
import webbrowser
import time
from pathlib import Path

def check_dependencies():
    """Validate required packages are installed"""
    required = ['fastapi', 'uvicorn', 'pandas', 'numpy', 'requests']
    missing = []

    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"❌ Missing required packages: {', '.join(missing)}")
        print(f"💡 Install with: pip install {' '.join(missing)}")
        return False
    return True

def validate_project_structure():
    """Check if required files exist"""
    required_files = [
        'src/web_ui_clean.py',
        'src/ml_pipeline.py',
        'src/fpl_client.py',
        'data/accuracy_tracking'
    ]

    missing = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing.append(file_path)

    if missing:
        print(f"❌ Missing required files: {', '.join(missing)}")
        return False
    return True

def run_tests():
    """Run basic system tests"""
    print("🧪 Running system validation tests...")

    try:
        # Test FPL client
        from src.fpl_client import FPLClient
        fpl = FPLClient()
        bootstrap = fpl.bootstrap_static()
        if not bootstrap or 'elements' not in bootstrap:
            print("❌ FPL API connection failed")
            return False
        print("✅ FPL API connection working")

        # Test ML pipeline
        from src.ml_pipeline import MLPipeline
        pipeline = MLPipeline()
        predictions = pipeline.predict_current()
        if predictions.empty:
            print("❌ Prediction system failed")
            return False
        print(f"✅ Predictions working - {len(predictions)} players")

        return True

    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Start FPL AI Dashboard')
    parser.add_argument('--port', type=int, default=8001, help='Port to run on (default: 8001)')
    parser.add_argument('--test', action='store_true', help='Run system tests before starting')
    parser.add_argument('--no-browser', action='store_true', help="Don't auto-open browser")
    args = parser.parse_args()

    print("🚀 FPL AI Dashboard - Enhanced Prediction System")
    print("=" * 70)
    print("📈 Target Accuracy: 80%+ (up from previous 55%)")
    print("🔧 Recent Fixes: Prediction scaling, bonus points, validation")
    print("=" * 70)

    # Ensure we're in the right directory
    fpl_dir = Path(__file__).parent
    os.chdir(fpl_dir)

    # Validate environment
    if not check_dependencies():
        return 1

    if not validate_project_structure():
        return 1

    # Run tests if requested
    if args.test:
        if not run_tests():
            return 1

    print(f"📁 Working directory: {fpl_dir}")
    print(f"🔗 Dashboard URL: http://localhost:{args.port}")
    print(f"📊 New Features: Prediction validation, enhanced bonus scoring, realistic caps")
    print("=" * 70)
    print("🎯 Quick Start Guide:")
    print("  1. Click 'Generate Team' to create predictions")
    print("  2. Click 'Verify Points' to check accuracy against real FPL data")
    print("  3. Click 'Update Accuracy' to fetch actual points from FPL API")
    print("  4. Use 'Improve Model' to retrain based on actual results")
    print("=" * 70)

    # Start the server
    try:
        if not args.no_browser:
            # Auto-open browser after a delay
            def open_browser():
                time.sleep(2)
                webbrowser.open(f'http://localhost:{args.port}')

            import threading
            threading.Thread(target=open_browser, daemon=True).start()

        print("🌟 Starting server...")
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "src.web_ui_clean:app",
            "--host", "0.0.0.0",
            "--port", str(args.port),
            "--reload"
        ], check=True)

    except KeyboardInterrupt:
        print("\n👋 FPL Dashboard stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting dashboard: {e}")
        print("💡 Try: pip install fastapi uvicorn")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())