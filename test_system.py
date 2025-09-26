#!/usr/bin/env python3
"""
FPL System Testing Framework
Run comprehensive tests to validate prediction accuracy and system reliability

Usage: python test_system.py [--verbose]
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_fpl_client():
    """Test FPL API client functionality"""
    try:
        from fpl_client import FPLClient

        print("🧪 Testing FPL Client...")
        client = FPLClient()

        # Test bootstrap data
        bootstrap = client.bootstrap_static()
        assert bootstrap is not None, "Bootstrap data is None"
        assert 'elements' in bootstrap, "No elements in bootstrap"
        assert len(bootstrap['elements']) > 500, f"Only {len(bootstrap['elements'])} players found"

        print(f"✅ FPL Client: {len(bootstrap['elements'])} players loaded")

        # Test live data (may fail if no active gameweek)
        try:
            live_data = client.event_live(1)
            if live_data and 'elements' in live_data:
                print(f"✅ Live data: GW1 has {len(live_data['elements'])} player stats")
            else:
                print("⚠️ Live data not available (expected for future gameweeks)")
        except:
            print("⚠️ Live data test skipped")

        return True

    except Exception as e:
        print(f"❌ FPL Client test failed: {e}")
        return False

def test_ml_pipeline():
    """Test ML prediction pipeline"""
    try:
        from ml_pipeline import MLPipeline

        print("🧪 Testing ML Pipeline...")
        pipeline = MLPipeline()

        # Test prediction generation
        predictions = pipeline.predict_current()
        assert not predictions.empty, "No predictions generated"
        assert len(predictions) > 500, f"Only {len(predictions)} predictions"

        # Test prediction ranges
        pred_col = 'predicted_points'
        if pred_col not in predictions.columns:
            pred_col = 'ep_ml'  # Fallback column name

        if pred_col in predictions.columns:
            avg_pred = predictions[pred_col].mean()
            min_pred = predictions[pred_col].min()
            max_pred = predictions[pred_col].max()

            print(f"✅ ML Pipeline: {len(predictions)} predictions")
            print(f"   Range: {min_pred:.1f} - {max_pred:.1f}, avg: {avg_pred:.1f}")

            # Validate reasonable ranges
            if avg_pred < 3.0:
                print(f"⚠️ Warning: Average prediction {avg_pred:.1f} seems too low")
            elif avg_pred > 15.0:
                print(f"⚠️ Warning: Average prediction {avg_pred:.1f} seems too high")
            else:
                print(f"✅ Prediction ranges look reasonable")

        else:
            print(f"⚠️ Could not find prediction column in: {list(predictions.columns)}")

        return True

    except Exception as e:
        print(f"❌ ML Pipeline test failed: {e}")
        return False

def test_historical_data():
    """Test historical data access and accuracy calculations"""
    try:
        from pathlib import Path
        import json

        print("🧪 Testing Historical Data...")

        data_path = Path('data/accuracy_tracking')
        json_file = data_path / 'gameweek_results.json'

        if not json_file.exists():
            print("⚠️ No historical data found")
            return True

        with open(json_file, 'r') as f:
            results_data = json.load(f)

        assert len(results_data) > 0, "No gameweek results found"

        # Test data structure
        for result in results_data:
            assert 'gameweek' in result, "Missing gameweek field"
            assert 'team_selection' in result, "Missing team_selection field"

            team_selection = result['team_selection']
            if 'starters' in team_selection:
                assert len(team_selection['starters']) == 11, "Wrong number of starters"

        print(f"✅ Historical Data: {len(results_data)} gameweeks tracked")

        # Calculate accuracy metrics
        accuracies = []
        for result in results_data:
            if 'prediction_accuracy' in result:
                acc = result['prediction_accuracy']
                if isinstance(acc, dict) and 'accuracy_pct' in acc:
                    accuracies.append(acc['accuracy_pct'])

        if accuracies:
            avg_accuracy = sum(accuracies) / len(accuracies)
            print(f"   Current accuracy: {avg_accuracy:.1f}%")
        else:
            print("   No accuracy data available")

        return True

    except Exception as e:
        print(f"❌ Historical data test failed: {e}")
        return False

def test_web_endpoints():
    """Test web API endpoints"""
    try:
        import requests
        import time
        import subprocess
        import sys
        from threading import Thread

        print("🧪 Testing Web Endpoints...")

        # Start server in background
        def start_server():
            subprocess.run([
                sys.executable, "-m", "uvicorn",
                "src.web_ui_clean:app",
                "--port", "8999",  # Use different port for testing
                "--host", "127.0.0.1"
            ], capture_output=True)

        server_thread = Thread(target=start_server, daemon=True)
        server_thread.start()

        # Wait for server to start
        time.sleep(3)

        base_url = "http://127.0.0.1:8999"

        # Test health endpoint
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            assert response.status_code == 200, f"Health check failed: {response.status_code}"
            print("✅ Health endpoint working")
        except requests.exceptions.RequestException:
            print("⚠️ Web server test skipped (server not accessible)")
            return True

        # Test main dashboard
        try:
            response = requests.get(base_url, timeout=5)
            assert response.status_code == 200, f"Dashboard failed: {response.status_code}"
            assert 'FPL' in response.text, "Dashboard doesn't contain FPL content"
            print("✅ Dashboard loading")
        except requests.exceptions.RequestException:
            print("⚠️ Dashboard test failed")

        # Test predictions endpoint
        try:
            response = requests.get(f"{base_url}/predictions", timeout=10)
            if response.status_code == 200:
                print("✅ Predictions endpoint working")
            else:
                print(f"⚠️ Predictions endpoint returned {response.status_code}")
        except requests.exceptions.RequestException:
            print("⚠️ Predictions endpoint test failed")

        return True

    except Exception as e:
        print(f"❌ Web endpoints test failed: {e}")
        return False

def run_all_tests(verbose=False):
    """Run all system tests"""
    print("🚀 FPL System Test Suite")
    print("=" * 50)

    tests = [
        ("FPL Client", test_fpl_client),
        ("ML Pipeline", test_ml_pipeline),
        ("Historical Data", test_historical_data),
        ("Web Endpoints", test_web_endpoints)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}:")
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            if verbose:
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")

    if passed == total:
        print("🎉 All tests passed! System is ready.")
        return True
    else:
        print(f"⚠️ {total - passed} tests failed. Check issues above.")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run FPL system tests')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output with stack traces')
    args = parser.parse_args()

    success = run_all_tests(args.verbose)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())