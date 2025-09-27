#!/usr/bin/env python3
"""
Render.com startup script for FPL AI Dashboard
Optimized for free tier hosting with automated retraining
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    # Get port from environment (Render sets this)
    port = os.environ.get('PORT', '8000')

    print(f"🚀 Starting FPL AI Dashboard on port {port}")
    print("📊 Free Render deployment with automated retraining")
    print("🎯 Target: 70-75% accuracy with weekly optimization")

    # Ensure we're in the right directory
    os.chdir(Path(__file__).parent)

    # Initialize cloud system
    print("🔧 Initializing cloud system...")
    try:
        from src.cloud_init import CloudInitializer
        cloud_init = CloudInitializer()

        print("🏗️  Running cloud initialization...")
        result = cloud_init.initialize_system()

        if result["status"] == "success":
            print("✅ Cloud initialization completed successfully")
            print(f"📊 Steps completed: {', '.join(result['steps_completed'])}")
            if result.get("errors"):
                print(f"⚠️  Warnings: {', '.join(result['errors'])}")
        else:
            print(f"⚠️  Initialization had issues: {result['message']}")
            if result.get("errors"):
                print(f"❌ Errors: {', '.join(result['errors'])}")

    except Exception as e:
        print(f"⚠️  Initialization error (continuing anyway): {e}")

    # Set environment for model paths
    os.environ.setdefault('MODEL_PATH', 'models/2025_26')
    os.environ.setdefault('DATA_PATH', 'data')

    # Start the server
    cmd = [
        sys.executable, "-m", "uvicorn",
        "src.web_ui_clean:app",
        "--host", "0.0.0.0",
        "--port", port,
        "--workers", "1"  # Single worker for free tier
    ]

    print(f"🌟 Running: {' '.join(cmd)}")

    # Start the server in background to allow for post-startup tasks
    import threading
    import time
    import requests

    def post_startup_tasks():
        """Run post-startup tasks after server is ready"""
        time.sleep(10)  # Wait for server to fully start
        try:
            print("🔥 Generating initial predictions...")
            response = requests.get(f"http://localhost:{port}/enriched-predictions", timeout=60)
            if response.status_code == 200:
                data = response.json()
                if data.get('source') == 'fallback_predictor':
                    print("⚠️  Using fallback predictions (ML pipeline may have failed)")
                else:
                    print("✅ Successfully generated ML predictions")
            else:
                print(f"⚠️  Prediction generation returned {response.status_code}")
        except Exception as e:
            print(f"⚠️  Could not generate initial predictions: {e}")

    # Start post-startup tasks in background
    startup_thread = threading.Thread(target=post_startup_tasks)
    startup_thread.daemon = True
    startup_thread.start()

    # Run the server (this blocks)
    subprocess.run(cmd)

if __name__ == "__main__":
    main()