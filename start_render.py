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

    # Start the server
    cmd = [
        sys.executable, "-m", "uvicorn",
        "src.web_ui_clean:app",
        "--host", "0.0.0.0",
        "--port", port,
        "--workers", "1"  # Single worker for free tier
    ]

    print(f"🌟 Running: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()