#!/bin/bash
set -euo pipefail

echo "Setting up FPL AI System for 2025-26 Season..."

# Create venv
if [ ! -d "venv" ]; then
  python3 -m venv venv
fi
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# Create runtime folders
mkdir -p data/{raw,processed,historical,afcon} models/{2025_26,archived} logs reports

# Copy env template if .env not present
if [ ! -f ".env" ]; then
  if [ -f "env.template" ]; then
    cp env.template .env
    echo ".env created from env.template — please edit your credentials."
  else
    echo "env.template missing; please create .env manually."
  fi
fi

echo "FPL AI 2025-26 setup complete. Next steps:"
echo "1) Edit .env with your credentials"
echo "2) Run: python main.py validate"
echo "3) Run: python main.py setup"
echo "4) Run: python main.py analyze"

