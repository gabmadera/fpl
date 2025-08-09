FPL AI System – 2025-26 Season (Skeleton)

This repository contains a runnable skeleton for a personal FPL decision-support system tailored to the 2025-26 rule changes (double chip sets, defensive contribution scoring, AFCON management).

Quick start

- Create virtualenv and install requirements
  - python3 -m venv venv && source venv/bin/activate
  - pip install -r requirements.txt
- Copy env template and fill credentials
  - cp .env.template .env
- Validate setup
  - python main.py validate
- First run
  - python main.py setup
  - python main.py analyze

Main commands

- validate: checks environment and API reachability
- setup: prepares folders and performs a basic data collection
- analyze: runs the weekly analysis pipeline (stub outputs)
- schedule: starts the scheduler (long-running)
- chips: prints chip recommendations (stub)
- afcon: prints AFCON impact summary (stub)

Project layout

- src/: application code modules
- data/: raw/processed historical data (gitignored)
- models/: saved models and artifacts (gitignored)
- reports/: generated weekly reports (gitignored)
- logs/: runtime logs (gitignored)

Notes

- This is a framework/skeleton. Many functions are intentionally minimal and marked TODO, ready for iterative implementation.
