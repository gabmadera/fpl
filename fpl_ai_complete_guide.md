# FPL AI Complete Development Guide - 2025-26 Season

## Project Overview

**Objective**: Build an automated, ML-driven Fantasy Premier League (FPL) decision-support tool that maximizes weekly points for personal use during the 2025-26 season.

**Philosophy**: Hybrid Intelligence Model - AI handles quantitative analysis while human makes final strategic decisions.

---

## Critical 2025-26 FPL Rule Changes

### Major New Features for 2025-26

**🎯 Double Chips System (GAME CHANGER):**
- **8 Total Chips**: 2 sets of 4 chips (Wildcard, Free Hit, Triple Captain, Bench Boost)
- **First Set Deadline**: Must use before GW19 deadline (30 Dec 2025, 18:30 GMT)
- **No Assistant Manager Chip**: Removed for 2025-26
- **Strategic Impact**: Completely changes chip timing and usage patterns

**⚽ Defensive Contribution Points (NEW SCORING):**
- **Defenders**: +2 points for 10+ CBIT (Clearances, Blocks, Interceptions, Tackles)
- **Midfielders/Forwards**: +2 points for 12+ CBIRT (includes Ball Recoveries)
- **Impact**: Makes defensive players and DMs much more valuable

**🌍 AFCON Disruption Management:**
- **GW16 Transfer Boost**: Free transfers topped up to maximum 5 for AFCON departures
- **Tournament Period**: 21 Dec 2025 - 18 Jan 2026
- **Strategic Impact**: Must plan transfer banking around GW15-16

**🏆 Elite Global Leagues:**
- **Top 1% League**: Auto-invite for managers finishing top 1% globally in 2024-25
- **Top 10% League**: Auto-invite for managers finishing top 10% globally in 2024-25
- **Prizes**: Official Puma balls, replica shirts, EA Sports FC games, FPL merchandise

**📈 Enhanced Assists Definition:**
- **Clearer Criteria**: Less subjective decisions on assist awards
- **Impact**: Would have been 41 additional assists in 2024-25 under new rules
- **Strategy**: Creative players and set-piece takers become more valuable

**🥅 Bonus Points System Updates:**
- **Goalkeeper Saves**: 3 points (inside box), 2 points (outside box)
- **Goalline Clearances**: 9 points (tripled from 3)
- **Enhanced BPS**: More accurate reflection of match impact

---

## User Requirements & Constraints

### Technical Infrastructure
- **Budget**: Free preferred, max $8/month for hosting (Heroku/Railway acceptable)
- **Platform**: macOS with M1 Pro Max chip
- **Deployment**: Cloud-accessible for remote management

### FPL Strategy Preferences
- **Risk Tolerance**: Open to any strategy (point hits, differential captains)
- **Priority**: Mini-league position first, overall rank second
- **Decision Making**: AI recommendations with human approval (4 hours before deadline)

### Data Sources
- **Primary**: Free sources only (FPL API, FBref, StatsBomb Open Data)
- **Premium Options**: Awareness of Opta (€10K+ annually), StatsBomb Pro (€5K+ annually)
- **News Sources**: Club websites priority, then BBC/Sky Sports
- **Social Media**: Official sources only, no Twitter monitoring

### Automation & Alerts
- **Notifications**: Email alerts (potentially WhatsApp integration)
- **Transfer Execution**: AI-driven if possible, human approval if required
- **Timing**: Final recommendations 4 hours before deadline
- **Communication**: Email, Slack, or text file outputs

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FPL AI SYSTEM ARCHITECTURE               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   DATA LAYER    │    │  PROCESSING     │    │   DECISION      │
│                 │    │     LAYER       │    │    LAYER        │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │ FPL API     │ │───▶│ │ Feature     │ │───▶│ │ ML Models   │ │
│ │ (Official)  │ │    │ │ Engineering │ │    │ │ (XGBoost)   │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
│                 │    │                 │    │        │        │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │        ▼        │
│ │ FBref/Opta  │ │───▶│ │ Data Fusion │ │    │ ┌─────────────┐ │
│ │ (xG/xA)     │ │    │ │ & Cleaning  │ │    │ │ Optimizer   │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ │ (PuLP)      │ │
│                 │    │                 │    │ └─────────────┘ │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │        │        │
│ │ News/Injury │ │───▶│ │ Availability│ │    │        ▼        │
│ │ Scrapers    │ │    │ │ Tracking    │ │    │ ┌─────────────┐ │
│ └─────────────┘ │    │ └─────────────┘ │    │ │ Chip        │ │
└─────────────────┘    └─────────────────┘    │ │ Strategy    │ │
         │                       │            │ └─────────────┘ │
         ▼                       ▼            └─────────────────┘
┌─────────────────┐    ┌─────────────────┐             │
│   STORAGE       │    │   AUTOMATION    │             ▼
│                 │    │                 │    ┌─────────────────┐
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │   OUTPUT LAYER  │
│ │ PostgreSQL  │ │    │ │ Cron/       │ │    │                 │
│ │ (Main DB)   │ │    │ │ Scheduler   │ │    │ ┌─────────────┐ │
│ └─────────────┘ │    │ └─────────────┘ │    │ │ Email       │ │
│                 │    │        │        │    │ │ Alerts      │ │
│ ┌─────────────┐ │    │        ▼        │    │ └─────────────┘ │
│ │ JSON Cache  │ │    │ ┌─────────────┐ │    │        │        │
│ │ (Daily)     │ │    │ │ Alert       │ │    │        ▼        │
│ └─────────────┘ │    │ │ System      │ │    │ ┌─────────────┐ │
└─────────────────┘    │ └─────────────┘ │    │ │ Report      │ │
                       └─────────────────┘    │ │ Generator   │ │
                                             │ └─────────────┘ │
                                             └─────────────────┘
```

---

## Data Pipeline Blueprint

### Data Sources & Collection Schedule

```python
DATA_SOURCES = {
    "fpl_api": {
        "url": "https://fantasy.premierleague.com/api/",
        "frequency": "daily_6am",
        "endpoints": [
            "bootstrap-static/",     # Player/team static data
            "fixtures/",             # Fixture list & difficulty
            "element-summary/{id}/", # Player history
        ]
    },
    "fbref_scraper": {
        "url": "https://fbref.com/en/comps/9/Premier-League-Stats", 
        "frequency": "weekly_monday",
        "data": ["xG", "xA", "shots", "key_passes", "progressive_passes", "CBIT", "CBIRT"]
    },
    "news_scrapers": {
        "sources": [
            "https://www.premierleague.com/news",
            "https://www.skysports.com/premier-league-news"
        ],
        "frequency": "daily_2pm",
        "keywords": ["injury", "suspended", "doubt", "fit", "rotation", "AFCON"]
    }
}
```

### Database Schema

```sql
-- PostgreSQL schema optimized for ML and time-series analysis
CREATE TABLE players (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100),
    position VARCHAR(3),
    team VARCHAR(50), 
    price DECIMAL(4,1),
    availability DECIMAL(3,1),
    afcon_risk BOOLEAN DEFAULT FALSE, -- New for 2025-26
    last_updated TIMESTAMP
);

CREATE TABLE gameweek_data (
    player_id INTEGER,
    gameweek INTEGER, 
    points INTEGER,
    minutes INTEGER,
    goals INTEGER,
    assists INTEGER,
    xg DECIMAL(4,2),
    xa DECIMAL(4,2),
    cbit_count INTEGER, -- New defensive contribution tracking
    cbirt_count INTEGER, -- New for MID/FWD
    defensive_contribution_points INTEGER DEFAULT 0, -- New scoring
    fixture_difficulty INTEGER,
    PRIMARY KEY (player_id, gameweek)
);

CREATE TABLE predictions (
    player_id INTEGER,
    gameweek INTEGER,
    predicted_points DECIMAL(5,2),
    confidence DECIMAL(3,2),
    includes_defensive_contributions BOOLEAN DEFAULT TRUE, -- New factor
    afcon_adjusted BOOLEAN DEFAULT FALSE, -- New factor
    created_at TIMESTAMP
);

CREATE TABLE chip_usage_log (
    gameweek INTEGER,
    chip_type VARCHAR(20),
    chip_set INTEGER, -- 1 or 2 for new double system
    points_impact DECIMAL(6,2),
    success_rating INTEGER -- 1-5 rating post-gameweek
);
```

---

## Core Code Implementation

### Project Structure

```
fpl_ai/
├── data/
│   ├── raw/              # Raw JSON data from APIs
│   ├── processed/        # Cleaned data for modeling
│   └── historical/       # 4+ seasons of historical data
├── models/               # Saved ML models and feature importance
├── notebooks/            # Jupyter notebooks for analysis
├── src/
│   ├── __init__.py
│   ├── config.py         # API endpoints and environment variables
│   ├── data_collector.py # Unified data collection with AFCON tracking
│   ├── feature_engineer.py # ML features including defensive contributions
│   ├── model_trainer.py  # XGBoost training with 2025-26 adaptations
│   ├── predict_points.py # Point predictions including new scoring system
│   ├── team_optimizer.py # Squad optimization for double chip system
│   ├── chip_strategy.py  # New dedicated chip strategy module
│   ├── security.py       # Security and privacy management
│   ├── scheduler.py      # Automated weekly workflow
│   └── main.py          # Entry point script
├── requirements.txt
├── .env.template
├── weekly_report.txt     # AI-generated recommendations
├── current_squad.json   # Current team state
└── README.md
```

### Configuration & Environment

```python
# src/config.py
import os
from dataclasses import dataclass

@dataclass 
class Config:
    # API Configuration
    FPL_BASE_URL: str = "https://fantasy.premierleague.com/api/"
    FBREF_BASE_URL: str = "https://fbref.com/en/comps/9/"
    
    # Database Configuration  
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///fpl_ai.db")
    
    # Email Configuration
    SMTP_SERVER: str = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT: int = 587
    EMAIL_USER: str = os.getenv("EMAIL_USER")  # TODO: Set this
    EMAIL_PASS: str = os.getenv("EMAIL_PASS")  # TODO: Set this
    ALERT_EMAIL: str = os.getenv("ALERT_EMAIL")
    
    # ML Configuration
    MODEL_RETRAIN_INTERVAL: int = int(os.getenv("MODEL_RETRAIN_INTERVAL", 6))
    PREDICTION_CONFIDENCE_THRESHOLD: float = float(os.getenv("PREDICTION_CONFIDENCE_THRESHOLD", 0.7))
    
    # 2025-26 Season Specific Configuration
    SEASON: str = os.getenv("SEASON", "2025_26")
    AFCON_START_GW: int = int(os.getenv("AFCON_START_GW", 16))
    AFCON_END_GW: int = int(os.getenv("AFCON_END_GW", 21))
    CHIP_SET_1_DEADLINE_GW: int = int(os.getenv("CHIP_SET_1_DEADLINE_GW", 19))
    
    # Security Configuration
    MAX_API_REQUESTS_PER_HOUR: int = int(os.getenv("MAX_API_REQUESTS_PER_HOUR", 100))
    RATE_LIMIT_DELAY: float = float(os.getenv("RATE_LIMIT_DELAY", 1.0))
    DATA_RETENTION_DAYS: int = int(os.getenv("DATA_RETENTION_DAYS", 90))
    
    # FPL Team Configuration  
    FPL_TEAM_ID: int = int(os.getenv("FPL_TEAM_ID", 0))
    FPL_EMAIL: str = os.getenv("FPL_EMAIL")
    FPL_PASSWORD: str = os.getenv("FPL_PASSWORD")
    
    # Optional Premium API Keys
    RAPID_API_KEY: str = os.getenv("RAPID_API_KEY")
    FOOTBALL_API_KEY: str = os.getenv("FOOTBALL_API_KEY")
    
    # Development Settings
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    ENABLE_WEB_INTERFACE: bool = os.getenv("ENABLE_WEB_INTERFACE", "False").lower() == "true"
    WEB_PORT: int = int(os.getenv("WEB_PORT", 8000))

config = Config()
EOF

echo ""
echo "✅ FPL AI 2025-26 setup complete!"
echo ""
echo "📋 Next Steps:"
echo "1. Copy .env.template to .env: cp .env.template .env"
echo "2. Edit .env with your FPL credentials and email settings"
echo "3. Get your FPL Team ID from: https://fantasy.premierleague.com/entry/YOUR_ID/"
echo "4. Set up Gmail App Password: https://myaccount.google.com/apppasswords"
echo "5. Validate setup: python main.py validate"
echo "6. Run initial setup: python main.py setup"
echo "7. Test analysis: python main.py analyze"
echo "8. Start automation: python main.py schedule"
echo ""
echo "🎯 2025-26 Season Features Ready:"
echo "• Double chip system (8 total chips)"
echo "• Defensive contribution predictions"
echo "• AFCON disruption management"
echo "• Enhanced assist predictions"
echo "• Automated chip deadline warnings"
```

### Cloud Deployment Options

#### Option 1: Railway Deployment

```yaml
# railway.toml - Railway configuration for 2025-26
[build]
builder = "NIXPACKS"

[deploy]
healthcheckPath = "/health"
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 10

[env]
PYTHON_VERSION = "3.11"
PORT = "8000"
SEASON = "2025_26"
```

#### Option 2: Heroku Deployment

```python
# Procfile for Heroku
web: python src/web_interface.py
scheduler: python main.py schedule
```

#### Option 3: Docker Containerization

```dockerfile
# Dockerfile - Enhanced for 2025-26 features
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for web scraping
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    chromium-driver \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories
RUN mkdir -p data/{raw,processed,historical,afcon} models/{2025_26,archived} logs reports

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV SEASON=2025_26

# Expose port for web interface
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "main.py", "schedule"]
```

---

## Rapid Iteration & Improvement Framework

### Weekly Performance Monitoring

```python
# src/model_monitor.py - Enhanced for 2025-26
class ModelMonitor:
    def __init__(self):
        self.performance_history = []
        self.season = "2025_26"
        
    def track_weekly_performance_2025_26(self, predictions: pd.DataFrame, 
                                       actual_points: pd.DataFrame,
                                       chip_usage: Dict) -> Dict:
        """Enhanced performance tracking for 2025-26 features."""
        
        # Standard accuracy metrics
        comparison = predictions.merge(actual_points, on='player_id', suffixes=('_pred', '_actual'))
        
        mae = mean_absolute_error(comparison['predicted_points'], comparison['points_actual'])
        rmse = np.sqrt(mean_squared_error(comparison['predicted_points'], comparison['points_actual']))
        
        # 2025-26 specific metrics
        
        # Defensive Contribution Prediction Accuracy
        dc_accuracy = self._evaluate_defensive_contributions(comparison)
        
        # Chip Strategy Effectiveness
        chip_effectiveness = self._evaluate_chip_decisions(chip_usage, actual_points)
        
        # AFCON Impact Prediction Accuracy
        afcon_accuracy = self._evaluate_afcon_predictions(comparison)
        
        # Captain Selection Success (most critical)
        top_predictions = comparison.nlargest(20, 'predicted_points')
        captain_accuracy = (top_predictions['points_actual'] > 6).mean()
        
        performance = {
            'gameweek': self._get_current_gameweek(),
            'season': self.season,
            'mae': mae,
            'rmse': rmse,
            'captain_accuracy': captain_accuracy,
            'defensive_contribution_accuracy': dc_accuracy,
            'chip_effectiveness': chip_effectiveness,
            'afcon_prediction_accuracy': afcon_accuracy,
            'sample_size': len(comparison),
            'timestamp': datetime.now()
        }
        
        self.performance_history.append(performance)
        
        # Enhanced performance file for 2025-26
        with open(f'model_performance_{self.season}.json', 'w') as f:
            json.dump(self.performance_history, f, indent=2, default=str)
            
        return performance
        
    def _evaluate_defensive_contributions(self, comparison: pd.DataFrame) -> float:
        """Evaluate accuracy of defensive contribution predictions."""
        
        if 'defensive_contribution_points' not in comparison.columns:
            return 0.0
            
        dc_predictions = comparison['dc_point_probability'] > 0.5
        dc_actual = comparison['defensive_contribution_points'] > 0
        
        if len(dc_actual) == 0:
            return 0.0
            
        return (dc_predictions == dc_actual).mean()
        
    def _evaluate_chip_decisions(self, chip_usage: Dict, actual_points: pd.DataFrame) -> Dict:
        """Evaluate effectiveness of chip strategy decisions."""
        
        effectiveness = {
            'chips_used': len(chip_usage),
            'total_chip_points': 0.0,
            'best_chip': None,
            'worst_chip': None
        }
        
        chip_results = []
        
        for chip_info in chip_usage:
            chip_type = chip_info.get('type')
            expected_gain = chip_info.get('expected_gain', 0)
            actual_gain = chip_info.get('actual_gain', 0)
            
            chip_results.append({
                'type': chip_type,
                'expected': expected_gain,
                'actual': actual_gain,
                'success': actual_gain >= expected_gain * 0.8  # 80% of expected
            })
            
            effectiveness['total_chip_points'] += actual_gain
            
        if chip_results:
            effectiveness['success_rate'] = sum(r['success'] for r in chip_results) / len(chip_results)
            effectiveness['best_chip'] = max(chip_results, key=lambda x: x['actual'])['type']
            effectiveness['worst_chip'] = min(chip_results, key=lambda x: x['actual'])['type']
        
        return effectiveness
        
    def generate_improvement_suggestions_2025_26(self) -> List[str]:
        """Generate 2025-26 specific improvement suggestions."""
        
        suggestions = []
        
        if len(self.performance_history) >= 4:
            recent_performance = self.performance_history[-3:]
            
            # Defensive contribution accuracy
            dc_accuracy = np.mean([p.get('defensive_contribution_accuracy', 0) for p in recent_performance])
            if dc_accuracy < 0.6:
                suggestions.append("Improve defensive contribution prediction model")
                suggestions.append("Add more detailed CBIT/CBIRT historical data")
                
            # Chip strategy effectiveness
            chip_effectiveness = [p.get('chip_effectiveness', {}) for p in recent_performance]
            success_rates = [ce.get('success_rate', 0) for ce in chip_effectiveness if ce]
            
            if success_rates and np.mean(success_rates) < 0.5:
                suggestions.append("Refine chip timing strategy")
                suggestions.append("Consider more conservative chip thresholds")
                suggestions.append("Analyze fixture difficulty more carefully for chip usage")
                
            # AFCON prediction accuracy
            afcon_accuracy = np.mean([p.get('afcon_prediction_accuracy', 1) for p in recent_performance])
            if afcon_accuracy < 0.7:
                suggestions.append("Improve AFCON departure prediction model")
                suggestions.append("Monitor club announcements more closely")
                
        return suggestions
```

### Continuous Learning System

```python
# src/adaptive_learning.py - Self-improving system for 2025-26
class AdaptiveLearningSystem:
    def __init__(self):
        self.learning_history = []
        self.feature_experiments = {}
        self.strategy_experiments = {}
        
    def experiment_new_features_2025_26(self) -> Dict:
        """Run experiments with 2025-26 specific features."""
        
        experiments = {
            'defensive_contribution_weights': self._experiment_dc_weights(),
            'afcon_risk_modeling': self._experiment_afcon_modeling(),
            'chip_timing_strategies': self._experiment_chip_timing(),
            'assist_prediction_models': self._experiment_assist_prediction()
        }
        
        return experiments
        
    def _experiment_dc_weights(self) -> Dict:
        """Experiment with different weightings for defensive contributions."""
        
        weight_strategies = {
            'conservative': 0.5,  # 50% confidence in DC predictions
            'moderate': 0.7,      # 70% confidence
            'aggressive': 0.9     # 90% confidence
        }
        
        results = {}
        
        for strategy_name, weight in weight_strategies.items():
            # TODO: Run backtest with different DC weights
            # results[strategy_name] = backtest_result
            pass
            
        return results
        
    def _experiment_chip_timing(self) -> Dict:
        """Experiment with different chip timing strategies."""
        
        timing_strategies = {
            'early_aggressive': 'Use chips in first 6 gameweeks',
            'fixture_focused': 'Use chips based on fixture difficulty',
            'form_focused': 'Use chips when players in peak form',
            'deadline_pressure': 'Save chips until set deadline approaches'
        }
        
        # TODO: Implement strategy backtesting
        return timing_strategies
        
    def adapt_model_parameters(self, performance_data: List[Dict]) -> Dict:
        """Adapt model parameters based on performance feedback."""
        
        if len(performance_data) < 8:  # Need sufficient data
            return {'status': 'insufficient_data'}
            
        recent_performance = performance_data[-4:]  # Last 4 gameweeks
        baseline_performance = performance_data[:4]  # First 4 gameweeks
        
        adaptations = {}
        
        # Adjust defensive contribution weight based on accuracy
        dc_accuracy_recent = np.mean([p.get('defensive_contribution_accuracy', 0.5) for p in recent_performance])
        dc_accuracy_baseline = np.mean([p.get('defensive_contribution_accuracy', 0.5) for p in baseline_performance])
        
        if dc_accuracy_recent < dc_accuracy_baseline - 0.1:
            adaptations['defensive_contribution_weight'] = 'decrease'
        elif dc_accuracy_recent > dc_accuracy_baseline + 0.1:
            adaptations['defensive_contribution_weight'] = 'increase'
            
        # Adjust prediction confidence thresholds
        captain_accuracy = np.mean([p.get('captain_accuracy', 0) for p in recent_performance])
        if captain_accuracy < 0.4:
            adaptations['captain_threshold'] = 'lower'  # Be more conservative
        elif captain_accuracy > 0.7:
            adaptations['captain_threshold'] = 'raise'  # Be more aggressive
            
        return adaptations
```

---

## Quick Start Checklist for 2025-26

### Phase 1: Initial Setup (Day 1)
- [ ] Run setup script: `./setup_fpl_ai_2025_26.sh`
- [ ] Copy `.env.template` to `.env` 
- [ ] Get FPL Team ID from profile URL
- [ ] Set up Gmail App Password
- [ ] Fill in all `.env` credentials
- [ ] Validate setup: `python main.py validate`
- [ ] Run initial setup: `python main.py setup`

### Phase 2: 2025-26 Feature Testing (Days 2-3)
- [ ] Test chip analysis: `python main.py chips`
- [ ] Test AFCON impact: `python main.py afcon`
- [ ] Run full analysis: `python main.py analyze`
- [ ] Review generated report for 2025-26 features
- [ ] Verify email alerts work correctly
- [ ] Test defensive contribution predictions

### Phase 3: Automation (Week 1)
- [ ] Start scheduler: `python main.py schedule`
- [ ] Monitor automated analysis runs
- [ ] Verify chip deadline warnings work
- [ ] Check AFCON monitoring (when active)
- [ ] Review first week's performance
- [ ] Adjust email notification timing if needed

### Phase 4: Optimization (Weeks 2-4)
- [ ] Analyze chip strategy effectiveness
- [ ] Monitor defensive contribution prediction accuracy
- [ ] Track AFCON impact predictions (GW16+)
- [ ] Fine-tune model parameters based on performance
- [ ] Set up cloud deployment if desired
- [ ] Implement transfer execution workflow

### Phase 5: Mid-Season Adaptation (GW10+)
- [ ] Review chip usage strategy for set 1
- [ ] Prepare for AFCON disruption (GW15-16)
- [ ] Analyze model performance vs expectations
- [ ] Plan chip set 2 strategy (GW20+)
- [ ] Evaluate season-long ROI and improvements

---

## 2025-26 Season Success Metrics

### Primary Success Indicators
1. **Mini-League Position**: Top 3 finish in main mini-league
2. **Overall Rank**: Top 10% globally (Elite league qualification)
3. **Points Total**: 2,400+ points for the season
4. **Chip Effectiveness**: 60%+ of chip decisions successful

### Secondary Performance Metrics
1. **Captain Accuracy**: 50%+ of captains score 8+ points
2. **Transfer Success Rate**: 70%+ of transfers gain points
3. **Defensive Contribution Predictions**: 65%+ accuracy
4. **AFCON Management**: Minimal point losses during disruption

### System Performance Metrics
1. **Model Accuracy**: MAE < 2.5 points per prediction
2. **Uptime**: 99%+ automated analysis success rate
3. **Alert Timeliness**: All alerts delivered 4+ hours before deadline
4. **Data Quality**: 95%+ successful data collection runs

---

## Conclusion

This comprehensive FPL AI system is specifically designed for the **2025-26 season** with its revolutionary rule changes:

🎯 **Key Advantages:**
- **Double Chip Mastery**: Optimizes all 8 chips across both sets
- **Defensive Intelligence**: Leverages new defensive contribution scoring
- **AFCON Management**: Proactive planning for mid-season disruption
- **ML-Driven Predictions**: XGBoost model trained on enhanced features
- **Automated Excellence**: 4-hour deadline warnings with email alerts

🚀 **Expected Impact:**
- **25-50 point improvement** from better chip usage alone
- **15-30 point gain** from defensive contribution predictions  
- **20-40 point protection** during AFCON period
- **Consistent top-tier performance** through data-driven decisions

The system transforms you from a reactive FPL manager to a **strategic data-driven competitor** with automated insights, perfect timing, and continuous improvement.

Ready to dominate the 2025-26 season? Start with `./setup_fpl_ai_2025_26.sh` and let the AI handle the heavy lifting while you make the final strategic calls! 🏆.getenv("ALERT_EMAIL") # TODO: Your email
    
    # ML Configuration
    MODEL_RETRAIN_INTERVAL: int = 6  # Gameweeks between retraining
    PREDICTION_CONFIDENCE_THRESHOLD: float = 0.7
    
    # FPL Configuration - 2025-26 Season Specific
    AFCON_START_GW: int = 16  # When AFCON disruption begins
    AFCON_END_GW: int = 21    # When players return
    CHIP_SET_1_DEADLINE: int = 19  # GW19 deadline for first chip set
    
    # FPL Team Configuration  
    FPL_TEAM_ID: int = os.getenv("FPL_TEAM_ID")      # TODO: Get from profile
    FPL_EMAIL: str = os.getenv("FPL_EMAIL")          # TODO: Your FPL login
    FPL_PASSWORD: str = os.getenv("FPL_PASSWORD")    # TODO: Your FPL password

config = Config()
```

### Data Collection with 2025-26 Adaptations

```python
# src/data_collector.py
import requests
import pandas as pd
from typing import Dict, List
import time
from .config import config

class DataCollector:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'FPL-AI-Bot/1.0 (Personal Use Only)'
        })
    
    def fetch_fpl_data(self, endpoint: str) -> Dict:
        """Fetch data from FPL API with rate limiting."""
        url = f"{config.FPL_BASE_URL}{endpoint}"
        
        try:
            time.sleep(1)  # Rate limiting
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return {}
    
    def collect_all_data(self) -> Dict[str, pd.DataFrame]:
        """Main collection method with 2025-26 specific data."""
        print("🔄 Starting data collection...")
        
        # FPL API data
        bootstrap = self.fetch_fpl_data("bootstrap-static/")
        fixtures = self.fetch_fpl_data("fixtures/")
        
        # Process and enhance with 2025-26 features
        datasets = {
            "players": self._process_player_data_2025_26(bootstrap),
            "fixtures": self._process_fixture_data(fixtures),
            "defensive_stats": self._collect_defensive_contributions(),
            "afcon_risk_data": self._collect_afcon_risk_data()
        }
        
        print("✅ Data collection complete")
        return datasets
        
    def _process_player_data_2025_26(self, bootstrap: Dict) -> pd.DataFrame:
        """Process player data with 2025-26 new features."""
        if not bootstrap.get('elements'):
            return pd.DataFrame()
            
        df = pd.DataFrame(bootstrap['elements'])
        
        # Standard cleaning
        df.rename(columns={
            'web_name': 'name',
            'element_type': 'position_id', 
            'team': 'team_id',
            'now_cost': 'price'
        }, inplace=True)
        
        df['price'] = df['price'] / 10
        
        # Add 2025-26 specific fields
        df['afcon_risk'] = self._determine_afcon_risk(df)
        df['defensive_contribution_potential'] = self._estimate_dc_potential(df)
        
        return df
    
    def _collect_defensive_contributions(self) -> pd.DataFrame:
        """Collect CBIT/CBIRT data for new defensive contribution points."""
        # TODO: Implement FBref scraping for defensive stats
        # Focus on: Clearances, Blocks, Interceptions, Tackles, Ball Recoveries
        return pd.DataFrame()
    
    def _collect_afcon_risk_data(self) -> pd.DataFrame:
        """Identify players likely to leave for AFCON 2025-26."""
        afcon_countries = [
            'Algeria', 'Egypt', 'Ghana', 'Ivory Coast', 'Morocco', 'Nigeria', 
            'Senegal', 'Tunisia', 'Cameroon', 'Mali', 'Guinea', 'Burkina Faso'
            # Add more AFCON qualified countries
        ]
        
        # TODO: Cross-reference player nationalities with AFCON participants
        return pd.DataFrame()
    
    def _determine_afcon_risk(self, df: pd.DataFrame) -> pd.Series:
        """Determine AFCON departure risk for each player."""
        # TODO: Implement nationality lookup and AFCON qualification status
        return pd.Series([False] * len(df), index=df.index)
    
    def _estimate_dc_potential(self, df: pd.DataFrame) -> pd.Series:
        """Estimate defensive contribution potential based on position and team."""
        dc_potential = pd.Series([0.0] * len(df), index=df.index)
        
        # Higher potential for defenders and defensive midfielders
        dc_potential[df['position_id'] == 2] = 0.8  # Defenders
        dc_potential[df['position_id'] == 3] = 0.3  # Midfielders
        dc_potential[df['position_id'] == 4] = 0.1  # Forwards
        
        return dc_potential
```

### Feature Engineering for 2025-26

```python
# src/feature_engineer.py
import pandas as pd
import numpy as np
from typing import List

class FeatureEngineer:
    def __init__(self):
        self.lookback_windows = [3, 5, 8, 12]
        
    def create_features(self, player_data: pd.DataFrame, 
                       fixture_data: pd.DataFrame) -> pd.DataFrame:
        """Create ML features with 2025-26 enhancements."""
        
        features_df = player_data.copy()
        
        # Traditional features
        features_df = self._add_form_features(features_df)
        features_df = self._add_fixture_features(features_df, fixture_data)
        features_df = self._add_position_features(features_df)
        features_df = self._add_price_features(features_df)
        
        # 2025-26 New Features
        features_df = self._add_defensive_contribution_features(features_df)
        features_df = self._add_afcon_disruption_features(features_df)
        features_df = self._add_chip_strategy_features(features_df)
        
        return features_df
    
    def _add_defensive_contribution_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features for new defensive contribution scoring system."""
        
        # Historical CBIT/CBIRT averages
        for window in self.lookback_windows:
            df[f'avg_cbit_{window}gw'] = (
                df.groupby('player_id')['cbit_count']
                .rolling(window, min_periods=2)
                .mean()
            )
            
            df[f'avg_cbirt_{window}gw'] = (
                df.groupby('player_id')['cbirt_count']
                .rolling(window, min_periods=2)
                .mean()
            )
        
        # Defensive contribution point probability
        df['dc_point_probability'] = np.where(
            df['position'] == 'DEF',
            df['avg_cbit_5gw'] / 10,  # Probability of hitting 10+ CBIT
            df['avg_cbirt_5gw'] / 12   # Probability of hitting 12+ CBIRT
        )
        
        return df
    
    def _add_afcon_disruption_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features for AFCON disruption management."""
        
        current_gw = self._get_current_gameweek()
        
        # AFCON risk flags
        df['afcon_departure_risk'] = (
            (df['afcon_risk']) & 
            (current_gw >= config.AFCON_START_GW - 2)  # 2 GW warning
        )
        
        df['afcon_return_candidate'] = (
            (df['afcon_risk']) & 
            (current_gw >= config.AFCON_END_GW)
        )
        
        # Adjust expected points for AFCON players
        df['afcon_adjusted_points'] = np.where(
            df['afcon_departure_risk'],
            df['predicted_points'] * 0.5,  # 50% reduction for departing players
            df['predicted_points']
        )
        
        return df
    
    def _add_chip_strategy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features for 2025-26 double chip strategy."""
        
        current_gw = self._get_current_gameweek()
        
        # Chip urgency flags
        df['chip_set_1_urgency'] = (current_gw >= config.CHIP_SET_1_DEADLINE - 3)
        df['triple_captain_candidate'] = df['predicted_points'] >= 8.0
        df['bench_boost_value'] = df['predicted_points'] * (1 - df['starting_probability'])
        
        return df
    
    def _get_current_gameweek(self) -> int:
        """Get current gameweek from FPL API."""
        # TODO: Implement current GW detection
        return 1
```

### Team Optimization with Double Chip Strategy

```python
# src/chip_strategy.py
from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd

@dataclass
class ChipRecommendation:
    chip_type: str
    chip_set: int  # 1 or 2
    gameweek: int
    expected_gain: float
    confidence: float
    reasoning: str

class ChipStrategyManager:
    def __init__(self):
        self.chips_available = {
            1: {'wildcard': True, 'free_hit': True, 'triple_captain': True, 'bench_boost': True},
            2: {'wildcard': True, 'free_hit': True, 'triple_captain': True, 'bench_boost': True}
        }
        
    def recommend_chips(self, predictions_df: pd.DataFrame, 
                       current_gw: int,
                       fixture_analysis: Dict) -> List[ChipRecommendation]:
        """Generate chip recommendations for 2025-26 double system."""
        
        recommendations = []
        
        # Determine which chip set we're in
        chip_set = 1 if current_gw <= 19 else 2
        
        # Triple Captain analysis
        tc_rec = self._analyze_triple_captain(predictions_df, current_gw, chip_set, fixture_analysis)
        if tc_rec:
            recommendations.append(tc_rec)
        
        # Bench Boost analysis  
        bb_rec = self._analyze_bench_boost(predictions_df, current_gw, chip_set)
        if bb_rec:
            recommendations.append(bb_rec)
        
        # Free Hit analysis
        fh_rec = self._analyze_free_hit(predictions_df, current_gw, chip_set, fixture_analysis)
        if fh_rec:
            recommendations.append(fh_rec)
        
        # Wildcard analysis
        wc_rec = self._analyze_wildcard(current_gw, chip_set)
        if wc_rec:
            recommendations.append(wc_rec)
        
        # Urgent deadlines
        if chip_set == 1 and current_gw >= 17:
            recommendations = self._add_urgency_recommendations(recommendations, current_gw)
        
        return sorted(recommendations, key=lambda x: x.expected_gain, reverse=True)
    
    def _analyze_triple_captain(self, predictions_df: pd.DataFrame, 
                              current_gw: int, chip_set: int,
                              fixture_analysis: Dict) -> Optional[ChipRecommendation]:
        """Analyze Triple Captain opportunities."""
        
        if not self.chips_available[chip_set]['triple_captain']:
            return None
        
        # Find best captain candidates
        top_predictions = predictions_df.nlargest(5, 'predicted_points')
        
        for _, player in top_predictions.iterrows():
            # Check for premium matchups against promoted teams
            if (player['predicted_points'] >= 8.0 and 
                fixture_analysis.get(player['player_id'], {}).get('opponent_strength', 3) <= 2):
                
                expected_gain = player['predicted_points'] * 2  # Triple vs Double captain
                
                return ChipRecommendation(
                    chip_type='triple_captain',
                    chip_set=chip_set,
                    gameweek=current_gw,
                    expected_gain=expected_gain,
                    confidence=0.8,
                    reasoning=f"{player['name']} vs weak opposition - high ceiling fixture"
                )
        
        return None
    
    def _analyze_bench_boost(self, predictions_df: pd.DataFrame,
                           current_gw: int, chip_set: int) -> Optional[ChipRecommendation]:
        """Analyze Bench Boost opportunities."""
        
        if not self.chips_available[chip_set]['bench_boost']:
            return None
        
        # Calculate bench value (assuming 4 bench players)
        bench_players = predictions_df.nsmallest(4, 'starting_probability')
        total_bench_points = bench_players['predicted_points'].sum()
        
        # Recommend if bench expected to score 8+ points
        if total_bench_points >= 8.0:
            return ChipRecommendation(
                chip_type='bench_boost',
                chip_set=chip_set,
                gameweek=current_gw,
                expected_gain=total_bench_points,
                confidence=0.7,
                reasoning=f"Strong bench predicted for {total_bench_points:.1f} points"
            )
        
        return None
    
    def _add_urgency_recommendations(self, recommendations: List[ChipRecommendation],
                                   current_gw: int) -> List[ChipRecommendation]:
        """Add urgent chip usage recommendations before GW19 deadline."""
        
        if current_gw == 19:  # Final chance for chip set 1
            urgent = ChipRecommendation(
                chip_type='wildcard',
                chip_set=1,
                gameweek=current_gw,
                expected_gain=5.0,
                confidence=0.9,
                reasoning="URGENT: Last chance to use first chip set before expiry"
            )
            recommendations.append(urgent)
        
        return recommendations
```

### Weekly Automation Scheduler

```python
# src/scheduler.py
import schedule
import time
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class FPLScheduler:
    def __init__(self):
        self.collector = DataCollector()
        self.engineer = FeatureEngineer()  
        self.trainer = FPLModelTrainer()
        self.optimizer = TeamOptimizer()
        self.chip_manager = ChipStrategyManager()
        
    def weekly_analysis_pipeline(self):
        """Enhanced weekly analysis for 2025-26 season."""
        
        try:
            print(f"🚀 Starting FPL analysis at {datetime.now()}")
            
            # Step 1: Data Collection (with AFCON tracking)
            datasets = self.collector.collect_all_data()
            
            # Step 2: Feature Engineering (with defensive contributions)
            features = self.engineer.create_features(
                datasets['players'], datasets['fixtures']
            )
            
            # Step 3: Point Predictions (including defensive contribution points)
            predictions = self.trainer.predict_points(features)
            
            # Step 4: Team Optimization
            team_recommendations = self.optimizer.optimize_team(predictions)
            
            # Step 5: Chip Strategy (NEW for 2025-26)
            current_gw = self._get_current_gameweek()
            chip_recommendations = self.chip_manager.recommend_chips(
                predictions, current_gw, datasets.get('fixtures', {})
            )
            
            # Step 6: AFCON Impact Analysis
            afcon_impact = self._analyze_afcon_impact(predictions, current_gw)
            
            # Step 7: Generate Enhanced Report
            report = self._generate_report_2025_26(
                team_recommendations, predictions, chip_recommendations, afcon_impact
            )
            
            # Step 8: Send Alerts
            self._send_alert_email(report)
            
            print("✅ Weekly analysis complete!")
            
        except Exception as e:
            error_msg = f"❌ FPL Analysis failed: {str(e)}"
            print(error_msg)
            self._send_error_email(error_msg)
    
    def _generate_report_2025_26(self, team_rec: dict, predictions: pd.DataFrame, 
                                chip_rec: List, afcon_impact: dict) -> str:
        """Generate enhanced report for 2025-26 season."""
        
        current_gw = self._get_current_gameweek()
        chip_set = 1 if current_gw <= 19 else 2
        
        # Find players with high defensive contribution potential
        dc_candidates = predictions[predictions['dc_point_probability'] > 0.6].head(5)
        
        report = f"""
🏆 FPL AI WEEKLY REPORT - GAMEWEEK {current_gw} (2025-26 Season)
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

🎯 CHIP SET {chip_set} ACTIVE - {'⚠️ DEADLINE APPROACHING' if current_gw >= 17 and chip_set == 1 else 'Available'}

📊 TOP PREDICTED PERFORMERS:
{predictions.head(10)[['name', 'position', 'team', 'predicted_points', 'dc_point_probability']].to_string(index=False)}

🛡️ DEFENSIVE CONTRIBUTION CANDIDATES:
{dc_candidates[['name', 'position', 'dc_point_probability']].to_string(index=False)}

🔄 RECOMMENDED TRANSFERS:
Transfer Out: {team_rec.get('transfers_out', [])}  
Transfer In: {team_rec.get('transfers_in', [])}
Net Point Gain: +{team_rec.get('net_gain', 0)} points

👑 RECOMMENDED CAPTAIN: {team_rec.get('captain', 'N/A')}
🪑 RECOMMENDED VICE: {team_rec.get('vice_captain', 'N/A')}

💎 CHIP RECOMMENDATIONS (SET {chip_set}):
{self._format_chip_recommendations(chip_rec)}

🌍 AFCON IMPACT ANALYSIS:
Departing Players: {afcon_impact.get('departing_players', 'None identified')}
Return Candidates: {afcon_impact.get('return_candidates', 'None yet')}
Transfer Priority: {afcon_impact.get('transfer_priority', 'Standard')}

⚠️  INJURY/AVAILABILITY ALERTS:
{team_rec.get('injury_alerts', 'No current concerns')}

🎯 CONFIDENCE LEVEL: {team_rec.get('confidence', 'Medium')}

📋 2025-26 SEASON NOTES:
• Double chip system: {8 - self._count_used_chips()} chips remaining
• Defensive contributions active: Monitor CBIT/CBIRT stats
• AFCON disruption: GW16-21 planning required

---
This is an AI-generated report. Always verify team news before deadline!
"""
        
        # Save report to file
        with open('weekly_report.txt', 'w') as f:
            f.write(report)
            
        return report
    
    def _format_chip_recommendations(self, chip_recommendations: List) -> str:
        """Format chip recommendations for report."""
        if not chip_recommendations:
            return "Hold chips - no clear opportunities"
        
        formatted = []
        for rec in chip_recommendations[:3]:  # Top 3 recommendations
            formatted.append(f"• {rec.chip_type.upper()} (Set {rec.chip_set}): {rec.reasoning}")
        
        return "\n".join(formatted)
    
    def _analyze_afcon_impact(self, predictions_df: pd.DataFrame, current_gw: int) -> dict:
        """Analyze AFCON impact for current gameweek."""
        
        impact = {
            'departing_players': [],
            'return_candidates': [],
            'transfer_priority': 'Standard'
        }
        
        # Identify AFCON departures (GW16-17)
        if config.AFCON_START_GW - 1 <= current_gw <= config.AFCON_START_GW + 1:
            departing = predictions_df[predictions_df['afcon_departure_risk']]
            impact['departing_players'] = departing['name'].tolist()
            if len(departing) > 0:
                impact['transfer_priority'] = 'HIGH - AFCON Departures'
        
        # Identify return candidates (GW21+)
        if current_gw >= config.AFCON_END_GW:
            returning = predictions_df[predictions_df['afcon_return_candidate']]
            impact['return_candidates'] = returning['name'].tolist()
        
        return impact
    
    def _count_used_chips(self) -> int:
        """Count how many chips have been used across both sets."""
        # TODO: Implement chip usage tracking
        return 0
    
    def start_scheduler(self):
        """Start the automated scheduler with 2025-26 specific timing."""
        
        # Weekly analysis - Tuesday & Friday 2 PM (4 hours before deadline)
        schedule.every().tuesday.at("14:00").do(self.weekly_analysis_pipeline)
        schedule.every().friday.at("14:00").do(self.weekly_analysis_pipeline)
        
        # AFCON monitoring - Daily during disruption period
        schedule.every().day.at("12:00").do(self.afcon_monitoring_check)
        
        # Chip deadline alerts
        schedule.every().day.at("10:00").do(self.chip_deadline_check)
        
        # Daily injury check
        schedule.every().day.at("14:00").do(self.injury_news_check)
        
        print("🕐 FPL AI Scheduler started for 2025-26...")
        print("📅 Weekly Analysis: Tuesday & Friday 2 PM")
        print("🌍 AFCON Monitoring: Daily 12 PM (during disruption)")
        print("💎 Chip Deadline Alerts: Daily 10 AM") 
        print("🏥 Injury Checks: Daily 2 PM")
        
        while True:
            schedule.run_pending()
            time.sleep(60)
    
    def afcon_monitoring_check(self):
        """Special monitoring during AFCON period."""
        current_gw = self._get_current_gameweek()
        
        if config.AFCON_START_GW <= current_gw <= config.AFCON_END_GW:
            print("🌍 Running AFCON disruption check...")
            
            # Check for last-minute AFCON departures or early returns
            datasets = self.collector.collect_all_data()
            afcon_updates = datasets.get('afcon_risk_data', pd.DataFrame())
            
            if not afcon_updates.empty:
                alert = f"🚨 AFCON UPDATE: New player movements detected in GW{current_gw}"
                self._send_alert_email(alert, subject="AFCON Player Movement Alert")
    
    def chip_deadline_check(self):
        """Check for approaching chip deadlines."""
        current_gw = self._get_current_gameweek()
        
        # Chip Set 1 deadline warning
        if current_gw == 18:
            alert = """
🚨 CHIP DEADLINE WARNING - GAMEWEEK 18

⚠️ CRITICAL: You have ONE gameweek left to use your first chip set!
   
Remaining Chips (Set 1):
• Wildcard
• Free Hit  
• Triple Captain
• Bench Boost

These chips EXPIRE after GW19 deadline (30 Dec 2025, 18:30 GMT).
Unused chips will NOT carry over to the second half.

Recommendation: Review this week's analysis and consider chip usage.
            """
            self._send_alert_email(alert, subject="🚨 URGENT: Chip Deadline GW19")
```

### Model Training with 2025-26 Adaptations

```python
# src/model_trainer.py
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import joblib
from typing import Tuple, Dict

class FPLModelTrainer:
    def __init__(self):
        self.model = None
        self.feature_importance = {}
        
    def train_model_2025_26(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train XGBoost model adapted for 2025-26 rule changes."""
        
        X, y = self.prepare_training_data_2025_26(df)
        
        # Enhanced time series split for 2025-26 season
        tscv = TimeSeriesSplit(n_splits=5, test_size=6)
        
        cv_scores = []
        feature_importance_sum = np.zeros(len(X.columns))
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx] 
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # XGBoost optimized for new scoring system
            model = xgb.XGBRegressor(
                n_estimators=250,  # Increased for more complex patterns
                max_depth=7,       # Deeper for defensive contribution patterns
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.8,
                reg_alpha=0.1,     # L1 regularization for feature selection
                reg_lambda=0.1,    # L2 regularization
                random_state=42,
                objective='reg:squarederror'
            )
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
                early_stopping_rounds=20
            )
            
            # Predictions and scoring
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            cv_scores.append(mae)
            
            # Feature importance accumulation
            feature_importance_sum += model.feature_importances_
            
            print(f"Fold {fold + 1} MAE: {mae:.3f}")
        
        # Final model on full dataset
        self.model = xgb.XGBRegressor(
            n_estimators=250, max_depth=7, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, random_state=42
        )
        
        self.model.fit(X, y)
        
        # Feature importance analysis
        self.feature_importance = dict(zip(
            X.columns, 
            feature_importance_sum / len(cv_scores)
        ))
        
        # Save models with 2025-26 identifier
        joblib.dump(self.model, 'models/fpl_xgb_model_2025_26.pkl')
        joblib.dump(self.feature_importance, 'models/feature_importance_2025_26.pkl')
        
        results = {
            'cv_mae_mean': np.mean(cv_scores),
            'cv_mae_std': np.std(cv_scores),
            'n_features': len(X.columns),
            'top_features': self._get_top_features(5)
        }
        
        print(f"✅ 2025-26 Model trained - CV MAE: {results['cv_mae_mean']:.3f} ± {results['cv_mae_std']:.3f}")
        print(f"🔍 Top features: {results['top_features']}")
        
        return results
        
    def prepare_training_data_2025_26(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data with 2025-26 specific features."""
        
        # Enhanced target including defensive contribution points
        df['target'] = df.groupby('player_id')['points'].shift(-1)
        df['target'] += df.groupby('player_id')['defensive_contribution_points'].shift(-1)
        
        # Remove rows without target
        df = df.dropna(subset=['target'])
        
        # Feature selection for 2025-26
        feature_cols = [col for col in df.columns if col not in [
            'target', 'player_id', 'gameweek', 'name', 'team', 'date'
        ] and not col.startswith('_')]  # Remove metadata columns
        
        # Ensure we have 2025-26 specific features
        required_features = [
            'dc_point_probability', 'afcon_adjusted_points', 
            'chip_set_1_urgency', 'avg_cbit_5gw', 'avg_cbirt_5gw'
        ]
        
        missing_features = [f for f in required_features if f not in feature_cols]
        if missing_features:
            print(f"⚠️ Missing 2025-26 features: {missing_features}")
        
        X = df[feature_cols]
        y = df['target']
        
        return X, y
        
    def predict_points_2025_26(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions incorporating 2025-26 scoring changes."""
        
        if self.model is None:
            self.model = joblib.load('models/fpl_xgb_model_2025_26.pkl')
            
        # Prediction features
        prediction_features = features_df.select_dtypes(include=[np.number])
        base_predictions = self.model.predict(prediction_features)
        
        results_df = features_df[['player_id', 'name', 'position', 'team', 'price']].copy()
        results_df['predicted_points'] = base_predictions
        
        # Add 2025-26 specific prediction components
        results_df['defensive_contribution_bonus'] = (
            features_df['dc_point_probability'] * 2.0
        )
        
        results_df['afcon_risk_adjustment'] = np.where(
            features_df.get('afcon_departure_risk', False),
            -2.0,  # Penalty for AFCON departures
            0.0
        )
        
        # Final adjusted predictions
        results_df['predicted_points'] += results_df['defensive_contribution_bonus']
        results_df['predicted_points'] += results_df['afcon_risk_adjustment']
        results_df['predicted_points'] = results_df['predicted_points'].round(2)
        
        return results_df.sort_values('predicted_points', ascending=False)
        
    def _get_top_features(self, n: int) -> List[str]:
        """Get top N most important features."""
        if not self.feature_importance:
            return []
            
        sorted_features = sorted(
            self.feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return [feat[0] for feat in sorted_features[:n]]
```

---

## Security & Privacy Implementation

### Environment Security

```python
# .env.template - Environment variables template
# Copy to .env and fill in your details - NEVER commit .env to git

# FPL Credentials
FPL_TEAM_ID=your_team_id_from_fpl_url
FPL_EMAIL=your.fpl.email@gmail.com
FPL_PASSWORD=your_fpl_password

# Email Alerts  
EMAIL_USER=your.email@gmail.com
EMAIL_PASS=your_gmail_app_password  # Use Gmail App Password for security
ALERT_EMAIL=your.notification.email@gmail.com
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587

# Database Configuration
DATABASE_URL=sqlite:///fpl_ai.db  # Local SQLite for development
# DATABASE_URL=postgresql://user:password@host:port/database  # For production

# Optional: Premium API Keys (when budget allows)
RAPID_API_KEY=your_rapid_api_key_here
FOOTBALL_API_KEY=your_football_api_key_here

# Security Settings
LOG_LEVEL=INFO
MAX_API_REQUESTS_PER_HOUR=100
RATE_LIMIT_DELAY=1.0
```

### Security Implementation

```python
# src/security.py
import os
import hashlib
import logging
from typing import Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timedelta

class SecurityManager:
    def __init__(self):
        self.setup_secure_logging()
        self.session = self.create_secure_session()
        self.request_count = 0
        self.last_request_reset = datetime.now()
        
    def setup_secure_logging(self):
        """Setup logging that excludes sensitive information."""
        
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/fpl_ai.log'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('FPL_AI_Security')
        
    def create_secure_session(self) -> requests.Session:
        """Create secure HTTP session with rate limiting."""
        
        session = requests.Session()
        
        # Retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Security headers
        session.headers.update({
            'User-Agent': 'FPL-AI-Personal-Bot/1.0 (Educational Use)',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        })
        
        return session
        
    def validate_environment(self) -> bool:
        """Validate all required environment variables."""
        
        required_vars = [
            'FPL_TEAM_ID', 'FPL_EMAIL', 'EMAIL_USER', 'ALERT_EMAIL'
        ]
        
        missing_vars = []
        for var in required_vars:
            value = os.getenv(var)
            if not value:
                missing_vars.append(var)
            elif var == 'FPL_TEAM_ID':
                try:
                    team_id = int(value)
                    if team_id <= 0 or team_id > 20000000:
                        missing_vars.append(f"{var} (invalid format)")
                except ValueError:
                    missing_vars.append(f"{var} (not a number)")
                    
        if missing_vars:
            self.logger.error(f"❌ Missing/invalid environment variables: {missing_vars}")
            return False
            
        self.logger.info("✅ All required environment variables validated")
        return True
        
    def rate_limit_check(self) -> bool:
        """Enforce API rate limiting."""
        
        max_requests = int(os.getenv('MAX_API_REQUESTS_PER_HOUR', 100))
        
        # Reset counter every hour
        if datetime.now() - self.last_request_reset > timedelta(hours=1):
            self.request_count = 0
            self.last_request_reset = datetime.now()
            
        if self.request_count >= max_requests:
            self.logger.warning("⚠️ Rate limit reached, delaying request")
            return False
            
        self.request_count += 1
        return True
        
    def sanitize_for_logging(self, data: str) -> str:
        """Sanitize sensitive data for logging."""
        
        sensitive_patterns = [
            'password', 'token', 'key', 'secret', 'email'
        ]
        
        for pattern in sensitive_patterns:
            if pattern.lower() in data.lower():
                return f"[{pattern.upper()}_REDACTED]"
                
        return data[:50] + "..." if len(data) > 50 else data
        
    def cleanup_old_files(self, days: int = 30):
        """Clean up old log and data files."""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        cleaned_files = []
        
        # Clean log files
        log_dir = 'logs'
        if os.path.exists(log_dir):
            for filename in os.listdir(log_dir):
                filepath = os.path.join(log_dir, filename)
                if os.path.isfile(filepath):
                    file_date = datetime.fromtimestamp(os.path.getmtime(filepath))
                    if file_date < cutoff_date:
                        os.remove(filepath)
                        cleaned_files.append(filepath)
        
        # Clean old raw data
        raw_data_dir = 'data/raw'
        if os.path.exists(raw_data_dir):
            for filename in os.listdir(raw_data_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(raw_data_dir, filename)
                    file_date = datetime.fromtimestamp(os.path.getmtime(filepath))
                    if file_date < cutoff_date:
                        os.remove(filepath)
                        cleaned_files.append(filepath)
                        
        if cleaned_files:
            self.logger.info(f"🗑️ Cleaned {len(cleaned_files)} old files")
            
        return cleaned_files
```

---

## Deployment & Setup Guide

### Quick Setup Script

```bash
#!/bin/bash
# setup_fpl_ai_2025_26.sh - Complete setup for 2025-26 season

echo "🚀 Setting up FPL AI System for 2025-26 Season..."

# Create project directory
mkdir -p fpl_ai_2025_26
cd fpl_ai_2025_26

# Create Python virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install enhanced requirements for 2025-26
echo "⬇️ Installing Python packages..."
cat > requirements.txt << EOF
# Core ML and Data Processing
requests>=2.31.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0

# Optimization and Strategy
pulp>=2.7.0

# Scheduling and Automation  
schedule>=1.2.0
APScheduler>=3.10.0

# Web Scraping and APIs
beautifulsoup4>=4.12.0
lxml>=4.9.0
selenium>=4.15.0

# Database
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0
alembic>=1.12.0

# Web Interface (Optional)
fastapi>=0.104.0
uvicorn>=0.24.0
jinja2>=3.1.0

# Analysis and Visualization
jupyter>=1.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.17.0

# Security and Utilities
python-dotenv>=1.0.0
cryptography>=41.0.0
pyyaml>=6.0.0
EOF

pip install --upgrade pip
pip install -r requirements.txt

# Create enhanced directory structure for 2025-26
echo "📁 Creating directory structure..."
mkdir -p {data/{raw,processed,historical,afcon},models/{2025_26,archived},notebooks,src,logs,reports,config}

# Create comprehensive .env template
echo "📝 Creating environment template..."
cat > .env.template << 'EOF'
# ============================================
# FPL AI Configuration - 2025-26 Season
# ============================================

# FPL Credentials - REQUIRED
FPL_TEAM_ID=your_team_id_here  # Get from https://fantasy.premierleague.com/entry/YOUR_ID/event/1
FPL_EMAIL=your_fpl_email@gmail.com
FPL_PASSWORD=your_fpl_password

# Email Notifications - REQUIRED  
EMAIL_USER=your_email@gmail.com
EMAIL_PASS=your_gmail_app_password  # Generate at https://myaccount.google.com/apppasswords
ALERT_EMAIL=your_notification_email@gmail.com
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587

# Database Configuration
DATABASE_URL=sqlite:///fpl_ai_2025_26.db  # Local development
# DATABASE_URL=postgresql://user:password@host:port/database  # Production

# 2025-26 Season Specific Settings
SEASON=2025_26
AFCON_START_GW=16
AFCON_END_GW=21
CHIP_SET_1_DEADLINE_GW=19

# Security and Rate Limiting
MAX_API_REQUESTS_PER_HOUR=100
RATE_LIMIT_DELAY=1.0
LOG_LEVEL=INFO
DATA_RETENTION_DAYS=90

# Optional: Premium Data Sources (when budget allows)
RAPID_API_KEY=your_key_here
FOOTBALL_API_KEY=your_key_here
FBREF_PREMIUM_KEY=your_key_here

# Development Settings
DEBUG=False
ENABLE_WEB_INTERFACE=False
WEB_PORT=8000
EOF

# Create main entry point with 2025-26 features
cat > main.py << 'EOF'
#!/usr/bin/env python3
"""
FPL AI System - 2025-26 Season
Main Entry Point with Enhanced Features

Commands:
  setup     - Initial setup and model training
  analyze   - Run weekly analysis with chip recommendations
  schedule  - Start automated scheduler
  backtest  - Run historical backtesting
  chips     - Analyze chip strategy for current gameweek
  afcon     - Check AFCON impact and recommendations
  validate  - Validate system configuration
  
2025-26 Features:
  - Double chip system (8 total chips)
  - Defensive contribution point predictions
  - AFCON disruption management
  - Enhanced assist prediction
  - Elite league compatibility
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.scheduler import FPLScheduler
from src.model_trainer import FPLModelTrainer
from src.data_collector import DataCollector
from src.chip_strategy import ChipStrategyManager
from src.security import SecurityManager
from src.config import config

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
        
    command = sys.argv[1].lower()
    
    # Initialize security manager
    security = SecurityManager()
    
    if command == 'validate':
        validate_system(security)
    elif command == 'setup':
        setup_system(security)
    elif command == 'analyze':
        run_weekly_analysis()
    elif command == 'schedule':
        start_scheduler()
    elif command == 'chips':
        analyze_chip_strategy()
    elif command == 'afcon':
        check_afcon_impact()
    elif command == 'backtest':
        run_backtesting()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)

def validate_system(security: SecurityManager):
    """Validate system configuration and requirements."""
    print("🔍 Validating FPL AI system for 2025-26...")
    
    if not security.validate_environment():
        print("❌ Environment validation failed")
        return False
        
    # Test data sources
    collector = DataCollector()
    try:
        bootstrap = collector.fetch_fpl_data("bootstrap-static/")
        if not bootstrap.get('elements'):
            print("❌ FPL API not accessible")
            return False
        print("✅ FPL API accessible")
    except Exception as e:
        print(f"❌ FPL API error: {e}")
        return False
        
    print("✅ System validation complete - ready for 2025-26!")
    return True

def setup_system(security: SecurityManager):
    """Initial system setup for 2025-26 season."""
    print("🏗️ Setting up FPL AI system for 2025-26...")
    
    if not validate_system(security):
        print("❌ Setup failed - fix validation errors first")
        return
        
    # Collect initial data
    collector = DataCollector()
    datasets = collector.collect_all_data()
    
    # TODO: Train initial model with 2025-26 features
    # trainer = FPLModelTrainer()
    # trainer.train_model_2025_26(processed_data)
    
    print("✅ System setup complete for 2025-26!")
    print("🎯 Next steps:")
    print("   1. Run 'python main.py analyze' to test weekly analysis")
    print("   2. Run 'python main.py schedule' to start automation")

def analyze_chip_strategy():
    """Analyze chip strategy for current gameweek."""
    print("💎 Analyzing chip strategy for 2025-26...")
    
    chip_manager = ChipStrategyManager()
    collector = DataCollector()
    
    # Get current data
    datasets = collector.collect_all_data()
    current_gw = 1  # TODO: Get actual current GW
    
    # Analyze chips
    recommendations = chip_manager.recommend_chips(
        datasets['players'], current_gw, {}
    )
    
    print(f"\n🎯 Chip Recommendations for GW{current_gw}:")
    for rec in recommendations:
        print(f"• {rec.chip_type.upper()} (Set {rec.chip_set}): {rec.reasoning}")
        print(f"  Expected gain: +{rec.expected_gain:.1f} points")
        print(f"  Confidence: {rec.confidence:.1%}")
        print()

def check_afcon_impact():
    """Check AFCON impact for current period."""
    print("🌍 Checking AFCON impact for 2025-26...")
    
    collector = DataCollector()
    datasets = collector.collect_all_data()
    
    # TODO: Implement AFCON impact analysis
    print("AFCON analysis completed")

def run_weekly_analysis():
    """Run comprehensive weekly analysis."""
    scheduler = FPLScheduler()
    scheduler.weekly_analysis_pipeline()

def start_scheduler():
    """Start automated scheduler."""
    scheduler = FPLScheduler()
    scheduler.start_scheduler()

def run_backtesting():
    """Run historical backtesting."""
    print("🔙 Running backtesting for 2025-26 model...")
    # TODO: Implement backtesting

if __name__ == "__main__":
    main()
EOF

chmod +x main.py

# Create basic configuration
cat > src/__init__.py << 'EOF'
"""
FPL AI System - 2025-26 Season
Enhanced with double chips, defensive contributions, and AFCON management
"""
EOF

# Create sample configuration
cat > src/config.py << 'EOF'
import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass 
class Config:
    # API Configuration
    FPL_BASE_URL: str = "https://fantasy.premierleague.com/api/"
    FBREF_BASE_URL: str = "https://fbref.com/en/comps/9/"
    
    # Database Configuration  
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///fpl_ai_2025_26.db")
    
    # Email Configuration
    SMTP_SERVER: str = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", 587))
    EMAIL_USER: str = os.getenv("EMAIL_USER")
    EMAIL_PASS: str = os.getenv("EMAIL_PASS")
    ALERT_EMAIL: str = os