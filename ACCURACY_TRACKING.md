# FPL Prediction Accuracy Tracking System

## Overview

This system provides comprehensive tracking and analysis of your FPL AI predictions vs actual gameweek results. It logs team suggestions, compares them against reality, and provides insights for improving prediction accuracy.

## How to Run the Web UI

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Install requirements
pip install -r requirements.txt

# 3. Run the web UI
python -m uvicorn src.web_ui_enhanced:app --host 0.0.0.0 --port 8000 --reload
```

The web UI will be available at http://localhost:8000

## New Web UI Endpoints

### Team Suggestion Logging
- **POST** `/log-team-suggestion/{gameweek}` - Log team suggestion for tracking
- **GET** `/team-history?last_n_gameweeks=5` - View team suggestion history

### Results Collection  
- **POST** `/collect-actual-results/{gameweek}` - Collect actual results for comparison
- **GET** `/gameweek-results/{gameweek}` - View results for specific gameweek

### Accuracy Analysis
- **GET** `/accuracy-report?last_n_gameweeks=10` - Comprehensive accuracy report
- **GET** `/improvement-suggestions` - AI suggestions for improving accuracy

## Command Line Usage

### Log Team Suggestions
```bash
# Log team suggestion for next gameweek
python src/accuracy_workflow.py log-suggestion --gameweek 5

# Log for current gameweek (auto-detected)
python src/accuracy_workflow.py log-suggestion
```

### Collect Actual Results
```bash
# Collect results for completed gameweek
python src/accuracy_workflow.py collect-results --gameweek 4

# Collect for previous gameweek (auto-detected)
python src/accuracy_workflow.py collect-results
```

### Generate Reports
```bash
# Generate accuracy report for last 10 gameweeks
python src/accuracy_workflow.py generate-report --last-n 10

# Quick report for last 5 gameweeks
python src/accuracy_workflow.py generate-report --last-n 5
```

### Auto-Improve Models
```bash
# Automatically retrain models if performance is declining
python src/accuracy_workflow.py auto-improve
```

### Weekly Workflow
```bash
# Run complete weekly workflow (log suggestions, collect results, generate report, improve models)
python src/accuracy_workflow.py weekly-workflow --gameweek 5
```

## Weekly Workflow Process

### Before Each Gameweek
1. **Log Team Suggestion**: Save your AI's team selection and predictions
2. **Review Transfer Suggestions**: Log recommended transfers
3. **Set Captain/Vice**: Record captaincy decisions

### After Each Gameweek
1. **Collect Results**: Gather actual points scored by your team
2. **Compare Performance**: Analyze predictions vs reality
3. **Calculate Metrics**: MAE, correlation, captain accuracy
4. **Generate Report**: Review overall performance trends

### Weekly Improvements
1. **Analyze Accuracy**: Review prediction errors and patterns
2. **Retrain Models**: Auto-retrain if performance declining
3. **Update Features**: Enhance feature engineering based on insights
4. **Optimize Strategy**: Adjust team selection and transfer logic

## Key Metrics Tracked

### Prediction Accuracy
- **Mean Absolute Error (MAE)**: Average difference between predicted and actual points
- **Root Mean Square Error (RMSE)**: Weighted error metric
- **Correlation**: How well predictions correlate with actual results
- **Captain Accuracy**: Specific accuracy for captaincy decisions

### Team Performance  
- **Total Points**: Actual points scored by suggested team
- **Points vs Predicted**: How close actual matched predictions
- **Points vs Optimal**: How close to theoretically best team
- **Transfer Effectiveness**: Whether transfers improved or hurt performance

### Strategic Insights
- **Formation Effectiveness**: Which formations work best
- **Price Point Analysis**: Optimal spending distribution
- **Position-Specific Accuracy**: Which positions predict better
- **Captaincy Success Rate**: Captain choice accuracy over time

## Data Storage

All tracking data is stored in `data/accuracy_tracking/`:
- `gameweek_suggestions.json` - Team suggestions for each gameweek
- `gameweek_results.json` - Actual results and comparisons
- `gw{N}_team_predictions.csv` - Individual player predictions per gameweek
- Logs are stored in `logs/accuracy_workflow.log`

## Example Weekly Routine

```bash
# Monday (before gameweek deadline)
python src/accuracy_workflow.py log-suggestion --gameweek 8

# Tuesday (after gameweek completes)  
python src/accuracy_workflow.py collect-results --gameweek 7

# Wednesday (analysis day)
python src/accuracy_workflow.py generate-report --last-n 5
python src/accuracy_workflow.py auto-improve

# Or run everything at once
python src/accuracy_workflow.py weekly-workflow --gameweek 8
```

## Integration with Existing System

The tracking system integrates with your existing ML pipeline:

- **MLPipeline**: Uses current predictions for team suggestions
- **TeamSelector**: Leverages optimal team selection logic  
- **WeeklyMLRetrainer**: Connects with model retraining system
- **Web UI**: Provides interactive visualization of performance

## Performance Improvement Process

1. **Monitor Accuracy**: Track MAE and correlation trends
2. **Identify Patterns**: Find specific areas where predictions fail
3. **Retrain Models**: Auto-retrain when performance degrades
4. **Enhance Features**: Add new features based on insights
5. **Optimize Strategy**: Adjust selection logic based on results

## Success Metrics

Track these KPIs to measure system effectiveness:

- **Weekly Prediction Accuracy** > 80%
- **Captain Choice Success** > 70% 
- **Points vs Optimal Gap** < 15 points/week
- **Transfer Success Rate** > 50% (net positive)
- **Season-long Points Total** tracking vs benchmarks

This system ensures your FPL AI continuously learns and improves its predictions based on real-world performance data.