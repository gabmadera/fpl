# Maximum Accuracy FPL System 🎯

**Advanced ML-Powered FPL Prediction System targeting 70-75%+ accuracy**

A comprehensive, production-ready Fantasy Premier League prediction system that combines cutting-edge machine learning, aggressive optimization strategies, and automated retraining to achieve maximum prediction accuracy.

## 🚀 Key Features

### **Maximum Accuracy Engine**
- **Target**: 70-75%+ prediction accuracy (baseline: 53.7%)
- **Ensemble Models**: XGBoost + LightGBM + Random Forest with dynamic weighting
- **Advanced Features**: 150+ engineered features including xG/xA, fixture analysis, form trends
- **Real-time Optimization**: Dynamic model weighting based on recent performance

### **Automated Weekly Retraining**
- **Smart Monitoring**: Automatically detects completed gameweeks
- **Immediate Response**: Triggers retraining 2 hours after gameweek completion
- **Intelligent Scheduling**: Background monitoring with fault tolerance
- **Performance Tracking**: Accuracy improvement tracking and alerts

### **Aggressive Strategy Engine**
- **Differential Picks**: Identifies low-ownership gems (≤5% ownership for maximum strategy)
- **Captain Optimization**: Advanced captaincy selection with risk/reward analysis
- **Value Discovery**: Budget player identification and premium justification
- **Risk Management**: Configurable risk tolerance with 4 strategy levels

### **Real-time Monitoring Dashboard**
- **Live Accuracy Tracking**: Real-time accuracy metrics and trends
- **Performance Alerts**: Automated alerts for accuracy drops or improvements
- **Visual Analytics**: Interactive charts and performance visualizations
- **Target Progress**: Timeline tracking toward 75% accuracy goal

## 📊 Strategy Levels

| Strategy | Target Accuracy | Risk Level | Differential Threshold | Features |
|----------|----------------|------------|----------------------|----------|
| **Conservative** | 55-60% | Low | 25% ownership | Steady, reliable picks |
| **Balanced** | 60-65% | Moderate | 15% ownership | Balanced risk/reward |
| **Aggressive** | 65-70% | High | 10% ownership | Higher upside potential |
| **Maximum** | 70-75%+ | Very High | 5% ownership | All-out accuracy attack |

## 🛠 Installation & Setup

### Prerequisites
```bash
# Python 3.8+
pip install pandas numpy scikit-learn xgboost lightgbm
pip install fastapi uvicorn plotly
pip install requests beautifulsoup4 joblib
```

### Quick Start
```bash
# Clone the repository
git clone <repository-url>
cd FPL

# Install dependencies
pip install -r requirements.txt

# Run the complete system
python run_maximum_accuracy_system.py --mode full --strategy maximum

# Or run specific modes
python run_maximum_accuracy_system.py --mode predict --strategy aggressive
python run_maximum_accuracy_system.py --mode serve --port 8000
```

## 🎮 Usage Modes

### 1. **Web Dashboard** (Recommended)
```bash
python run_maximum_accuracy_system.py --mode serve --port 8000
```
- Full web interface at `http://localhost:8000`
- Real-time predictions and monitoring
- Interactive controls and visualizations
- API documentation at `/docs`

### 2. **Prediction Generation**
```bash
python run_maximum_accuracy_system.py --mode predict --strategy maximum
```
- Generates optimized predictions for current gameweek
- Saves results to `data/predictions/maximum_accuracy_predictions.csv`
- Shows top 20 recommendations with strategy flags

### 3. **Model Retraining**
```bash
python run_maximum_accuracy_system.py --mode retrain
```
- Forces immediate model retraining
- Updates ensemble weights based on recent performance
- Shows accuracy improvement metrics

### 4. **Monitoring Only**
```bash
python run_maximum_accuracy_system.py --mode monitor
```
- Starts automated monitoring without web interface
- Continuously checks for gameweek completion
- Triggers automatic retraining when needed

### 5. **Full System**
```bash
python run_maximum_accuracy_system.py --mode full --strategy maximum
```
- Complete system with all components active
- Web dashboard + automated monitoring + retraining
- Production-ready deployment mode

## 📈 Accuracy Timeline

The system follows an aggressive improvement schedule:

| Weeks | Target Accuracy | Focus |
|-------|----------------|--------|
| 1-2 | 55-58% | Initial system improvements |
| 3-5 | 60-65% | Model adaptation and optimization |
| 6-10 | 65-70% | Target achievement phase |
| 10+ | 70-75%+ | Consistent excellence maintenance |

## 🎯 Core Components

### **MaximumAccuracySystem**
```python
from src.maximum_accuracy_system import MaximumAccuracySystem

# Initialize with maximum accuracy configuration
system = MaximumAccuracySystem()

# Generate optimized predictions
predictions = system.generate_maximum_accuracy_predictions()

# Automatic retraining
result = system.automatic_retrain(gameweek=10)
```

### **AggressiveAccuracyOptimizer**
```python
from src.aggressive_accuracy_optimizer import AggressiveAccuracyOptimizer, OptimizationStrategy

# Initialize with maximum aggression
optimizer = AggressiveAccuracyOptimizer(OptimizationStrategy.MAXIMUM)

# Optimize predictions
optimized = optimizer.optimize_predictions(predictions_df)

# Captain optimization
captains = optimizer.optimize_captaincy_selection(predictions_df)

# Differential picks
differentials = optimizer.optimize_differential_picks(predictions_df)
```

### **AutomatedRetrainingScheduler**
```python
from src.automated_retraining_scheduler_v2 import AutomatedRetrainingScheduler

# Start automated monitoring
scheduler = AutomatedRetrainingScheduler()
scheduler.start_monitoring()

# Force retraining
event = scheduler.force_retrain(gameweek=10)

# Check status
status = scheduler.get_status()
```

### **AccuracyMonitoringDashboard**
```python
from src.accuracy_monitoring_dashboard import AccuracyMonitoringDashboard

# Initialize dashboard
dashboard = AccuracyMonitoringDashboard()

# Capture accuracy snapshot
snapshot = dashboard.capture_accuracy_snapshot(predictions_df, actual_results_df)

# Generate dashboard data
dashboard_data = dashboard.generate_dashboard_data()

# Create visualizations
charts = dashboard.create_accuracy_visualizations()
```

## 🌐 API Endpoints

### **Predictions**
- `GET /predictions/maximum-accuracy` - Get optimized predictions
- `GET /predictions/captain-optimization` - Captain recommendations
- `GET /predictions/differentials` - Differential picks
- `GET /predictions/team-optimization` - Optimal team selection

### **Monitoring**
- `GET /monitoring/accuracy-dashboard` - Dashboard data
- `GET /monitoring/accuracy-visualizations` - Performance charts
- `GET /accuracy/targets` - Target progress tracking

### **Automation**
- `POST /retraining/trigger` - Manual retraining trigger
- `GET /retraining/status` - Scheduler status
- `GET /retraining/history` - Retraining history
- `POST /automation/start` - Start automated system
- `POST /automation/stop` - Stop automated system

### **Configuration**
- `POST /optimization/strategy` - Set optimization strategy
- `GET /system/status` - Complete system status

## 🔧 Configuration

### **Model Configuration**
```python
from src.maximum_accuracy_system import ModelConfig, AccuracyTarget

config = ModelConfig(
    target_accuracy=AccuracyTarget.MAXIMUM,
    xgb_weight=0.4,
    lgb_weight=0.35,
    rf_weight=0.25,
    use_advanced_features=True,
    feature_selection_k=150,
    differential_threshold=0.05,  # 5% for maximum strategy
    risk_tolerance=0.8,
    captain_confidence_threshold=0.7
)
```

### **Strategy Configuration**
Each strategy has specific configurations optimized for different risk/reward profiles:

```python
# Maximum Strategy Configuration
{
    'differential_threshold': 0.05,    # 5% ownership threshold
    'risk_multiplier': 1.5,           # 50% boost for high-risk picks
    'captain_confidence_min': 0.5,    # Lower confidence requirement
    'value_weight': 0.9,              # Reduced value weighting
    'form_weight': 1.5,               # Increased form importance
    'fixture_weight': 0.8             # Reduced fixture dependence
}
```

## 📊 Performance Metrics

### **Accuracy Tracking**
- **Overall Accuracy**: Primary metric (within ±2 points)
- **Position Accuracy**: GKP, DEF, MID, FWD breakdowns
- **Captain Accuracy**: Success rate of captain recommendations
- **Differential Accuracy**: Low-ownership pick performance
- **Value Pick Accuracy**: Budget player identification

### **Model Performance**
- **Model Confidence**: Ensemble agreement score
- **Prediction Variance**: Uncertainty measurement
- **Ensemble Stability**: Model consistency over time
- **Feature Importance**: Key driver identification

### **System Health**
- **Retraining Success Rate**: Automated retraining reliability
- **Data Quality Score**: Input data completeness
- **Response Time**: Prediction generation speed
- **Uptime**: System availability

## 🚨 Monitoring & Alerts

### **Accuracy Alerts**
- **Critical**: Accuracy < 55% (immediate action required)
- **Warning**: Accuracy < 65% (consider strategy adjustment)
- **Success**: Accuracy ≥ 75% (target achieved)

### **System Alerts**
- **Low Model Confidence**: < 50% (review feature quality)
- **High Prediction Variance**: > 25 points (check stability)
- **Behind Schedule**: > 2 weeks behind target timeline
- **Retraining Failure**: Automated retraining errors

## 🎯 Advanced Features

### **Ensemble Learning**
- **Dynamic Weighting**: Performance-based model weights
- **Cross-Validation**: Time series validation for reliability
- **Feature Selection**: 150+ features reduced to optimal set
- **Hyperparameter Optimization**: Automated parameter tuning

### **Feature Engineering**
- **Advanced xG/xA**: Multiple data source integration
- **Fixture Analysis**: Opposition strength and venue impact
- **Form Metrics**: Multi-timeframe performance analysis
- **Value Indicators**: Price efficiency and ownership analysis

### **Aggressive Strategies**
- **Ultra Differentials**: < 2% ownership targeting
- **Emerging Talent**: Young player identification
- **Momentum Catching**: Form breakout detection
- **Contrarian Value**: Low-owned quality players

## 🔬 Data Sources

### **Primary Sources**
- **FPL Official API**: Live player data and statistics
- **FBRef**: Advanced player statistics and xG/xA data
- **Understat**: Expected goals and shot analysis
- **Alternative Sources**: Backup data providers

### **Enhanced Data**
- **Fixture Difficulty**: Multi-factor opponent analysis
- **Team Strength**: Home/away attack/defense ratings
- **Player News**: Injury and availability updates
- **Historical Performance**: Multi-season trend analysis

## 🏗 Architecture

### **Component Structure**
```
Maximum Accuracy System
├── Core Engine (MaximumAccuracySystem)
├── ML Pipeline (Ensemble Models)
├── Optimization Engine (AggressiveAccuracyOptimizer)
├── Automation Layer (AutomatedRetrainingScheduler)
├── Monitoring Dashboard (AccuracyMonitoringDashboard)
├── Web API (FastAPI Endpoints)
└── Data Sources (FPL API, FBRef, Understat)
```

### **Data Flow**
1. **Data Collection**: Multi-source player and fixture data
2. **Feature Engineering**: 150+ advanced features
3. **Model Training**: Ensemble with dynamic weighting
4. **Optimization**: Strategy-specific adjustments
5. **Monitoring**: Real-time accuracy tracking
6. **Retraining**: Automated weekly improvements

## 🚀 Production Deployment

### **Docker Deployment**
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

EXPOSE 8000
CMD ["python", "run_maximum_accuracy_system.py", "--mode", "full", "--port", "8000"]
```

### **Environment Variables**
```bash
# Optional API keys for enhanced data
export ODDSAPI_KEY="your_odds_api_key"
export API_FOOTBALL_KEY="your_api_football_key"
export TWITTER_API_KEY="your_twitter_key"

# System configuration
export MAX_ACCURACY_STRATEGY="maximum"
export TARGET_ACCURACY="0.75"
export RETRAINING_INTERVAL="weekly"
```

### **Monitoring Setup**
```bash
# Health check endpoint
curl http://localhost:8000/health

# System status
curl http://localhost:8000/system/status

# Start automation
curl -X POST http://localhost:8000/automation/start
```

## 📈 Performance Optimization

### **Caching Strategy**
- **Prediction Cache**: 1-hour cache for predictions
- **Data Cache**: 6-hour cache for FPL data
- **Model Cache**: Persistent model storage
- **Dashboard Cache**: Real-time dashboard optimization

### **Scaling Considerations**
- **Async Processing**: Background retraining
- **Database Optimization**: Efficient data storage
- **API Rate Limiting**: Respectful API usage
- **Resource Management**: Memory and CPU optimization

## 🔧 Troubleshooting

### **Common Issues**

**1. Low Initial Accuracy**
```bash
# Check data quality
python -c "from src.maximum_accuracy_system import MaximumAccuracySystem; print(MaximumAccuracySystem().get_system_status())"

# Force retraining
python run_maximum_accuracy_system.py --mode retrain
```

**2. Retraining Failures**
```bash
# Check scheduler status
curl http://localhost:8000/retraining/status

# Review logs
tail -f logs/automated_retraining.log
```

**3. API Connection Issues**
```bash
# Test FPL API connection
python -c "from src.fpl_client import FPLClient; print(FPLClient().bootstrap_static())"
```

### **Performance Tuning**

**1. Increase Accuracy Target**
```python
# Adjust model configuration
config = ModelConfig(
    target_accuracy=AccuracyTarget.MAXIMUM,
    risk_tolerance=0.9,  # More aggressive
    differential_threshold=0.03  # Even lower ownership
)
```

**2. Feature Engineering**
```python
# Enable all advanced features
config.use_advanced_features = True
config.feature_selection_k = 200  # More features
```

**3. Ensemble Optimization**
```python
# Adjust ensemble weights based on performance
optimizer.dynamic_model_reweighting({
    'xgb': 0.15,  # Lower error = higher weight
    'lgb': 0.18,
    'rf': 0.20
})
```

## 📚 Advanced Usage

### **Custom Strategy Development**
```python
# Create custom optimization strategy
custom_config = {
    'differential_threshold': 0.08,
    'risk_multiplier': 1.2,
    'captain_confidence_min': 0.6,
    'value_weight': 1.0,
    'form_weight': 1.4,
    'fixture_weight': 0.9
}

optimizer = AggressiveAccuracyOptimizer()
optimizer.strategy_configs[OptimizationStrategy.CUSTOM] = custom_config
```

### **Integration with External Tools**
```python
# Export predictions for external analysis
predictions = system.generate_maximum_accuracy_predictions()
predictions.to_csv('fpl_predictions.csv', index=False)

# Integration with team selection tools
from src.team_selector import TeamSelector
selector = TeamSelector()
optimal_team = selector.select_team(predictions, budget=100.0)
```

## 🤝 Contributing

We welcome contributions to improve the Maximum Accuracy System:

1. **Bug Reports**: Submit issues with detailed error logs
2. **Feature Requests**: Propose new optimization strategies
3. **Model Improvements**: Enhanced algorithms and features
4. **Documentation**: Usage examples and tutorials

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **FPL Official API**: Primary data source
- **FBRef & Understat**: Advanced statistics
- **Scikit-learn, XGBoost, LightGBM**: Machine learning frameworks
- **FastAPI**: Web framework
- **Fantasy Football Community**: Inspiration and feedback

---

**Built for maximum FPL accuracy. Deploy with confidence. Achieve 70-75%+ prediction accuracy.** 🎯