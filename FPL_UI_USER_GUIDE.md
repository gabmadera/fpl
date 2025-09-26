# 🚀 FPL AI Dashboard - Complete User Guide

**Professional Fantasy Premier League Prediction System with 80%+ Accuracy Target**

## 🎯 Overview

The FPL AI Dashboard is a comprehensive machine learning system designed to generate optimal Fantasy Premier League team suggestions with advanced AI features including:

- **Enhanced ML Pipeline** with 9 specialized models
- **Real-time FPL data integration** for accurate predictions
- **Backtesting framework** for strategy validation
- **Online learning system** for continuous improvement
- **Bonus point prediction** using XGBoost
- **Player similarity analysis** with embeddings
- **Dynamic ensemble weights** for model optimization
- **Historical accuracy tracking** with real FPL data

---

## 🚀 Getting Started

### Quick Start
```bash
# Start the dashboard
python run_clean_dashboard.py

# Or with custom port
python run_clean_dashboard.py --port 8002

# Run system tests first
python run_clean_dashboard.py --test
```

### Access the Dashboard
- **URL**: http://localhost:8001 (default port)
- **Browser**: Any modern web browser
- **Auto-opens**: Browser opens automatically (disable with `--no-browser`)

---

## 📋 UI Layout & Features

### 1. Header Section
- **FPL AI Dashboard** title with ⚽ emoji
- **Current GW Status**: Shows current/next gameweek info
- Updates automatically from FPL API

### 2. Main Dashboard Grid (5 columns)

#### Column 1: 📋 What To Do
**Real-time action recommendations based on model performance:**
- ⚠️ **Model Needs Improvement**: Shows when accuracy drops
- ✅ **Model Performing Well**: Shows when accuracy is good
- 📊 **Collect More Data**: Initial setup guidance
- **Dynamic suggestions** based on current state
- **Quick action buttons** for immediate tasks

#### Column 2-4: 💡 Current Suggestion (Football Field View)
**Visual team display with enhanced information:**
- **Football field layout** with realistic positioning
- **Player cards** showing:
  - Player name and position
  - Team abbreviation (e.g., MCI, LIV)
  - Next opponent and venue (H/A)
  - Predicted points for current gameweek
- **Captain/Vice-Captain badges** (C/VC)
- **Bench section** with substitute players
- **Team summary** with formation, total cost, predicted points

#### Column 5: 📊 Performance
**Live performance metrics:**
- **Average Accuracy**: Overall model performance
- **GWs Tracked**: Number of gameweeks analyzed
- **Last GW**: Most recent gameweek accuracy
- **Status Badge**: Excellent/Good/Needs Work/Poor

### 3. Transfer Changes Section
**Shows differences from previous gameweek:**
- **➕ Players In**: New additions with position and price
- **➖ Players Out**: Removed players with details
- **Transfer Cost**: Calculated based on free transfers
- **Comparison**: Previous GW → Current GW

### 4. Recent Performance Section
**Last 3 gameweeks detailed view:**
- **GW Number** with predicted vs actual points
- **Accuracy percentage** with color coding:
  - 🟢 Green: 80%+ (Excellent)
  - 🟡 Yellow: 60-79% (Good)
  - 🔴 Red: Below 60% (Needs work)

### 5. Historical Comparisons
**Complete season performance overview:**
- **Performance Overview**: Total GWs tracked, average accuracy
- **Gameweek Results**: Detailed table showing:
  - Formation used
  - Predicted vs actual points
  - Accuracy percentage
  - Point differences
- **Latest Team Details**: Full squad breakdown with actual vs predicted points

---

## 🎯 Available Actions

### Primary Action Buttons (Always Available)

#### 1. 🔄 Generate Team
**Creates new optimal team suggestion**
- **Function**: Runs ML pipeline to generate predictions
- **Output**: Complete 15-player squad with formation
- **Features**: Captain/vice-captain selection, budget optimization
- **API Endpoint**: `/predictions?force_refresh=true`

#### 2. 📊 Collect Results
**Fetches actual FPL results for finished gameweeks**
- **Function**: Gets real player points from FPL API
- **Process**: Compares predicted vs actual performance
- **Updates**: Performance metrics and accuracy scores
- **API Endpoint**: `/collect-real-results-improved/{gameweek}`

#### 3. 🟡 Update Accuracy
**Updates historical data with actual FPL points**
- **Function**: Prompts for gameweek number
- **Process**: Fetches official FPL player points
- **Updates**: Recalculates accuracy with real data
- **API Endpoint**: `/update-accuracy-with-actuals/{gameweek}`

#### 4. 🟠 Verify Points
**Verifies actual FPL points accuracy**
- **Function**: Shows detailed player-by-player verification
- **Features**: Point breakdown, prediction vs reality
- **Verification**: Links to official FPL website
- **API Endpoint**: `/verify-gameweek/{gameweek}`

#### 5. 🟣 View Details
**Refreshes detailed performance view**
- **Function**: Reloads recent performance section
- **Display**: Enhanced recent gameweeks data
- **Purpose**: Quick refresh without full page reload

#### 6. ⚫ Refresh All
**Complete dashboard refresh**
- **Function**: Reloads all dashboard sections
- **Process**: Re-fetches all data from APIs
- **Use case**: When data appears stale or after errors

### Advanced Action Buttons (Context-Aware)

#### 🔧 Improve Model (Shows when accuracy drops)
**Retrains model with actual gameweek results**
- **Trigger**: Appears when model performance is poor
- **Function**: Uses machine learning retraining
- **Process**: Incorporates actual FPL results into model
- **API Endpoint**: `/retrain-model`

### System Status Actions

#### Health Check
- **Endpoint**: `/health`
- **Function**: Basic system health verification
- **Response**: System status confirmation

#### Gameweek Status
- **Endpoint**: `/gameweek-status`
- **Function**: Current FPL gameweek information
- **Caching**: 6-hour cache for optimal performance

---

## 🔧 Advanced Features & Endpoints

### Model Performance Analysis
**Comprehensive model feedback system:**
- **Summary**: `/model-feedback/summary`
- **Analyze**: `/model-feedback/analyze`
- **Apply Improvements**: `/model-feedback/apply-improvements` (POST)

### Player Analysis
**Individual player verification:**
- **Specific Player**: `/verify-points/{gameweek}/{player_id}`
- **Team Verification**: `/verify-gameweek/{gameweek}`
- **Player Points**: `/actual-player-points/{gameweek}`

### Enhanced Predictions
**Enriched team data with opponent information:**
- **Endpoint**: `/enriched-predictions`
- **Features**: Opponent fixtures, venue info, team details

### Transfer Analysis
**Smart transfer recommendations:**
- **Endpoint**: `/transfer-analysis`
- **Features**: Transfer in/out suggestions, cost analysis

### Historical Data
**Complete historical performance:**
- **Endpoint**: `/historical-comparisons`
- **Features**: Season overview, gameweek comparisons
- **Format**: JSON with detailed statistics

---

## 📊 Hidden/Advanced ML Features

### 1. Backtesting Framework
**Validates strategies across multiple gameweeks**
- **Access**: Via Python API (not directly in UI)
- **Function**: Tests prediction accuracy historically
- **Use**: Background validation of model performance

### 2. Bonus Point Predictor
**XGBoost-based bonus point predictions**
- **Integration**: Built into main predictions
- **Models**: Separate predictors for 3, 2, and 1 bonus points
- **Features**: BPS threshold analysis by position

### 3. Online Learning System
**Continuous model improvement**
- **Process**: Updates models with new gameweek data
- **Methods**: SGDRegressor, PassiveAggressiveRegressor
- **Persistence**: Automatic model saving

### 4. Player Embeddings
**Advanced player similarity analysis**
- **Technique**: 32-dimensional embeddings using PCA
- **Function**: Finds similar players for recommendations
- **Clustering**: Position archetypes via K-means

### 5. Dynamic Ensemble Weights
**Adaptive model combination**
- **Process**: Adjusts model weights based on performance
- **Learning**: Online weight updates with exponential moving averages
- **Optimization**: Performance trend analysis

---

## 🎮 Step-by-Step Usage

### First Time Setup
1. **Start the dashboard** with `python run_clean_dashboard.py --test`
2. **Wait for system tests** to complete
3. **Access browser** at http://localhost:8001
4. **Click "Generate Team"** to create first prediction
5. **Review suggestion** in football field layout

### Weekly Workflow
1. **Check current gameweek** status in header
2. **Generate new team** if needed for upcoming GW
3. **After gameweek finishes**:
   - Click **"Collect Results"** to get actual points
   - Use **"Update Accuracy"** to verify with official data
   - Click **"Verify Points"** for detailed validation
4. **Review performance** in Recent Performance section
5. **Improve model** if accuracy drops below target

### Troubleshooting Workflow
1. **Refresh All** if data appears stale
2. **Verify Points** if results seem incorrect
3. **Check browser console** for JavaScript errors
4. **Restart server** if persistent issues
5. **Run with --test flag** to validate system components

---

## 🚨 Error Handling

### Common Issues & Solutions

#### "No predictions generated"
- **Cause**: FPL API unavailable or model failure
- **Solution**: Click "Generate Team" again, check internet connection

#### "No historical data found"
- **Cause**: Fresh installation or cleared data
- **Solution**: Generate predictions first, then collect results

#### "Failed to fetch FPL data"
- **Cause**: FPL API downtime or rate limiting
- **Solution**: Wait and retry, check FPL website status

#### JavaScript errors
- **Cause**: Browser compatibility or network issues
- **Solution**: Refresh page, try different browser, check console

### System Status Indicators
- **🟢 Green loading**: Normal operation
- **🟡 Yellow warning**: Data unavailable but recoverable
- **🔴 Red error**: System issue requiring attention
- **⚫ Gray unknown**: Unknown state, data unavailable

---

## 📈 Performance Expectations

### Accuracy Targets
- **Target**: 80%+ prediction accuracy
- **Good**: 70-79% accuracy
- **Needs Work**: 50-69% accuracy
- **Poor**: Below 50% accuracy

### Response Times
- **Dashboard Load**: 2-5 seconds
- **Generate Team**: 10-30 seconds
- **Collect Results**: 5-15 seconds
- **Historical Data**: 1-3 seconds

### Data Refresh Rates
- **Gameweek Status**: 6 hours cache
- **Team Predictions**: On demand
- **Performance Metrics**: Real-time
- **FPL API Data**: Live when requested

---

## 🔧 Technical Details

### System Requirements
- **Python**: 3.8+ with required packages
- **Browser**: Chrome, Firefox, Safari (modern versions)
- **Internet**: Required for FPL API access
- **Storage**: ~50MB for historical data

### Key Dependencies
- **FastAPI**: Web framework
- **Pandas/NumPy**: Data processing
- **Scikit-learn**: Machine learning
- **XGBoost**: Advanced modeling
- **Requests**: API integration

### Data Sources
- **FPL API**: Official Fantasy Premier League data
- **FBRef**: Advanced football statistics
- **Understat**: Expected goals and assists
- **Historical Data**: Local storage for accuracy tracking

---

## 💡 Tips for Best Results

### Optimization Tips
1. **Generate teams early** in the gameweek
2. **Collect results promptly** after gameweek ends
3. **Retrain model** when accuracy drops
4. **Verify critical gameweeks** manually
5. **Monitor transfer costs** for realistic strategies

### Advanced Usage
1. **Use browser developer tools** for debugging
2. **Monitor API responses** for data validation
3. **Compare with other FPL tools** for verification
4. **Track long-term performance trends**
5. **Experiment with different gameweek strategies**

### Interpretation Guidelines
- **High predicted points**: Strong fixtures, form, or value
- **Captain selection**: Highest expected return player
- **Formation changes**: Adaptive to available players
- **Transfer suggestions**: Based on performance and fixtures
- **Accuracy tracking**: Consider variance in football

---

This guide covers all available functionality in the FPL AI Dashboard. For technical issues or feature requests, check the console output and error logs for detailed troubleshooting information.