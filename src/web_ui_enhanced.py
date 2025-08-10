from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
import pandas as pd
from pathlib import Path
import numpy as np
from .ml_pipeline import MLPipeline
from .team_optimizer import TeamOptimizer
from .chip_strategy import ChipStrategyManager
from .fpl_client import FPLClient
from .scheduler import FPLScheduler
from .config import config
from .settings import load_exclusions
from .team_selector import TeamSelector


app = FastAPI(title="FPL AI Dashboard", version="0.1")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/predictions")
def predictions(limit: int = 0) -> list[dict]:
    path = Path("data/processed/predictions_current.csv")
    if not path.exists():
        ml = MLPipeline()
        df = ml.predict_current()
    else:
        df = pd.read_csv(path)
    
    # Enhanced FPL status/chance merging - fix column conflicts
    try:
        fpl = FPLClient()
        bs = fpl.bootstrap_static()
        elems = pd.DataFrame(bs.get("elements", []))
        teams = pd.DataFrame(bs.get("teams", []))
        
        if not elems.empty:
            # Only merge essential columns to avoid conflicts
            elems_subset = elems.rename(columns={
                "id": "player_id",
                "status": "current_fpl_status", 
                "chance_of_playing_next_round": "current_chance_next",
                "selected_by_percent": "current_ownership",
                "web_name": "current_web_name"
            })[["player_id", "current_fpl_status", "current_chance_next", "current_ownership", "current_web_name"]]
            
            df = df.merge(elems_subset, on="player_id", how="left")
            
            # Update/override the status columns with current FPL data
            df['fpl_status'] = df['current_fpl_status'].fillna(df.get('fpl_status', 'a'))
            # For chance_next, if FPL API returns None (pre-season), use default value of 100 for available players
            df['chance_next'] = df['current_chance_next'].fillna(100)
            df['ownership'] = df['current_ownership'].fillna(df.get('selected_by_percent', 5.0))
            
            # Clean up temporary columns
            df = df.drop(columns=['current_fpl_status', 'current_chance_next', 'current_ownership', 'current_web_name'], errors='ignore')
                
        if not teams.empty and 'team_id' in df.columns:
            teams_subset = teams.rename(columns={"id": "team_id", "short_name": "team_short"})[["team_id", "team_short"]]
            if "team_short" in teams_subset.columns:
                teams_subset["team_short"] = teams_subset["team_short"].astype(str).str.upper()
            # Use a different merge strategy to avoid overwriting existing team_short
            df = df.merge(teams_subset, on="team_id", how="left", suffixes=('', '_new'))
            # Fill missing team_short with new data
            if 'team_short_new' in df.columns:
                df['team_short'] = df['team_short'].fillna(df['team_short_new'])
                df = df.drop(columns=['team_short_new'], errors='ignore')
                
    except Exception as e:
        print(f"Enhanced data merging failed: {e}")
        # Ensure essential columns exist with defaults
        if 'fpl_status' not in df.columns:
            df['fpl_status'] = 'a'
        if 'chance_next' not in df.columns:
            df['chance_next'] = 100

    # Apply limit
    if limit and limit > 0:
        df = df.head(limit)
    
    return df.to_dict(orient="records")


@app.get("/summary")
def summary() -> dict:
    """Enhanced summary endpoint"""
    path = Path("data/processed/predictions_current.csv")
    if not path.exists():
        ml = MLPipeline()
        preds = ml.predict_current()
    else:
        preds = pd.read_csv(path)

    try:
        fpl = FPLClient()
        current_gw = fpl.current_gameweek()
        bs = fpl.bootstrap_static()
        teams = pd.DataFrame(bs.get("teams", []))
        team_map = {}
        if not teams.empty:
            for _, row in teams.iterrows():
                tid = int(row.get("id")) if row.get("id") is not None else None
                if tid is not None:
                    team_map[tid] = row.get("short_name") or row.get("name")
    except Exception:
        current_gw = 1
        team_map = {}

    return {
        "gameweek": current_gw,
        "teams": team_map,
        "total_players": len(preds),
        "avg_prediction": preds["predicted_points"].mean() if not preds.empty else 0
    }


@app.get("/team")
def get_optimal_team(gameweek: int = 1) -> dict:
    """Get optimal team selection with formation and captaincy"""
    try:
        path = Path("data/processed/predictions_current.csv")
        if not path.exists():
            ml = MLPipeline()
            df = ml.predict_current()
        else:
            df = pd.read_csv(path)
        
        selector = TeamSelector()
        optimal_team = selector.select_optimal_team(df, gameweek)
        
        return {
            "starters": optimal_team.starters,
            "bench": optimal_team.bench,
            "captain_id": optimal_team.captain_id,
            "vice_captain_id": optimal_team.vice_captain_id,
            "formation": optimal_team.formation,
            "total_cost": round(optimal_team.total_cost, 1),
            "predicted_points": round(optimal_team.predicted_points, 1),
            "chip_recommendation": optimal_team.chip_recommendation,
            "budget_remaining": round(100.0 - optimal_team.total_cost, 1)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/refresh")
def refresh() -> dict:
    """Refresh data"""
    try:
        ml = MLPipeline()
        ml.pipe.collect_fpl_snapshots()
        train_info = ml.train()
        predictions = ml.predict_current()
        
        return {
            "status": "ok",
            "train": train_info,
            "predictions_count": len(predictions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/gameweek-results/{gameweek}")
def get_gameweek_results(gameweek: int) -> dict:
    """Get actual vs predicted results for a specific gameweek"""
    try:
        from .weekly_retrainer import WeeklyMLRetrainer
        retrainer = WeeklyMLRetrainer()
        
        # Get actual results
        results_df = retrainer.collect_gameweek_results(gameweek)
        if results_df is None:
            return {"error": f"No results available for GW{gameweek}"}
        
        # Get predictions for that gameweek
        pred_file = Path(f"data/weekly_performance/gw{gameweek}_predictions.csv")
        if not pred_file.exists():
            return {"error": f"No predictions found for GW{gameweek}"}
        
        predictions_df = pd.read_csv(pred_file)
        
        # Merge results
        merged = results_df.merge(predictions_df, on="player_id", how="inner")
        
        # Calculate metrics
        if not merged.empty:
            mae = float(np.abs(merged["actual_points"] - merged["predicted_points"]).mean())
            total_players = len(merged)
            
            # Top performers comparison
            top_actual = merged.nlargest(10, "actual_points")[["player_id", "name", "actual_points", "predicted_points"]].to_dict("records")
            top_predicted = merged.nlargest(10, "predicted_points")[["player_id", "name", "actual_points", "predicted_points"]].to_dict("records")
            
            return {
                "gameweek": gameweek,
                "total_players": total_players,
                "mae": mae,
                "top_actual_performers": top_actual,
                "top_predicted_performers": top_predicted,
                "results_available": True
            }
        else:
            return {"error": "No matching data between predictions and results"}
            
    except Exception as e:
        return {"error": str(e)}


@app.get("/performance-history")
def get_performance_history() -> dict:
    """Get historical prediction performance across gameweeks"""
    try:
        from .weekly_retrainer import WeeklyMLRetrainer
        retrainer = WeeklyMLRetrainer()
        history = retrainer.get_performance_history()
        
        return {
            "history": history,
            "total_gameweeks": len(history)
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/track-gameweek/{gameweek}")  
def track_gameweek_performance(gameweek: int) -> dict:
    """Manually trigger collection of gameweek results and evaluation"""
    try:
        from .weekly_retrainer import WeeklyMLRetrainer
        retrainer = WeeklyMLRetrainer()
        
        # Collect results
        results_df = retrainer.collect_gameweek_results(gameweek)
        if results_df is None:
            return {"error": f"Could not collect results for GW{gameweek}"}
        
        # Evaluate predictions if we have them
        evaluation = retrainer.evaluate_previous_predictions(gameweek)
        
        return {
            "status": "success",
            "gameweek": gameweek,
            "results_collected": len(results_df),
            "evaluation": evaluation
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/transfer-suggestions")
def get_transfer_suggestions(current_gameweek: int = 1) -> dict:
    """Get intelligent transfer suggestions for next gameweek"""
    try:
        # Load current predictions
        path = Path("data/processed/predictions_current.csv")
        if not path.exists():
            ml = MLPipeline()
            df = ml.predict_current()
        else:
            df = pd.read_csv(path)
        
        # Get current optimal team as baseline
        selector = TeamSelector()
        current_optimal = selector.select_optimal_team(df, current_gameweek)
        current_team_ids = [p['player_id'] for p in current_optimal.starters + current_optimal.bench]
        
        # Get transfer suggestions
        transfer_suggestions = selector.evaluate_transfers(current_team_ids, df, current_gameweek + 1)
        
        # Weekly strategy insights
        insights = {
            "fixture_analysis": "Analyze upcoming fixtures for transfer timing",
            "price_changes": "Monitor player price changes to maximize team value", 
            "injury_updates": "Stay updated on player fitness and availability",
            "form_analysis": "Consider recent form trends beyond just predictions"
        }
        
        return {
            "current_gameweek": current_gameweek,
            "suggested_transfers": transfer_suggestions[:5],  # Top 5 suggestions
            "transfer_strategy": "Free transfers available: 1-2 per week",
            "weekly_insights": insights,
            "team_evolution": "Team will be continuously optimized based on form, fixtures, and injuries"
        }
        
    except Exception as e:
        return {"error": str(e)}


@app.post("/retrain-model")
def trigger_model_retraining() -> dict:
    """Manually trigger model retraining and improvement"""
    try:
        from .weekly_retrainer import WeeklyMLRetrainer
        retrainer = WeeklyMLRetrainer()
        
        # Get current gameweek
        fpl = FPLClient()
        bootstrap = fpl.bootstrap_static()
        events = bootstrap.get("events", [])
        
        current_gw = 1
        for event in events:
            if event.get("is_current", False):
                current_gw = event.get("id", 1)
                break
        
        # Perform adaptive retraining
        result = retrainer.adaptive_retrain(current_gw)
        
        return {
            "status": "success",
            "retraining_result": result,
            "message": "Model has been retrained with latest data"
        }
        
    except Exception as e:
        return {"error": str(e)}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Beautiful FPL-style interface"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FPL AI Dashboard - Premium Analytics</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <script src="https://unpkg.com/alpinejs@3.x.x/dist/cdn.min.js" defer></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.tailwindcss.com"></script>
    
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        :root {
            --fpl-purple: #37003c;
            --fpl-green: #00ff87;
            --fpl-pink: #ff0080;
            --fpl-cyan: #00e7ff;
            --fpl-dark: #1a1a2e;
            --gradient-main: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --gradient-card: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
        }
        
        body {
            background: var(--gradient-main);
            background-attachment: fixed;
            min-height: 100vh;
        }
        
        .glass-card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        }
        
        .hero-card {
            background: linear-gradient(135deg, var(--fpl-purple) 0%, #2d1b4e 100%);
            position: relative;
            overflow: hidden;
        }
        
        .hero-card::before {
            content: '';
            position: absolute;
            top: -50%;
            right: -50%;
            width: 100%;
            height: 200%;
            background: radial-gradient(circle, rgba(0, 255, 135, 0.1) 0%, transparent 70%);
            animation: float 6s ease-in-out infinite;
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0px) rotate(0deg); }
            50% { transform: translateY(-20px) rotate(180deg); }
        }
        
        .stat-card {
            background: var(--gradient-card);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        }
        
        .player-card {
            background: white;
            border-left: 4px solid var(--fpl-green);
            transition: all 0.3s ease;
        }
        
        .player-card:hover {
            transform: translateX(8px);
            box-shadow: -8px 8px 20px rgba(0, 0, 0, 0.1);
        }
        
        .position-gkp { background: linear-gradient(135deg, #10b981, #047857); }
        .position-def { background: linear-gradient(135deg, #3b82f6, #1d4ed8); }
        .position-mid { background: linear-gradient(135deg, #f59e0b, #d97706); }
        .position-fwd { background: linear-gradient(135deg, #ef4444, #dc2626); }
        
        .loading-spinner {
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        
        .fade-in {
            animation: fadeIn 0.6s ease-out;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .slide-in {
            animation: slideIn 0.8s ease-out;
        }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateX(-30px); }
            to { opacity: 1; transform: translateX(0); }
        }
        
        .search-input {
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(10px);
            border: 2px solid transparent;
            transition: all 0.3s ease;
        }
        
        .search-input:focus {
            background: white;
            border-color: var(--fpl-green);
            box-shadow: 0 0 20px rgba(0, 255, 135, 0.2);
        }
        
        .btn-primary {
            background: linear-gradient(135deg, var(--fpl-purple), #2d1b4e);
            color: white;
            border: none;
            transition: all 0.3s ease;
        }
        
        .btn-primary:hover {
            background: linear-gradient(135deg, #2d1b4e, var(--fpl-purple));
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(55, 0, 60, 0.3);
        }
        
        .filter-tab {
            background: rgba(255, 255, 255, 0.1);
            border: 2px solid transparent;
            color: rgba(255, 255, 255, 0.8);
            transition: all 0.3s ease;
        }
        
        .filter-tab.active {
            background: var(--fpl-green);
            color: var(--fpl-purple);
            border-color: var(--fpl-green);
            box-shadow: 0 4px 15px rgba(0, 255, 135, 0.3);
        }
        
        .pitch-bg {
            background: linear-gradient(135deg, #16a34a 0%, #15803d 100%);
            position: relative;
        }
        
        .pitch-bg::before {
            content: '';
            position: absolute;
            inset: 0;
            background-image: 
                radial-gradient(circle at 25% 25%, rgba(255,255,255,0.1) 2px, transparent 2px),
                radial-gradient(circle at 75% 75%, rgba(255,255,255,0.1) 1px, transparent 1px);
            background-size: 60px 60px, 30px 30px;
        }
        
        .player-chip {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border: 2px solid rgba(255, 255, 255, 0.3);
            transition: all 0.3s ease;
        }
        
        .player-chip:hover {
            transform: translateY(-5px) scale(1.05);
            box-shadow: 0 15px 30px rgba(0, 0, 0, 0.2);
        }
        
        .captain-chip {
            border-color: var(--fpl-green);
            box-shadow: 0 0 20px rgba(0, 255, 135, 0.4);
        }
        
        .vice-chip {
            border-color: var(--fpl-cyan);
            box-shadow: 0 0 20px rgba(0, 231, 255, 0.4);
        }
    </style>
</head>

<body x-data="fplApp()" x-init="init()" class="min-h-screen">
    <!-- Hero Section -->
    <div class="hero-card rounded-3xl m-6 p-8 text-white relative z-10 fade-in">
        <div class="flex flex-col lg:flex-row justify-between items-start lg:items-center">
            <div class="mb-6 lg:mb-0">
                <h1 class="text-5xl lg:text-6xl font-black mb-4 tracking-tight">
                    <i class="fas fa-futbol mr-4 text-green-400"></i>
                    FPL AI
                </h1>
                <p class="text-2xl text-purple-200 font-medium">Premium Analytics & Intelligence</p>
                <p class="text-purple-300 mt-2">Advanced machine learning predictions for Fantasy Premier League</p>
            </div>
            <div class="text-right">
                <div class="text-4xl font-bold text-green-400" x-text="'GW ' + gameweek"></div>
                <div class="text-purple-200">2025/26 Season</div>
                <div class="mt-4 text-sm text-purple-300">
                    Last updated: <span x-text="lastUpdate"></span>
                </div>
            </div>
        </div>
    </div>

    <!-- Quick Stats -->
    <div class="mx-6 mb-8 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 slide-in">
        <div class="stat-card rounded-2xl p-6 text-center">
            <div class="text-3xl mb-2">📊</div>
            <div class="text-2xl font-bold text-purple-800" x-text="totalPlayers"></div>
            <div class="text-sm text-gray-600">Total Players</div>
        </div>
        
        <div class="stat-card rounded-2xl p-6 text-center">
            <div class="text-3xl mb-2">⭐</div>
            <div class="text-lg font-bold text-purple-800" x-text="topPlayer"></div>
            <div class="text-sm text-gray-600">Top Predicted</div>
        </div>
        
        <div class="stat-card rounded-2xl p-6 text-center">
            <div class="text-3xl mb-2">📈</div>
            <div class="text-2xl font-bold text-purple-800" x-text="avgPrediction"></div>
            <div class="text-sm text-gray-600">Avg Prediction</div>
        </div>
        
        <div class="stat-card rounded-2xl p-6 text-center">
            <div class="text-3xl mb-2">🎯</div>
            <div class="text-lg font-bold text-green-600">AI Ready</div>
            <div class="text-sm text-gray-600">Model Status</div>
        </div>
    </div>

    <!-- Controls -->
    <div class="mx-6 mb-8 glass-card rounded-2xl p-6 fade-in">
        <div class="flex flex-wrap items-center gap-4 mb-6">
            <button @click="refreshData()" :disabled="loading" 
                    class="btn-primary px-8 py-3 rounded-full font-semibold flex items-center gap-3">
                <i class="fas fa-sync-alt" :class="{'loading-spinner': loading}"></i>
                <span x-text="loading ? 'Refreshing...' : 'Refresh Data'"></span>
            </button>
            
            <div class="relative flex-1 max-w-md">
                <i class="fas fa-search absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400"></i>
                <input x-model="searchQuery" @input="filterPlayers()" 
                       placeholder="Search players or teams..." 
                       class="search-input w-full pl-12 pr-4 py-3 rounded-full focus:outline-none">
            </div>
            
            <div class="text-sm text-gray-600" x-text="statusText"></div>
        </div>
        
        <!-- Position Filters -->
        <div class="flex gap-3 flex-wrap">
            <template x-for="pos in positions">
                <button @click="selectedPosition = pos; filterPlayers()" 
                        :class="selectedPosition === pos ? 'active' : ''"
                        class="filter-tab px-6 py-2 rounded-full font-medium text-sm"
                        x-text="pos === 'ALL' ? 'All Players' : pos">
                </button>
            </template>
        </div>
    </div>

    <!-- Main Content -->
    <div class="mx-6 mb-8 grid grid-cols-1 xl:grid-cols-3 gap-8">
        <!-- Predictions Chart -->
        <div class="xl:col-span-2 glass-card rounded-2xl p-8 fade-in">
            <div class="flex justify-between items-center mb-6">
                <h2 class="text-2xl font-bold text-gray-800">
                    <i class="fas fa-chart-line mr-3 text-purple-600"></i>
                    Top Predictions
                </h2>
                <div class="flex gap-2">
                    <button @click="chartView = 'points'; renderChart()" 
                            :class="chartView === 'points' ? 'bg-purple-600 text-white' : 'bg-gray-100'"
                            class="px-4 py-2 rounded-lg text-sm font-medium">Points</button>
                    <button @click="chartView = 'value'; renderChart()" 
                            :class="chartView === 'value' ? 'bg-purple-600 text-white' : 'bg-gray-100'"
                            class="px-4 py-2 rounded-lg text-sm font-medium">Value</button>
                </div>
            </div>
            <div id="predictionsChart" class="h-96"></div>
        </div>
        
        <!-- Quick Analysis -->
        <div class="glass-card rounded-2xl p-8 fade-in">
            <h2 class="text-2xl font-bold text-gray-800 mb-6">
                <i class="fas fa-brain mr-3 text-green-600"></i>
                AI Insights
            </h2>
            
            <div class="space-y-4">
                <div class="bg-green-50 border-l-4 border-green-400 p-4 rounded">
                    <h4 class="font-semibold text-green-800">Best Value Pick</h4>
                    <p class="text-green-700" x-text="bestValue"></p>
                </div>
                
                <div class="bg-blue-50 border-l-4 border-blue-400 p-4 rounded">
                    <h4 class="font-semibold text-blue-800">Top Differential</h4>
                    <p class="text-blue-700" x-text="topDifferential"></p>
                </div>
                
                <div class="bg-purple-50 border-l-4 border-purple-400 p-4 rounded">
                    <h4 class="font-semibold text-purple-800">Captain Pick</h4>
                    <p class="text-purple-700" x-text="captainPick"></p>
                </div>
            </div>
        </div>
    </div>

    <!-- Optimal Team Selection -->
    <div class="mx-6 mb-8 glass-card rounded-2xl p-8 fade-in" x-data="{ showTeam: false, optimalTeam: null, loading: false }">
        <div class="flex justify-between items-center mb-6">
            <h2 class="text-2xl font-bold text-gray-800">
                <i class="fas fa-users mr-3 text-blue-600"></i>
                Optimal Team Selection
            </h2>
            <button @click="loadOptimalTeam()" :disabled="loading"
                    class="btn-primary px-6 py-3 rounded-full font-semibold flex items-center gap-2">
                <i class="fas fa-magic" :class="{'loading-spinner': loading}"></i>
                <span x-text="loading ? 'Building Team...' : 'Generate Team'"></span>
            </button>
        </div>
        
        <div x-show="optimalTeam" x-transition class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <!-- Team Overview -->
            <div class="bg-gradient-to-br from-blue-50 to-purple-50 rounded-xl p-6">
                <h3 class="text-lg font-bold text-gray-800 mb-4">Team Overview</h3>
                <div class="space-y-3">
                    <div class="flex justify-between">
                        <span class="text-gray-600">Formation:</span>
                        <span class="font-bold" x-text="optimalTeam?.formation"></span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-600">Total Cost:</span>
                        <span class="font-bold text-green-600">£<span x-text="optimalTeam?.total_cost"></span>m</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-600">Budget Left:</span>
                        <span class="font-bold text-blue-600">£<span x-text="optimalTeam?.budget_remaining"></span>m</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-600">Predicted Points:</span>
                        <span class="font-bold text-purple-600" x-text="optimalTeam?.predicted_points"></span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-600">Chip Rec:</span>
                        <span class="font-bold" 
                              :class="optimalTeam?.chip_recommendation ? 'text-green-600' : 'text-gray-400'"
                              x-text="optimalTeam?.chip_recommendation || 'Save for Later'"></span>
                    </div>
                </div>
            </div>
            
            <!-- Starting XI -->
            <div class="bg-gradient-to-br from-green-50 to-blue-50 rounded-xl p-6">
                <h3 class="text-lg font-bold text-gray-800 mb-4">Starting XI</h3>
                <div class="space-y-2 max-h-80 overflow-y-auto">
                    <template x-for="player in (optimalTeam?.starters || [])">
                        <div class="flex items-center justify-between p-2 bg-white rounded-lg shadow-sm">
                            <div class="flex items-center space-x-3">
                                <span class="px-2 py-1 rounded text-xs font-bold text-white"
                                      :class="{
                                        'position-gkp': player.position === 'GKP',
                                        'position-def': player.position === 'DEF',
                                        'position-mid': player.position === 'MID',
                                        'position-fwd': player.position === 'FWD'
                                      }" x-text="player.position">
                                </span>
                                <div>
                                    <div class="font-semibold text-sm" x-text="player.name"></div>
                                    <div class="text-xs text-gray-500">
                                        <span x-text="player.team_short"></span> • 
                                        <span x-text="player.predicted_points?.toFixed(1)"></span>pts
                                    </div>
                                </div>
                            </div>
                            <div class="text-right">
                                <div class="font-bold text-green-600">£<span x-text="player.price"></span>m</div>
                                <div x-show="player.player_id === optimalTeam?.captain_id" 
                                     class="text-xs bg-yellow-200 text-yellow-800 px-1 rounded">C</div>
                                <div x-show="player.player_id === optimalTeam?.vice_captain_id" 
                                     class="text-xs bg-gray-200 text-gray-800 px-1 rounded">VC</div>
                            </div>
                        </div>
                    </template>
                </div>
            </div>
            
            <!-- Bench -->
            <div class="bg-gradient-to-br from-purple-50 to-pink-50 rounded-xl p-6">
                <h3 class="text-lg font-bold text-gray-800 mb-4">Bench</h3>
                <div class="space-y-2">
                    <template x-for="player in (optimalTeam?.bench || [])">
                        <div class="flex items-center justify-between p-2 bg-white rounded-lg shadow-sm">
                            <div class="flex items-center space-x-3">
                                <span class="px-2 py-1 rounded text-xs font-bold text-white"
                                      :class="{
                                        'position-gkp': player.position === 'GKP',
                                        'position-def': player.position === 'DEF',
                                        'position-mid': player.position === 'MID',
                                        'position-fwd': player.position === 'FWD'
                                      }" x-text="player.position">
                                </span>
                                <div>
                                    <div class="font-semibold text-sm" x-text="player.name"></div>
                                    <div class="text-xs text-gray-500">
                                        <span x-text="player.team_short"></span> • 
                                        <span x-text="player.predicted_points?.toFixed(1)"></span>pts
                                    </div>
                                </div>
                            </div>
                            <div class="font-bold text-green-600">£<span x-text="player.price"></span>m</div>
                        </div>
                    </template>
                </div>
            </div>
        </div>
    </div>

    <!-- Gameweek Performance Tracking -->
    <div class="mx-6 mb-8 glass-card rounded-2xl p-8 fade-in" x-data="{ performanceData: null, selectedGW: 1, loading: false }">
        <div class="flex justify-between items-center mb-6">
            <h2 class="text-2xl font-bold text-gray-800">
                <i class="fas fa-chart-bar mr-3 text-green-600"></i>
                Gameweek Performance Tracking
            </h2>
            <div class="flex gap-3">
                <select x-model="selectedGW" class="px-4 py-2 border rounded-lg">
                    <template x-for="gw in Array.from({length: 38}, (_, i) => i + 1)">
                        <option :value="gw" x-text="'GW' + gw"></option>
                    </template>
                </select>
                <button @click="trackGameweek()" :disabled="loading"
                        class="btn-primary px-6 py-2 rounded-lg font-semibold flex items-center gap-2">
                    <i class="fas fa-download" :class="{'loading-spinner': loading}"></i>
                    <span x-text="loading ? 'Loading...' : 'Get Results'"></span>
                </button>
            </div>
        </div>
        
        <div x-show="performanceData" x-transition class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <!-- Performance Summary -->
            <div class="bg-gradient-to-br from-purple-50 to-blue-50 rounded-xl p-6">
                <h3 class="text-lg font-bold text-gray-800 mb-4">Performance Summary</h3>
                <div class="space-y-3">
                    <div class="flex justify-between">
                        <span class="text-gray-600">Gameweek:</span>
                        <span class="font-bold" x-text="performanceData?.gameweek"></span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-600">Players Tracked:</span>
                        <span class="font-bold" x-text="performanceData?.total_players"></span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-600">Prediction Error (MAE):</span>
                        <span class="font-bold text-red-600" x-text="performanceData?.mae ? performanceData.mae.toFixed(2) + ' points' : 'N/A'"></span>
                    </div>
                </div>
            </div>
            
            <!-- Top Performers Comparison -->
            <div class="bg-gradient-to-br from-green-50 to-yellow-50 rounded-xl p-6">
                <h3 class="text-lg font-bold text-gray-800 mb-4">Top Performers vs Predictions</h3>
                <div class="text-sm">
                    <div class="mb-2 font-semibold text-green-700">Top Actual Performers:</div>
                    <template x-for="(player, index) in (performanceData?.top_actual_performers || []).slice(0, 3)">
                        <div class="flex justify-between text-xs mb-1">
                            <span x-text="player.name"></span>
                            <span>
                                <span class="text-green-600" x-text="player.actual_points + 'pts'"></span>
                                (pred: <span x-text="player.predicted_points?.toFixed(1)"></span>)
                            </span>
                        </div>
                    </template>
                </div>
            </div>
        </div>
        
        <div x-show="!performanceData && !loading" class="text-center py-8 text-gray-500">
            <i class="fas fa-calendar-alt text-4xl mb-4"></i>
            <p>Select a gameweek and click "Get Results" to track actual vs predicted performance</p>
            <p class="text-sm mt-2">Results will be available after gameweeks are completed (starting August 15th)</p>
        </div>
    </div>

    <!-- Players Table -->
    <div class="mx-6 mb-8 glass-card rounded-2xl p-8 fade-in">
        <h2 class="text-2xl font-bold text-gray-800 mb-6">
            <i class="fas fa-users mr-3 text-indigo-600"></i>
            Player Analysis
            <span class="text-base font-normal text-gray-500" x-text="'(' + filteredPlayers.length + ' players)'"></span>
        </h2>
        
        <div class="overflow-x-auto">
            <table class="w-full">
                <thead>
                    <tr class="border-b-2 border-gray-200">
                        <th class="text-left py-4 px-2 font-semibold text-gray-700">Player</th>
                        <th class="text-left py-4 px-2 font-semibold text-gray-700">Position</th>
                        <th class="text-left py-4 px-2 font-semibold text-gray-700">Team</th>
                        <th class="text-center py-4 px-2 font-semibold text-gray-700">Price</th>
                        <th class="text-center py-4 px-2 font-semibold text-gray-700">Prediction</th>
                        <th class="text-center py-4 px-2 font-semibold text-gray-700">Value</th>
                        <th class="text-center py-4 px-2 font-semibold text-gray-700">Status</th>
                    </tr>
                </thead>
                <tbody>
                    <template x-for="(player, index) in filteredPlayers">
                        <tr class="player-card border-b border-gray-100 hover:bg-gray-50">
                            <td class="py-4 px-2">
                                <div class="flex items-center">
                                    <span class="text-xs text-gray-500 mr-3 font-mono" x-text="index + 1"></span>
                                    <span class="font-semibold" x-text="player.name || 'Unknown'"></span>
                                </div>
                            </td>
                            <td class="py-4 px-2">
                                <span class="px-3 py-1 rounded-full text-xs font-bold text-white"
                                      :class="{
                                        'position-gkp': player.position === 'GKP',
                                        'position-def': player.position === 'DEF', 
                                        'position-mid': player.position === 'MID',
                                        'position-fwd': player.position === 'FWD'
                                      }"
                                      x-text="player.position || 'N/A'">
                                </span>
                            </td>
                            <td class="py-4 px-2">
                                <span class="font-medium" x-text="player.team_short || 'N/A'"></span>
                            </td>
                            <td class="py-4 px-2 text-center">
                                <span class="font-bold text-green-600" x-text="'£' + (player.price || 0) + 'm'"></span>
                            </td>
                            <td class="py-4 px-2 text-center">
                                <span class="text-xl font-bold text-purple-700" x-text="(player.predicted_points || 0).toFixed(1)"></span>
                            </td>
                            <td class="py-4 px-2 text-center">
                                <span class="text-sm font-semibold" x-text="((player.predicted_points || 0) / (player.price || 1)).toFixed(2)"></span>
                            </td>
                            <td class="py-4 px-2 text-center">
                                <span class="px-2 py-1 rounded-full text-xs" 
                                      :class="{
                                        'bg-green-100 text-green-800': player.fpl_status === 'a',
                                        'bg-red-100 text-red-800': ['i', 's', 'u'].includes(player.fpl_status),
                                        'bg-yellow-100 text-yellow-800': player.fpl_status === 'd',
                                        'bg-gray-100 text-gray-800': !player.fpl_status || player.fpl_status === 'n'
                                      }"
                                      x-text="getStatusText(player.fpl_status, player.chance_next)">
                                </span>
                            </td>
                        </tr>
                    </template>
                </tbody>
            </table>
        </div>
    </div>

    <script>
        function fplApp() {
            return {
                // Data
                players: [],
                filteredPlayers: [],
                
                // UI State
                loading: false,
                searchQuery: '',
                selectedPosition: 'ALL',
                positions: ['ALL', 'GKP', 'DEF', 'MID', 'FWD'],
                statusText: 'Ready',
                chartView: 'points',
                
                // Stats
                gameweek: 1,
                totalPlayers: 0,
                topPlayer: 'Loading...',
                avgPrediction: '0.0',
                lastUpdate: 'Now',
                bestValue: 'Analyzing...',
                topDifferential: 'Analyzing...',
                captainPick: 'Analyzing...',
                
                async init() {
                    await this.loadData();
                    await this.loadSummary();
                    this.filterPlayers();
                    this.renderChart();
                    this.updateInsights();
                },
                
                async loadData() {
                    try {
                        const response = await fetch('/predictions');
                        this.players = await response.json();
                        this.totalPlayers = this.players.length;
                        console.log(`Loaded ${this.players.length} players`);
                    } catch (error) {
                        console.error('Failed to load predictions:', error);
                        this.statusText = 'Error loading data';
                    }
                },
                
                async loadSummary() {
                    try {
                        const response = await fetch('/summary');
                        const data = await response.json();
                        this.gameweek = data.gameweek || 1;
                        this.avgPrediction = (data.avg_prediction || 0).toFixed(1);
                    } catch (error) {
                        console.error('Failed to load summary:', error);
                    }
                },
                
                filterPlayers() {
                    let filtered = this.players;
                    
                    // Position filter
                    if (this.selectedPosition !== 'ALL') {
                        filtered = filtered.filter(p => p.position === this.selectedPosition);
                    }
                    
                    // Search filter
                    if (this.searchQuery.trim()) {
                        const query = this.searchQuery.toLowerCase();
                        filtered = filtered.filter(p => 
                            (p.name || '').toLowerCase().includes(query) ||
                            (p.team_short || '').toLowerCase().includes(query)
                        );
                    }
                    
                    // Sort by predicted points
                    filtered.sort((a, b) => (b.predicted_points || 0) - (a.predicted_points || 0));
                    
                    this.filteredPlayers = filtered;
                },
                
                renderChart() {
                    const data = this.filteredPlayers.slice(0, 20);
                    if (!data.length) return;
                    
                    const trace = {
                        x: data.map(p => p.name || 'Unknown'),
                        y: data.map(p => this.chartView === 'value' ? 
                            (p.predicted_points || 0) / (p.price || 1) : 
                            (p.predicted_points || 0)
                        ),
                        type: 'bar',
                        marker: {
                            color: data.map(p => {
                                const colors = {
                                    'GKP': '#10b981', 'DEF': '#3b82f6', 
                                    'MID': '#f59e0b', 'FWD': '#ef4444'
                                };
                                return colors[p.position] || '#6b7280';
                            })
                        }
                    };
                    
                    const layout = {
                        margin: { t: 20, b: 100, l: 50, r: 20 },
                        plot_bgcolor: 'rgba(255,255,255,0.8)',
                        paper_bgcolor: 'transparent',
                        font: { family: 'Inter, sans-serif', color: '#374151' },
                        xaxis: { tickangle: -45, tickfont: { size: 10 } },
                        yaxis: { 
                            title: this.chartView === 'value' ? 'Value (Points/£)' : 'Predicted Points',
                            titlefont: { color: '#6b21a8' }
                        }
                    };
                    
                    Plotly.newPlot('predictionsChart', [trace], layout, {
                        displayModeBar: false,
                        responsive: true
                    });
                },
                
                updateInsights() {
                    if (!this.players.length) return;
                    
                    // Best value
                    const bestVal = this.players
                        .filter(p => (p.predicted_points || 0) > 0)
                        .sort((a, b) => ((b.predicted_points || 0) / (b.price || 1)) - ((a.predicted_points || 0) / (a.price || 1)))[0];
                    this.bestValue = bestVal ? `${bestVal.name} (${(bestVal.predicted_points / bestVal.price).toFixed(2)} pts/£)` : 'No data available';
                    
                    // Top player overall
                    const topPl = this.players.sort((a, b) => (b.predicted_points || 0) - (a.predicted_points || 0))[0];
                    this.topPlayer = topPl ? `${topPl.name} (${(topPl.predicted_points || 0).toFixed(1)} pts)` : 'No data available';
                    
                    // Captain pick (top predicted)
                    this.captainPick = topPl ? `${topPl.name} - ${(topPl.predicted_points || 0).toFixed(1)} pts` : 'No data available';
                    
                    // Top differential (low ownership, high predicted)
                    const differential = this.players
                        .filter(p => (p.ownership || 100) < 10 && (p.predicted_points || 0) > 0)
                        .sort((a, b) => (b.predicted_points || 0) - (a.predicted_points || 0))[0];
                    this.topDifferential = differential ? `${differential.name} (${(differential.ownership || 0)}% owned)` : 'Analyzing ownership data...';
                },
                
                async refreshData() {
                    this.loading = true;
                    this.statusText = 'Refreshing data...';
                    
                    try {
                        const response = await fetch('/refresh', { method: 'POST' });
                        const result = await response.json();
                        
                        if (response.ok) {
                            this.statusText = 'Data refreshed successfully!';
                            await this.loadData();
                            await this.loadSummary();
                            this.filterPlayers();
                            this.renderChart();
                            this.updateInsights();
                            this.lastUpdate = new Date().toLocaleTimeString();
                            
                            setTimeout(() => {
                                this.statusText = 'Ready';
                            }, 3000);
                        } else {
                            throw new Error(result.detail || 'Refresh failed');
                        }
                    } catch (error) {
                        this.statusText = 'Error refreshing data';
                        console.error('Refresh failed:', error);
                    } finally {
                        this.loading = false;
                    }
                },
                
                getStatusText(status, chance) {
                    const statusMap = {
                        'a': 'Available',
                        'i': 'Injured',
                        's': 'Suspended', 
                        'd': chance ? `Doubtful (${chance}%)` : 'Doubtful',
                        'u': 'Unavailable',
                        'n': 'Not in Squad'
                    };
                    return statusMap[status] || 'Unknown';
                }
            }
        }
        
        // Gameweek tracking function
        async function trackGameweek() {
            const trackingComponent = Alpine.$data(document.querySelector('[x-data*="performanceData"]'));
            const gw = trackingComponent.selectedGW;
            
            trackingComponent.loading = true;
            try {
                const response = await fetch(`/track-gameweek/${gw}`, { method: 'POST' });
                const data = await response.json();
                
                if (response.ok && data.status === 'success') {
                    // Get detailed results
                    const resultsResponse = await fetch(`/gameweek-results/${gw}`);
                    const resultsData = await resultsResponse.json();
                    
                    if (resultsResponse.ok && !resultsData.error) {
                        trackingComponent.performanceData = resultsData;
                    } else {
                        trackingComponent.performanceData = {
                            gameweek: gw,
                            error: resultsData.error || 'No results available yet',
                            total_players: 0,
                            mae: null
                        };
                    }
                } else {
                    trackingComponent.performanceData = {
                        gameweek: gw,
                        error: data.error || 'Failed to collect results',
                        total_players: 0,
                        mae: null
                    };
                }
            } catch (error) {
                console.error('Failed to track gameweek:', error);
                trackingComponent.performanceData = {
                    gameweek: gw,
                    error: 'Network error',
                    total_players: 0,
                    mae: null
                };
            } finally {
                trackingComponent.loading = false;
            }
        }
        
        // Team selection functions
        async function loadOptimalTeam() {
            const app = Alpine.$data(document.querySelector('[x-data]'));
            const teamComponent = Alpine.$data(document.querySelector('[x-data*="optimalTeam"]'));
            
            teamComponent.loading = true;
            try {
                const response = await fetch('/team?gameweek=' + app.gameweek);
                const data = await response.json();
                
                if (response.ok) {
                    teamComponent.optimalTeam = data;
                    teamComponent.showTeam = true;
                    console.log('Optimal team loaded:', data);
                } else {
                    throw new Error(data.detail || 'Failed to load team');
                }
            } catch (error) {
                console.error('Failed to load optimal team:', error);
                alert('Failed to load optimal team: ' + error.message);
            } finally {
                teamComponent.loading = false;
            }
        }
    </script>
</body>
</html>
    """