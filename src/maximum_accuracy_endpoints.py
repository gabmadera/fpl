"""
Maximum Accuracy System API Endpoints
Comprehensive FastAPI endpoints for the maximum accuracy FPL system
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import json
import asyncio
import logging

from .maximum_accuracy_system import MaximumAccuracySystem, AccuracyTarget, ModelConfig
from .automated_retraining_scheduler_v2 import AutomatedRetrainingScheduler, get_retraining_scheduler
from .aggressive_accuracy_optimizer import AggressiveAccuracyOptimizer, OptimizationStrategy
from .accuracy_monitoring_dashboard import AccuracyMonitoringDashboard
from .fpl_client import FPLClient
from .cache_manager import CacheManager

# Global instances
max_accuracy_system = MaximumAccuracySystem()
accuracy_dashboard = AccuracyMonitoringDashboard()
accuracy_optimizer = AggressiveAccuracyOptimizer()

app = FastAPI(title="Maximum Accuracy FPL System", version="1.0.0")


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """System health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system": "Maximum Accuracy FPL System",
        "version": "1.0.0"
    }


@app.get("/system/status")
def get_system_status() -> Dict[str, Any]:
    """Get comprehensive system status"""
    try:
        # Get status from all components
        max_accuracy_status = max_accuracy_system.get_system_status()
        scheduler_status = get_retraining_scheduler().get_status()
        dashboard_status = accuracy_dashboard.get_real_time_status()
        optimizer_status = accuracy_optimizer.get_optimization_summary()

        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "maximum_accuracy_system": max_accuracy_status,
                "automated_scheduler": scheduler_status,
                "accuracy_dashboard": dashboard_status,
                "accuracy_optimizer": optimizer_status
            },
            "overall_health": "operational"
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@app.get("/predictions/maximum-accuracy")
def get_maximum_accuracy_predictions(
    force_refresh: bool = Query(False, description="Force refresh predictions"),
    optimization_strategy: str = Query("maximum", description="Optimization strategy")
) -> Dict[str, Any]:
    """Get maximum accuracy predictions with aggressive optimization"""
    try:
        cache_manager = CacheManager()
        cache_key = f'max_accuracy_predictions_{optimization_strategy}'

        # Check cache if not forcing refresh
        if not force_refresh:
            cached_predictions = cache_manager.get_cached_data(cache_key)
            if cached_predictions and cache_manager.is_cache_valid(cache_key, hours=1):
                return {
                    "status": "success",
                    "predictions": cached_predictions,
                    "from_cache": True,
                    "timestamp": datetime.now().isoformat()
                }

        # Set optimization strategy
        strategy_map = {
            "conservative": OptimizationStrategy.CONSERVATIVE,
            "balanced": OptimizationStrategy.BALANCED,
            "aggressive": OptimizationStrategy.AGGRESSIVE,
            "maximum": OptimizationStrategy.MAXIMUM
        }

        if optimization_strategy in strategy_map:
            accuracy_optimizer.adjust_strategy(strategy_map[optimization_strategy])

        # Generate maximum accuracy predictions
        predictions_df = max_accuracy_system.generate_maximum_accuracy_predictions()

        # Apply aggressive optimization
        optimized_predictions = accuracy_optimizer.optimize_predictions(predictions_df)

        # Convert to JSON-serializable format
        predictions_data = optimized_predictions.fillna(0).to_dict(orient='records')

        # Cache results
        cache_manager.cache_data(cache_key, predictions_data)

        # Capture accuracy snapshot
        snapshot = accuracy_dashboard.capture_accuracy_snapshot(optimized_predictions)

        return {
            "status": "success",
            "predictions": predictions_data,
            "prediction_count": len(predictions_data),
            "optimization_strategy": optimization_strategy,
            "system_confidence": snapshot.model_confidence,
            "target_accuracy": snapshot.target_accuracy,
            "from_cache": False,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logging.error(f"Error generating maximum accuracy predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions/captain-optimization")
def get_captain_optimization() -> Dict[str, Any]:
    """Get optimized captain recommendations"""
    try:
        # Get latest predictions
        predictions_df = max_accuracy_system.generate_maximum_accuracy_predictions()

        # Apply captain optimization
        captain_optimized = accuracy_optimizer.optimize_captaincy_selection(predictions_df)

        # Get top captain candidates
        captain_candidates = captain_optimized[
            captain_optimized['captain_candidate'] == True
        ].nlargest(10, 'captaincy_score')

        captain_data = captain_candidates[[
            'player_id', 'name', 'position', 'team_name', 'price',
            'predicted_points', 'captaincy_score', 'captaincy_risk',
            'differential_captain', 'selected_by_percent'
        ]].fillna(0).to_dict(orient='records')

        return {
            "status": "success",
            "captain_candidates": captain_data,
            "total_candidates": len(captain_data),
            "optimization_strategy": accuracy_optimizer.strategy.value,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logging.error(f"Error in captain optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions/differentials")
def get_differential_picks() -> Dict[str, Any]:
    """Get optimized differential pick recommendations"""
    try:
        # Get latest predictions
        predictions_df = max_accuracy_system.generate_maximum_accuracy_predictions()

        # Apply differential optimization
        differential_optimized = accuracy_optimizer.optimize_differential_picks(predictions_df)

        # Get top differential candidates
        differentials = differential_optimized[
            differential_optimized['differential_candidate'] == True
        ].nlargest(20, 'differential_score')

        # Get ultra differentials if using maximum strategy
        ultra_differentials = []
        if 'ultra_differential' in differential_optimized.columns:
            ultra_differentials = differential_optimized[
                differential_optimized['ultra_differential'] == True
            ].nlargest(10, 'predicted_points').fillna(0).to_dict(orient='records')

        differential_data = differentials[[
            'player_id', 'name', 'position', 'team_name', 'price',
            'predicted_points', 'differential_score', 'selected_by_percent',
            'value_efficiency', 'is_differential'
        ]].fillna(0).to_dict(orient='records')

        return {
            "status": "success",
            "differential_picks": differential_data,
            "ultra_differentials": ultra_differentials,
            "total_differentials": len(differential_data),
            "ownership_threshold": accuracy_optimizer.strategy_configs[accuracy_optimizer.strategy]['differential_threshold'] * 100,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logging.error(f"Error in differential optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/monitoring/accuracy-dashboard")
def get_accuracy_dashboard() -> Dict[str, Any]:
    """Get comprehensive accuracy monitoring dashboard data"""
    try:
        dashboard_data = accuracy_dashboard.generate_dashboard_data()

        return {
            "status": "success",
            "dashboard": dashboard_data,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logging.error(f"Error generating accuracy dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/monitoring/accuracy-visualizations")
def get_accuracy_visualizations() -> Dict[str, Any]:
    """Get accuracy monitoring visualizations"""
    try:
        visualizations = accuracy_dashboard.create_accuracy_visualizations()

        return {
            "status": "success",
            "visualizations": visualizations,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logging.error(f"Error creating visualizations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retraining/trigger")
def trigger_manual_retraining(
    gameweek: Optional[int] = Query(None, description="Gameweek to retrain for"),
    background_tasks: BackgroundTasks = None
) -> Dict[str, Any]:
    """Manually trigger model retraining"""
    try:
        scheduler = get_retraining_scheduler()

        if gameweek is None:
            # Get current gameweek
            fpl = FPLClient()
            bs = fpl.bootstrap_static()
            events = bs.get('events', [])
            gameweek = next((e['id'] for e in events if e.get('is_current')), 1)

        # Trigger retraining in background
        if background_tasks:
            background_tasks.add_task(scheduler.trigger_retraining, gameweek, "Manual trigger")

            return {
                "status": "triggered",
                "message": f"Retraining triggered for gameweek {gameweek}",
                "gameweek": gameweek,
                "background": True,
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Synchronous retraining
            event = scheduler.trigger_retraining(gameweek, "Manual trigger")

            return {
                "status": "completed",
                "retraining_event": {
                    "gameweek": event.gameweek,
                    "status": event.status.value,
                    "trigger_reason": event.trigger_reason,
                    "completion_time": event.completion_time.isoformat() if event.completion_time else None,
                    "error_message": event.error_message
                },
                "timestamp": datetime.now().isoformat()
            }

    except Exception as e:
        logging.error(f"Error triggering retraining: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/retraining/status")
def get_retraining_status() -> Dict[str, Any]:
    """Get retraining scheduler status"""
    try:
        scheduler = get_retraining_scheduler()
        status = scheduler.get_status()

        return {
            "status": "success",
            "scheduler_status": status,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logging.error(f"Error getting retraining status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/retraining/history")
def get_retraining_history(limit: int = Query(20, description="Number of events to return")) -> Dict[str, Any]:
    """Get retraining history"""
    try:
        scheduler = get_retraining_scheduler()
        history = scheduler.get_retraining_history(limit)

        return {
            "status": "success",
            "retraining_history": history,
            "total_events": len(history),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logging.error(f"Error getting retraining history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/automation/start")
def start_automated_system() -> Dict[str, Any]:
    """Start the automated retraining system"""
    try:
        scheduler = get_retraining_scheduler()
        scheduler.start_monitoring()

        return {
            "status": "success",
            "message": "Automated retraining system started",
            "scheduler_status": scheduler.get_status(),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logging.error(f"Error starting automated system: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/automation/stop")
def stop_automated_system() -> Dict[str, Any]:
    """Stop the automated retraining system"""
    try:
        scheduler = get_retraining_scheduler()
        scheduler.stop_monitoring()

        return {
            "status": "success",
            "message": "Automated retraining system stopped",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logging.error(f"Error stopping automated system: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimization/strategy")
def set_optimization_strategy(strategy: str) -> Dict[str, Any]:
    """Set optimization strategy"""
    try:
        strategy_map = {
            "conservative": OptimizationStrategy.CONSERVATIVE,
            "balanced": OptimizationStrategy.BALANCED,
            "aggressive": OptimizationStrategy.AGGRESSIVE,
            "maximum": OptimizationStrategy.MAXIMUM
        }

        if strategy not in strategy_map:
            raise HTTPException(status_code=400, detail=f"Invalid strategy. Must be one of: {list(strategy_map.keys())}")

        accuracy_optimizer.adjust_strategy(strategy_map[strategy])

        return {
            "status": "success",
            "message": f"Optimization strategy set to {strategy}",
            "strategy": strategy,
            "config": accuracy_optimizer.strategy_configs[strategy_map[strategy]],
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logging.error(f"Error setting optimization strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/accuracy/targets")
def get_accuracy_targets() -> Dict[str, Any]:
    """Get accuracy targets and progress"""
    try:
        latest_snapshot = accuracy_dashboard.accuracy_snapshots[-1] if accuracy_dashboard.accuracy_snapshots else None

        target_timeline = max_accuracy_system.target_timeline

        progress_data = {
            "current_accuracy": latest_snapshot.overall_accuracy if latest_snapshot else 0.0,
            "target_accuracy": max_accuracy_system.config.target_accuracy.value,
            "gameweek": latest_snapshot.gameweek if latest_snapshot else 1,
            "progress_percentage": latest_snapshot.progress_percentage if latest_snapshot else 0.0,
            "weeks_behind_schedule": latest_snapshot.weeks_behind_schedule if latest_snapshot else 0,
            "target_timeline": target_timeline
        }

        return {
            "status": "success",
            "accuracy_targets": progress_data,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logging.error(f"Error getting accuracy targets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions/team-optimization")
def get_team_optimization(
    formation: str = Query("3-5-2", description="Team formation"),
    budget: float = Query(100.0, description="Team budget in millions"),
    aggressive_picks: bool = Query(True, description="Include aggressive differential picks")
) -> Dict[str, Any]:
    """Get optimized team selection"""
    try:
        # Get optimized predictions
        predictions_df = max_accuracy_system.generate_maximum_accuracy_predictions()
        optimized_predictions = accuracy_optimizer.optimize_predictions(predictions_df)

        # Simple team selection based on value and predictions
        team_selection = _select_optimal_team(
            optimized_predictions, formation, budget, aggressive_picks
        )

        return {
            "status": "success",
            "team_selection": team_selection,
            "formation": formation,
            "total_cost": sum(p['price'] for p in team_selection['players']),
            "expected_points": sum(p['predicted_points'] for p in team_selection['players']),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logging.error(f"Error in team optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _select_optimal_team(predictions_df: pd.DataFrame, formation: str, budget: float, aggressive_picks: bool) -> Dict[str, Any]:
    """Simple team selection algorithm"""
    try:
        # Parse formation
        formation_parts = formation.split('-')
        if len(formation_parts) != 3:
            raise ValueError("Invalid formation format")

        def_count = int(formation_parts[0])
        mid_count = int(formation_parts[1])
        fwd_count = int(formation_parts[2])

        # Position requirements
        position_requirements = {
            'GKP': 1,
            'DEF': def_count,
            'MID': mid_count,
            'FWD': fwd_count
        }

        selected_players = []
        remaining_budget = budget

        # Select players by position
        for position, count in position_requirements.items():
            position_players = predictions_df[predictions_df['position'] == position].copy()

            if aggressive_picks:
                # Boost differential picks
                differential_mask = position_players.get('differential_candidate', False)
                position_players.loc[differential_mask, 'adjusted_value'] = (
                    position_players.loc[differential_mask, 'predicted_points'] /
                    position_players.loc[differential_mask, 'price'] * 1.2
                )
            else:
                position_players['adjusted_value'] = (
                    position_players['predicted_points'] / position_players['price']
                )

            # Sort by adjusted value and select top players within budget
            position_players = position_players.sort_values('adjusted_value', ascending=False)

            for _, player in position_players.iterrows():
                if len([p for p in selected_players if p['position'] == position]) < count:
                    if player['price'] <= remaining_budget:
                        selected_players.append({
                            'player_id': player['player_id'],
                            'name': player['name'],
                            'position': player['position'],
                            'team': player.get('team_name', 'Unknown'),
                            'price': player['price'],
                            'predicted_points': player['predicted_points'],
                            'is_differential': player.get('differential_candidate', False),
                            'captain_candidate': player.get('captain_candidate', False)
                        })
                        remaining_budget -= player['price']

        # Find captain
        captain = max(selected_players, key=lambda x: x['predicted_points'])
        captain['is_captain'] = True

        return {
            'players': selected_players,
            'captain': captain,
            'remaining_budget': remaining_budget,
            'formation': formation
        }

    except Exception as e:
        logging.error(f"Error in team selection: {e}")
        return {'players': [], 'captain': None, 'remaining_budget': budget, 'formation': formation}


@app.get("/", response_class=HTMLResponse)
def get_dashboard_ui() -> str:
    """Serve the maximum accuracy dashboard UI"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Maximum Accuracy FPL System</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
            .card { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .metric { display: inline-block; margin-right: 30px; }
            .metric-label { font-size: 14px; color: #666; }
            .metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
            .success { color: #27ae60; }
            .warning { color: #f39c12; }
            .critical { color: #e74c3c; }
            .btn { padding: 10px 20px; background: #3498db; color: white; border: none; border-radius: 4px; cursor: pointer; margin-right: 10px; }
            .btn:hover { background: #2980b9; }
            .btn-success { background: #27ae60; }
            .btn-warning { background: #f39c12; }
            .btn-danger { background: #e74c3c; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎯 Maximum Accuracy FPL System</h1>
                <p>Advanced ML-powered FPL predictions targeting 70-75%+ accuracy</p>
            </div>

            <div class="card">
                <h2>System Status</h2>
                <div id="system-status">Loading...</div>
            </div>

            <div class="card">
                <h2>Quick Actions</h2>
                <button class="btn" onclick="getPredictions()">Get Latest Predictions</button>
                <button class="btn btn-warning" onclick="triggerRetraining()">Trigger Retraining</button>
                <button class="btn btn-success" onclick="startAutomation()">Start Automation</button>
                <button class="btn btn-danger" onclick="stopAutomation()">Stop Automation</button>
            </div>

            <div class="card">
                <h2>Maximum Accuracy Predictions</h2>
                <div id="predictions">Click "Get Latest Predictions" to load predictions</div>
            </div>

            <div class="card">
                <h2>Accuracy Monitoring</h2>
                <div id="accuracy-dashboard">Loading dashboard...</div>
            </div>
        </div>

        <script>
            async function loadSystemStatus() {
                try {
                    const response = await fetch('/system/status');
                    const data = await response.json();
                    document.getElementById('system-status').innerHTML = `
                        <div class="metric">
                            <div class="metric-label">Overall Health</div>
                            <div class="metric-value success">${data.overall_health}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Last Update</div>
                            <div class="metric-value">${new Date(data.timestamp).toLocaleString()}</div>
                        </div>
                    `;
                } catch (error) {
                    document.getElementById('system-status').innerHTML = `<span class="critical">Error loading status: ${error}</span>`;
                }
            }

            async function getPredictions() {
                try {
                    document.getElementById('predictions').innerHTML = 'Loading predictions...';
                    const response = await fetch('/predictions/maximum-accuracy?optimization_strategy=maximum');
                    const data = await response.json();

                    const topPredictions = data.predictions.slice(0, 10);
                    const html = `
                        <p><strong>Strategy:</strong> ${data.optimization_strategy} | <strong>Count:</strong> ${data.prediction_count} | <strong>Confidence:</strong> ${(data.system_confidence * 100).toFixed(1)}%</p>
                        <table style="width: 100%; border-collapse: collapse;">
                            <thead>
                                <tr style="background: #f8f9fa;">
                                    <th style="padding: 8px; text-align: left; border: 1px solid #ddd;">Player</th>
                                    <th style="padding: 8px; text-align: left; border: 1px solid #ddd;">Position</th>
                                    <th style="padding: 8px; text-align: left; border: 1px solid #ddd;">Team</th>
                                    <th style="padding: 8px; text-align: right; border: 1px solid #ddd;">Price</th>
                                    <th style="padding: 8px; text-align: right; border: 1px solid #ddd;">Predicted Points</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${topPredictions.map(p => `
                                    <tr>
                                        <td style="padding: 8px; border: 1px solid #ddd;">${p.name}</td>
                                        <td style="padding: 8px; border: 1px solid #ddd;">${p.position}</td>
                                        <td style="padding: 8px; border: 1px solid #ddd;">${p.team_name || 'Unknown'}</td>
                                        <td style="padding: 8px; text-align: right; border: 1px solid #ddd;">£${p.price}m</td>
                                        <td style="padding: 8px; text-align: right; border: 1px solid #ddd;">${p.predicted_points.toFixed(1)}</td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    `;
                    document.getElementById('predictions').innerHTML = html;
                } catch (error) {
                    document.getElementById('predictions').innerHTML = `<span class="critical">Error loading predictions: ${error}</span>`;
                }
            }

            async function triggerRetraining() {
                try {
                    const response = await fetch('/retraining/trigger', { method: 'POST' });
                    const data = await response.json();
                    alert(`Retraining ${data.status}: ${data.message}`);
                } catch (error) {
                    alert(`Error: ${error}`);
                }
            }

            async function startAutomation() {
                try {
                    const response = await fetch('/automation/start', { method: 'POST' });
                    const data = await response.json();
                    alert(data.message);
                } catch (error) {
                    alert(`Error: ${error}`);
                }
            }

            async function stopAutomation() {
                try {
                    const response = await fetch('/automation/stop', { method: 'POST' });
                    const data = await response.json();
                    alert(data.message);
                } catch (error) {
                    alert(`Error: ${error}`);
                }
            }

            async function loadAccuracyDashboard() {
                try {
                    const response = await fetch('/monitoring/accuracy-dashboard');
                    const data = await response.json();
                    const dashboard = data.dashboard;

                    document.getElementById('accuracy-dashboard').innerHTML = `
                        <div class="metric">
                            <div class="metric-label">Current Accuracy</div>
                            <div class="metric-value ${dashboard.current_metrics.overall_accuracy >= 0.7 ? 'success' : dashboard.current_metrics.overall_accuracy >= 0.6 ? 'warning' : 'critical'}">${(dashboard.current_metrics.overall_accuracy * 100).toFixed(1)}%</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Target Accuracy</div>
                            <div class="metric-value">${(dashboard.current_metrics.target_accuracy * 100).toFixed(1)}%</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Progress</div>
                            <div class="metric-value">${dashboard.progress_data.progress_percentage.toFixed(1)}%</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Active Alerts</div>
                            <div class="metric-value ${dashboard.active_alerts.length > 0 ? 'warning' : 'success'}">${dashboard.active_alerts.length}</div>
                        </div>
                    `;
                } catch (error) {
                    document.getElementById('accuracy-dashboard').innerHTML = `<span class="critical">Error loading dashboard: ${error}</span>`;
                }
            }

            // Load initial data
            loadSystemStatus();
            loadAccuracyDashboard();
        </script>
    </body>
    </html>
    """


# Add endpoints to existing web UI
def add_maximum_accuracy_endpoints(existing_app: FastAPI):
    """Add maximum accuracy endpoints to existing FastAPI app"""

    # Mount the maximum accuracy endpoints
    existing_app.mount("/max-accuracy", app)

    return existing_app