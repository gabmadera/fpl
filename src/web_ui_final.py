from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import pandas as pd
from pathlib import Path
from datetime import datetime

from .ml_pipeline import MLPipeline
from .fpl_client import FPLClient
from .comprehensive_accuracy_tracker import ComprehensiveAccuracyTracker
from .cache_manager import CacheManager

app = FastAPI(title="FPL AI Dashboard - Final Working Version", version="1.0")

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

@app.get("/predictions")
def predictions(force_refresh: bool = False) -> dict:
    cache_manager = CacheManager()
    
    try:
        # Use cache unless force refresh
        if not force_refresh:
            cached_predictions = cache_manager.get_cached_data('predictions')
            if cached_predictions is not None and not cached_predictions.empty:
                df = cached_predictions
            else:
                ml = MLPipeline()
                df = ml.predict_current()
                cache_manager.cache_data('predictions', df)
        else:
            # Force refresh - clear cache and generate new predictions
            cache_manager.clear_cache('predictions')
            ml = MLPipeline()
            df = ml.predict_current()
            cache_manager.cache_data('predictions', df, {'force_refresh': True})
        
        # Normalize/ensure position column for UI
        try:
            position_mapping = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
            if 'position' not in df.columns:
                if 'element_type' in df.columns:
                    df['position'] = df['element_type'].map(position_mapping)
                elif 'position_id' in df.columns:
                    df['position'] = df['position_id'].map(position_mapping)
        except Exception:
            pass

        # Convert to records for JSON
        players = df.head(50).to_dict('records') if not df.empty else []
        
        return {
            "status": "success",
            "players": players,
            "total_count": len(df),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/suggestions/latest")
def latest_suggestion() -> dict:
    try:
        tracker = ComprehensiveAccuracyTracker()
        if not tracker.suggestions_history:
            return {"status": "empty"}
        s = tracker.suggestions_history[-1]
        captain_name = next((p.get('name') for p in s.starters if p.get('player_id') == s.captain_id), None)
        vice_name = next((p.get('name') for p in s.starters if p.get('player_id') == s.vice_captain_id), None)
        return {
            "status": "success",
            "gameweek": s.gameweek,
            "timestamp": s.timestamp.isoformat(),
            "formation": s.formation,
            "predicted_points": s.predicted_points,
            "total_cost": s.total_cost,
            "captain": captain_name,
            "vice_captain": vice_name,
            "starters": s.starters[:11],
            "bench": s.bench[:4]
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/suggestions/history")
def suggestions_history(limit: int = 10) -> dict:
    try:
        tracker = ComprehensiveAccuracyTracker()
        hist = tracker.suggestions_history[-limit:]
        out = []
        for s in hist:
            out.append({
                "gameweek": s.gameweek,
                "timestamp": s.timestamp.isoformat(),
                "formation": s.formation,
                "predicted_points": s.predicted_points,
                "total_cost": s.total_cost
            })
        return {"status": "success", "suggestions": out}
    except Exception as e:
        return {"error": str(e)}

# Quick Actions
@app.post("/quick-actions/log-current-suggestion")
def log_current_suggestion() -> dict:
    try:
        tracker = ComprehensiveAccuracyTracker()
        fpl = FPLClient()
        
        # Get current gameweek
        bs = fpl.bootstrap_static()
        events = bs.get("events", [])
        current_gw = next((e["id"] for e in events if e.get("is_current", False)), 1)
        next_gw = next((e["id"] for e in events if e.get("is_next", False)), None) or (current_gw + 1)
        
        # Get fresh predictions (force refresh to get latest team names)
        ml = MLPipeline()
        predictions_df = ml.predict_current()
        
        if predictions_df.empty:
            return {"error": "No predictions available"}
        
        # Log suggestion
        suggestion = tracker.log_gameweek_suggestion(next_gw, predictions_df, [])
        
        # Get captain and vice-captain names
        captain_name = "Unknown"
        vice_name = "Unknown"
        
        for player in suggestion.starters:
            if player.get('player_id') == suggestion.captain_id:
                captain_name = player.get('name', 'Unknown')
            elif player.get('player_id') == suggestion.vice_captain_id:
                vice_name = player.get('name', 'Unknown')
        
        return {
            "status": "success",
            "gameweek": next_gw,
            "predicted_points": suggestion.predicted_points,
            "formation": suggestion.formation,
            "captain": captain_name,
            "vice_captain": vice_name,
            "total_cost": suggestion.total_cost,
            "starters": suggestion.starters[:11],  # Show first 11 starters
            "bench": suggestion.bench[:4],  # Show bench players
            "message": f"Team suggestion logged for GW{next_gw}"
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.post("/quick-actions/collect-last-results")
def collect_last_results() -> dict:
    try:
        tracker = ComprehensiveAccuracyTracker()
        fpl = FPLClient()
        
        # Get previous gameweek
        bs = fpl.bootstrap_static()
        events = bs.get("events", [])
        # Use last finished event if available
        prev_gw = max([e["id"] for e in events if e.get("finished", False)], default=1)
        
        # Special handling for GW1 - create a basic result since no suggestion was logged
        if prev_gw == 1:
            # For GW1, just return a simple message without trying to fetch results
            # to avoid rate limiting issues
            return {
                "status": "success",
                "gameweek": 1,
                "message": f"GW1 - No team suggestion was logged",
                "note": "You need to log a team suggestion before you can collect results. Start by logging a suggestion for the next gameweek!",
                "advice": "Use 'Log Current Suggestion' to track your team predictions."
            }
        
        # Normal flow for other gameweeks
        result = tracker.collect_gameweek_results(prev_gw)
        if result is None:
            return {"error": f"No team suggestion found for GW{prev_gw}. Log a suggestion first to track results."}
        
        return {
            "status": "success",
            "gameweek": prev_gw,
            "actual_points": result.actual_team_points,
            "predicted_points": result.team_selection.predicted_points,
            "captain_points": result.actual_captain_points,
            "points_vs_optimal": result.points_vs_optimal
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.post("/quick-actions/generate-quick-report")
def generate_quick_report() -> dict:
    try:
        tracker = ComprehensiveAccuracyTracker()
        report = tracker.get_comprehensive_report(5)
        
        return {
            "status": "success",
            "report": report
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.post("/quick-actions/auto-improve")
def auto_improve() -> dict:
    try:
        tracker = ComprehensiveAccuracyTracker()
        suggestions = tracker.get_improvement_suggestions()
        
        needs_retrain = any("retraining" in s.lower() or "accuracy declining" in s.lower() 
                          for s in suggestions)
        
        if needs_retrain:
            from .weekly_retrainer import WeeklyMLRetrainer
            retrainer = WeeklyMLRetrainer()
            
            fpl = FPLClient()
            bs = fpl.bootstrap_static()
            events = bs.get("events", [])
            current_gw = next((e["id"] for e in events if e.get("is_current", False)), 1)
            
            result = retrainer.adaptive_retrain(current_gw)
            
            return {
                "status": "success",
                "action": "model_retrained",
                "retrain_result": result,
                "suggestions": suggestions
            }
        else:
            return {
                "status": "success", 
                "action": "no_action_needed",
                "message": "Model performance is good",
                "suggestions": suggestions
            }
        
    except Exception as e:
        return {"error": str(e)}

@app.post("/force-retrain")
def force_retrain() -> dict:
    try:
        from .weekly_retrainer import WeeklyMLRetrainer
        retrainer = WeeklyMLRetrainer()
        
        fpl = FPLClient()
        bs = fpl.bootstrap_static()
        events = bs.get("events", [])
        current_gw = next((e["id"] for e in events if e.get("is_current", False)), 1)
        
        # Force full retraining
        result = retrainer._full_retrain(current_gw)
        
        # Clear all caches
        cache_manager = CacheManager()
        cache_manager.clear_cache()
        
        return {
            "status": "success",
            "retrain_result": result,
            "gameweek": current_gw,
            "message": f"Manual full retraining completed for GW{current_gw}. All caches cleared."
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/cache-status")
def cache_status() -> dict:
    try:
        cache_manager = CacheManager()
        status = cache_manager.get_all_cache_status()
        
        return {
            "status": "success",
            "cache_status": status,
            "system_time": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FPL AI Dashboard - Final Working Version</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .btn-primary { 
            @apply bg-gradient-to-r from-purple-600 to-blue-600 text-white px-6 py-3 rounded-lg font-semibold hover:from-purple-700 hover:to-blue-700 transition-all duration-300 cursor-pointer;
        }
        .glass-card {
            background: rgba(255, 255, 255, 0.25);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.18);
        }
        .fade-in {
            animation: fadeIn 0.5s ease-in;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .loading-spinner {
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        .hidden { display: none !important; }
    </style>
</head>
<body class="min-h-screen bg-gradient-to-br from-purple-100 via-blue-50 to-green-100">

    <!-- Header -->
    <header class="bg-white shadow-lg">
        <div class="mx-auto px-6 py-4">
            <div class="flex items-center justify-between">
                <div class="flex items-center space-x-4">
                    <div class="text-3xl">⚽</div>
                    <div>
                        <h1 class="text-3xl font-bold bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent">
                            FPL AI Dashboard
                        </h1>
                        <p class="text-gray-600">Smart Learning Mode - Final Working Version</p>
                    </div>
                </div>
            </div>
        </div>
    </header>

    <!-- Quick Actions Panel -->
    <div class="mx-6 mb-8 mt-8 bg-gradient-to-r from-green-500 to-blue-600 rounded-2xl p-6 text-white fade-in">
        <h2 class="text-2xl font-bold mb-4">🚀 Quick Actions</h2>
        <div class="grid grid-cols-2 lg:grid-cols-5 gap-4">
            <button onclick="quickLogSuggestion()" 
                    class="bg-white bg-opacity-20 hover:bg-opacity-30 rounded-lg p-4 transition-all cursor-pointer">
                <div class="text-2xl mb-2">📝</div>
                <div class="text-sm font-medium">Log Current Suggestion</div>
            </button>
            <button onclick="quickCollectResults()" 
                    class="bg-white bg-opacity-20 hover:bg-opacity-30 rounded-lg p-4 transition-all cursor-pointer">
                <div class="text-2xl mb-2">📊</div>
                <div class="text-sm font-medium">Collect Last Results</div>
            </button>
            <button onclick="quickGenerateReport()" 
                    class="bg-white bg-opacity-20 hover:bg-opacity-30 rounded-lg p-4 transition-all cursor-pointer">
                <div class="text-2xl mb-2">📈</div>
                <div class="text-sm font-medium">Generate Report</div>
            </button>
            <button onclick="quickAutoImprove()" 
                    class="bg-white bg-opacity-20 hover:bg-opacity-30 rounded-lg p-4 transition-all cursor-pointer">
                <div class="text-2xl mb-2">🔧</div>
                <div class="text-sm font-medium">Auto Improve</div>
            </button>
            <button onclick="forceRetrain()" 
                    class="bg-red-500 bg-opacity-30 hover:bg-opacity-50 rounded-lg p-4 transition-all cursor-pointer border-2 border-red-300">
                <div class="text-2xl mb-2">🔴</div>
                <div class="text-sm font-medium">FORCE RETRAIN</div>
                <div class="text-xs opacity-75">Emergency Use</div>
            </button>
        </div>
    </div>

    <!-- Test Panel -->
    <div class="mx-6 mb-8 glass-card rounded-2xl p-6 fade-in">
        <h2 class="text-xl font-bold text-gray-800 mb-4">🧪 Test Functions</h2>
        <div class="grid grid-cols-2 lg:grid-cols-5 gap-4">
            <button onclick="testBasicJS()" class="btn-primary">Test JavaScript</button>
            <button onclick="testAPI()" class="btn-primary">Test API Health</button>
            <button onclick="testPredictions()" class="btn-primary">Test Predictions</button>
            <button onclick="showCacheStatus()" class="btn-primary">Show Cache Status</button>
            <button onclick="forceRefreshPredictions()" class="btn-primary bg-orange-600 hover:bg-orange-700">Force Refresh</button>
        </div>
    </div>

    <!-- Predictions Panel -->
    <div class="mx-6 mb-8 glass-card rounded-2xl p-6 fade-in">
        <div class="flex justify-between items-center mb-4">
            <h2 class="text-xl font-bold text-gray-800">📊 Current Predictions</h2>
            <button onclick="loadPredictions()" class="btn-primary">Load Predictions</button>
        </div>
        <div id="predictions-content" class="text-gray-600">Click "Load Predictions" to see current player predictions...</div>
    </div>

    <!-- Suggested Team Panel -->
    <div class="mx-6 mb-8 glass-card rounded-2xl p-6 fade-in">
        <h2 class="text-xl font-bold text-gray-800 mb-4">🧩 Suggested Team</h2>
        <div id="suggested-team" class="text-gray-700">No suggestion loaded yet.</div>
    </div>

    <!-- Results Display -->
    <div class="mx-6 mb-8 glass-card rounded-2xl p-6 fade-in">
        <h2 class="text-xl font-bold text-gray-800 mb-4">📋 Results</h2>
        <div id="results-content" class="text-gray-600">Results will appear here...</div>
    </div>

    <script>
        let globalLoading = false;

        // Basic JavaScript Test
        function testBasicJS() {
            alert('✅ JavaScript is working perfectly!');
            updateResults('✅ JavaScript test passed');
        }

        // API Health Test
        async function testAPI() {
            try {
                updateResults('🧪 Testing API connection...');
                const response = await fetch('/health');
                const data = await response.json();
                
                if (response.ok) {
                    alert('✅ API is working! Status: ' + data.status);
                    updateResults('✅ API health check passed: ' + JSON.stringify(data));
                } else {
                    throw new Error('API returned error status');
                }
            } catch (error) {
                console.error('API test failed:', error);
                alert('❌ API test failed: ' + error.message);
                updateResults('❌ API test failed: ' + error.message);
            }
        }

        // Test Predictions Endpoint
        async function testPredictions() {
            try {
                updateResults('🔍 Testing predictions endpoint...');
                const response = await fetch('/predictions');
                const data = await response.json();
                
                console.log('Predictions response:', data);
                
                if (response.ok && data.status === 'success') {
                    alert(`✅ Predictions loaded! Found ${data.total_count} players`);
                    updateResults(`✅ Predictions test passed: ${data.total_count} players loaded`);
                } else {
                    throw new Error(data.error || 'Predictions endpoint failed');
                }
            } catch (error) {
                console.error('Predictions test failed:', error);
                alert('❌ Predictions test failed: ' + error.message);
                updateResults('❌ Predictions test failed: ' + error.message);
            }
        }

        // Force refresh predictions (clear cache)
        async function forceRefreshPredictions() {
            try {
                updateResults('🔄 Force refreshing predictions...');
                
                const response = await fetch('/predictions?force_refresh=true');
                const data = await response.json();
                
                console.log('Force refresh response:', data);
                
                if (response.ok && data.status === 'success') {
                    alert(`✅ Predictions force refreshed! Found ${data.total_count} players with latest data`);
                    updateResults(`✅ Force refresh successful: ${data.total_count} players with updated team names`);
                    // Auto-load the fresh predictions
                    displayPredictions(data.players, data.total_count);
                } else {
                    throw new Error(data.error || 'Force refresh failed');
                }
            } catch (error) {
                console.error('Force refresh failed:', error);
                alert('❌ Force refresh failed: ' + error.message);
                updateResults('❌ Force refresh failed: ' + error.message);
            }
        }

        // Load and display predictions
        async function loadPredictions() {
            try {
                updatePredictions('🔄 Loading predictions...');
                
                const response = await fetch('/predictions');
                const data = await response.json();
                
                console.log('Predictions loaded:', data);
                
                if (response.ok && data.status === 'success') {
                    displayPredictions(data.players, data.total_count);
                } else {
                    throw new Error(data.error || 'Failed to load predictions');
                }
            } catch (error) {
                console.error('Failed to load predictions:', error);
                updatePredictions('❌ Failed to load predictions: ' + error.message);
            }
        }

        function displayPredictions(players, totalCount) {
            const content = document.getElementById('predictions-content');
            
            if (!players || players.length === 0) {
                content.innerHTML = '❌ No predictions available';
                return;
            }

            // Create a comprehensive display
            let html = `
                <div class="mb-4 flex justify-between items-center">
                    <strong>📊 ${totalCount} total players loaded</strong>
                    <div class="text-sm text-gray-600">Showing top ${Math.min(players.length, 24)} predictions</div>
                </div>
            `;
            
            // Display in a responsive grid
            html += '<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">';
            
            // Show more players (up to 24)
            players.slice(0, 24).forEach((player, index) => {
                const position = getPositionName(player.element_type || player.position_id);
                const predictedPoints = parseFloat(player.predicted_points || 0).toFixed(1);
                
                // Better price handling
                let price = 0;
                if (player.now_cost) {
                    price = parseFloat(player.now_cost);
                } else if (player.price) {
                    price = parseFloat(player.price);
                }
                
                // Use team_name if available, otherwise fall back to team_short
                const teamName = player.team_name || player.team_short || 'Unknown Team';
                const opp = player.next_opponent_short || player.next_opponent || '';
                const ha = player.next_is_home || '';
                const fdr = (player.next_fdr !== null && player.next_fdr !== undefined) ? `FDR ${player.next_fdr}` : '';
                
                // Get additional stats
                const form = player.form || 'N/A';
                const selected = player.selected_by_percent || '0';
                const status = player.fpl_status || 'a';
                const chance = player.chance_next || 100;
                
                // Color coding for positions
                const positionColors = {
                    'GKP': 'bg-yellow-100 text-yellow-800',
                    'DEF': 'bg-green-100 text-green-800', 
                    'MID': 'bg-blue-100 text-blue-800',
                    'FWD': 'bg-red-100 text-red-800'
                };
                const posColor = positionColors[position] || 'bg-gray-100 text-gray-800';
                
                html += `
                    <div class="bg-white rounded-lg p-3 border shadow-sm hover:shadow-md transition-shadow">
                        <div class="flex justify-between items-start mb-2">
                            <div class="font-bold text-gray-800 text-sm">${player.name || 'Unknown'}</div>
                            <div class="text-xs ${posColor} px-2 py-1 rounded font-medium">${position}</div>
                        </div>
                        <div class="text-xs text-gray-600 mb-2">${teamName}${opp ? ` • ${ha}${opp}` : ''} ${fdr}</div>
                        <div class="flex justify-between items-center mb-1">
                            <span class="text-lg font-bold text-purple-600">${predictedPoints}</span>
                            <span class="text-sm text-gray-600">£${price}m</span>
                        </div>
                        <div class="flex justify-between text-xs text-gray-500">
                            <span>Form: ${form}</span>
                            <span>Own: ${selected}%</span>
                        </div>
                        ${chance < 100 ? `<div class="text-xs text-orange-600 font-medium mt-1">Chance: ${chance}%</div>` : ''}
                        ${(player.xg_per90 || player.xa_per90) ? `<div class="text-xs text-gray-600 mt-1">xG/90: ${(player.xg_per90||0).toFixed ? (player.xg_per90||0).toFixed(2) : player.xg_per90} • xA/90: ${(player.xa_per90||0).toFixed ? (player.xa_per90||0).toFixed(2) : player.xa_per90}</div>` : ''}
                    </div>
                `;
            });
            
            html += '</div>';
            
            // Add show more button if there are more players
            if (players.length > 24) {
                html += `
                    <div class="mt-4 text-center">
                        <button onclick="showAllPredictions()" class="btn-primary">
                            Show All ${players.length} Players
                        </button>
                    </div>
                `;
            }
            
            content.innerHTML = html;
            
            // Store players data globally for show all function
            window.allPlayersData = players;
        }
        
        function showAllPredictions() {
            if (window.allPlayersData) {
                displayAllPredictions(window.allPlayersData);
            }
        }
        
        function displayAllPredictions(players) {
            const content = document.getElementById('predictions-content');
            
            let html = `
                <div class="mb-4 flex justify-between items-center">
                    <strong>📊 All ${players.length} Players</strong>
                    <button onclick="loadPredictions()" class="btn-primary text-sm px-4 py-2">Back to Top 24</button>
                </div>
                <div class="max-h-96 overflow-y-auto">
                    <table class="w-full text-sm">
                        <thead class="bg-gray-100 sticky top-0">
                            <tr>
                                <th class="text-left p-2">Player</th>
                                <th class="text-left p-2">Team</th>
                                <th class="text-left p-2">Pos</th>
                                <th class="text-right p-2">Points</th>
                                <th class="text-right p-2">Price</th>
                                <th class="text-right p-2">Form</th>
                                <th class="text-right p-2">Own%</th>
                            </tr>
                        </thead>
                        <tbody>
            `;
            
            players.forEach(player => {
                const position = getPositionName(player.element_type || player.position_id);
                const predictedPoints = parseFloat(player.predicted_points || 0).toFixed(1);
                const price = parseFloat(player.now_cost || player.price || 0);
                const teamName = player.team_name || player.team_short || 'Unknown';
                const opp = player.next_opponent_short || player.next_opponent || '';
                const ha = player.next_is_home || '';
                const fdr = (player.next_fdr !== null && player.next_fdr !== undefined) ? `FDR ${player.next_fdr}` : '';
                const form = player.form || 'N/A';
                const selected = player.selected_by_percent || '0';
                
                html += `
                    <tr class="border-b hover:bg-gray-50">
                        <td class="p-2 font-medium">${player.name || 'Unknown'}</td>
                        <td class="p-2 text-gray-600">${teamName}${opp ? ` • ${ha}${opp}` : ''} ${fdr}</td>
                        <td class="p-2">${position}</td>
                        <td class="p-2 text-right font-bold text-purple-600">${predictedPoints}</td>
                        <td class="p-2 text-right">£${price}m</td>
                        <td class="p-2 text-right">${form}</td>
                        <td class="p-2 text-right">${selected}%</td>
                    </tr>
                `;
            });
            
            html += '</tbody></table></div>';
            content.innerHTML = html;
        }

        function getPositionName(elementType) {
            // Handle both numeric and string position formats
            const positions = {
                1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD',
                'GKP': 'GKP', 'DEF': 'DEF', 'MID': 'MID', 'FWD': 'FWD'
            };
            return positions[elementType] || positions[parseInt(elementType)] || 'UNK';
        }

        // Show cache status
        async function showCacheStatus() {
            try {
                updateResults('💾 Loading cache status...');
                
                const response = await fetch('/cache-status');
                const data = await response.json();
                
                console.log('Cache status:', data);
                
                if (response.ok && data.cache_status) {
                    let statusText = '💾 Cache Status:\\n\\n';
                    
                    for (const [type, status] of Object.entries(data.cache_status)) {
                        const validText = status.valid ? '✅ Valid' : '❌ Expired';
                        const ageText = status.age || 'No data';
                        statusText += `${type.replace('_', ' ')}: ${validText} (Age: ${ageText})\\n`;
                    }
                    
                    alert(statusText);
                    updateResults('✅ Cache status loaded - check console for details');
                } else {
                    throw new Error(data.error || 'Failed to load cache status');
                }
            } catch (error) {
                console.error('Cache status failed:', error);
                updateResults('❌ Cache status failed: ' + error.message);
            }
        }

        // Quick Actions Implementation
        async function quickLogSuggestion() {
            if (globalLoading) return;
            
            try {
                globalLoading = true;
                updateResults('📝 Logging team suggestion...');
                
                const response = await fetch('/quick-actions/log-current-suggestion', { method: 'POST' });
                const data = await response.json();
                
                console.log('Log suggestion response:', data);
                
                if (response.ok && data.status === 'success') {
                    renderSuggestedTeam(data);
                    updateResults(`✅ Team logged for GW${data.gameweek}: ${data.formation}, ${data.predicted_points.toFixed(1)} pts, £${data.total_cost}m`);
                } else {
                    throw new Error(data.error || 'Failed to log suggestion');
                }
            } catch (error) {
                console.error('Log suggestion failed:', error);
                updateResults('❌ Log suggestion failed: ' + error.message);
            } finally {
                globalLoading = false;
            }
        }

        async function quickCollectResults() {
            if (globalLoading) return;
            
            try {
                globalLoading = true;
                updateResults('📊 Collecting results...');
                
                const response = await fetch('/quick-actions/collect-last-results', { method: 'POST' });
                const data = await response.json();
                
                console.log('Collect results response:', data);
                
                if (response.ok && data.status === 'success') {
                    if (data.gameweek === 1) {
                        // Special GW1 handling
                        const message = `📊 ${data.message}\\n\\n${data.note}\\n\\n💡 ${data.advice}`;
                        alert(message);
                        updateResults(`ℹ️ ${data.message} - Log a suggestion first`);
                    } else if (data.total_players) {
                        // GW1 with actual data
                        const message = `📊 ${data.message}:\\nTotal players: ${data.total_players}\\nAverage points: ${data.average_points}\\nTop scorer: ${data.top_scorer}`;
                        alert(message);
                        updateResults(`✅ ${data.message}: ${data.total_players} players, avg ${data.average_points} pts`);
                    } else {
                        // Normal gameweek handling
                        const accuracy = data.predicted_points > 0 ? 
                            ((1 - Math.abs(data.actual_points - data.predicted_points) / data.predicted_points) * 100).toFixed(1) : 'N/A';
                        
                        const message = `📊 Results collected for GW${data.gameweek}:\\nActual: ${data.actual_points} pts\\nPredicted: ${data.predicted_points.toFixed(1)} pts\\nAccuracy: ${accuracy}%\\nCaptain: ${data.captain_points} pts`;
                        alert(message);
                        updateResults(`✅ Results collected for GW${data.gameweek}: ${data.actual_points} actual vs ${data.predicted_points.toFixed(1)} predicted`);
                    }
                } else {
                    throw new Error(data.error || 'Failed to collect results');
                }
            } catch (error) {
                console.error('Collect results failed:', error);
                alert('❌ Failed to collect results: ' + error.message);
                updateResults('❌ Collect results failed: ' + error.message);
            } finally {
                globalLoading = false;
            }
        }

        async function quickGenerateReport() {
            if (globalLoading) return;
            
            try {
                globalLoading = true;
                updateResults('📈 Generating report...');
                
                const response = await fetch('/quick-actions/generate-quick-report', { method: 'POST' });
                const data = await response.json();
                
                console.log('Generate report response:', data);
                
                if (response.ok && data.status === 'success' && data.report) {
                    const report = data.report;
                    if (report.summary) {
                        const summary = report.summary;
                        const message = `📈 Accuracy Report:\\n\\nGameweeks: ${summary.gameweeks_analyzed || 0}\\nAccuracy: ${(summary.prediction_accuracy_pct || 0).toFixed(1)}%\\nAvg Error: ${(summary.average_weekly_error || 0).toFixed(1)}\\n\\nCheck console for full details.`;
                        alert(message);
                        updateResults(`✅ Report generated: ${summary.gameweeks_analyzed || 0} gameweeks, ${(summary.prediction_accuracy_pct || 0).toFixed(1)}% accuracy`);
                    } else {
                        alert('📈 Report generated - check console for details');
                        updateResults('✅ Report generated - check console for full data');
                    }
                } else {
                    throw new Error(data.error || 'Failed to generate report');
                }
            } catch (error) {
                console.error('Generate report failed:', error);
                alert('❌ Failed to generate report: ' + error.message);
                updateResults('❌ Generate report failed: ' + error.message);
            } finally {
                globalLoading = false;
            }
        }

        async function quickAutoImprove() {
            if (globalLoading) return;
            
            try {
                globalLoading = true;
                updateResults('🔧 Auto improving...');
                
                const response = await fetch('/quick-actions/auto-improve', { method: 'POST' });
                const data = await response.json();
                
                console.log('Auto improve response:', data);
                
                if (response.ok && data.status === 'success') {
                    if (data.action === 'model_retrained') {
                        alert('🔧 Model improvements detected and applied!\\nPerformance should improve in upcoming predictions.');
                        updateResults('✅ Model retrained - performance should improve');
                    } else {
                        alert('✅ Model performance is good!\\nNo improvements needed at this time.');
                        updateResults('✅ Model performance is good - no improvements needed');
                    }
                } else {
                    throw new Error(data.error || 'Failed to auto-improve');
                }
            } catch (error) {
                console.error('Auto improve failed:', error);
                alert('❌ Auto-improvement failed: ' + error.message);
                updateResults('❌ Auto improve failed: ' + error.message);
            } finally {
                globalLoading = false;
            }
        }

        async function forceRetrain() {
            if (globalLoading) return;
            
            if (!confirm('⚠️ FORCE RETRAIN WARNING ⚠️\\n\\nThis will completely retrain your model with all available data and clear all caches. This may take several minutes.\\n\\nOnly use this if you suspect the model is not learning properly or after major data issues.\\n\\nProceed?')) {
                return;
            }
            
            try {
                globalLoading = true;
                updateResults('🔴 Force retraining model...');
                
                const response = await fetch('/force-retrain', { method: 'POST' });
                const data = await response.json();
                
                console.log('Force retrain response:', data);
                
                if (response.ok && data.status === 'success') {
                    alert(`🔴 FORCE RETRAIN COMPLETED\\n\\nModel has been completely retrained for GW${data.gameweek}.\\nAll caches cleared.\\n\\n${data.message}`);
                    updateResults(`✅ Force retrain completed for GW${data.gameweek} - all caches cleared`);
                } else {
                    throw new Error(data.error || 'Force retrain failed');
                }
            } catch (error) {
                console.error('Force retrain failed:', error);
                alert('❌ Force retrain failed: ' + error.message);
                updateResults('❌ Force retrain failed: ' + error.message);
            } finally {
                globalLoading = false;
            }
        }

        // Utility functions
        function updateResults(message) {
            const results = document.getElementById('results-content');
            const timestamp = new Date().toLocaleTimeString();
            results.innerHTML += `<div class="mb-2 p-2 bg-gray-50 rounded"><span class="text-xs text-gray-500">${timestamp}</span> - ${message}</div>`;
            results.scrollTop = results.scrollHeight;
        }

        function updatePredictions(message) {
            document.getElementById('predictions-content').innerHTML = message;
        }

        // Auto-run basic test on page load
        window.addEventListener('load', function() {
            updateResults('🚀 Page loaded - JavaScript is working!');
            console.log('✅ FPL AI Dashboard loaded successfully');
            // Load latest suggestion if available
            fetch('/suggestions/latest').then(r => r.json()).then(data => {
                if (data && data.status === 'success') {
                    renderSuggestedTeam(data);
                    updateResults(`✅ Loaded latest suggestion for GW${data.gameweek}`);
                }
            }).catch(() => {});
        });

        // Render suggested team (XI + bench) into the page
        function renderSuggestedTeam(data) {
            const el = document.getElementById('suggested-team');
            if (!el) return;
            const starters = Array.isArray(data.starters) ? data.starters : [];
            const bench = Array.isArray(data.bench) ? data.bench : [];
            const header = `Formation: ${data.formation || ''} • Captain: ${data.captain || ''} • Vice: ${data.vice_captain || ''} • Cost: £${data.total_cost || ''}m`;
            let startersHtml = starters.map((p) => {
                const pos = getPositionName(p.position || p.element_type);
                const teamName = p.team_name || p.team_short || '';
                const pts = (p.predicted_points && p.predicted_points.toFixed) ? p.predicted_points.toFixed(1) : (p.predicted_points || 0);
                return `
                    <div class="bg-white rounded p-2 border">
                        <div class="text-sm font-semibold">${p.name}</div>
                        <div class="text-xs text-gray-600">${pos} • ${teamName}</div>
                        <div class="text-xs text-purple-700">${pts} pts</div>
                    </div>`;
            }).join('');
            let benchHtml = bench.map((p) => {
                const pos = getPositionName(p.position || p.element_type);
                return `<div class="text-xs bg-gray-50 p-2 rounded border">${p.name} (${pos})</div>`;
            }).join('');
            el.innerHTML = `
                <div class="text-sm text-gray-700 mb-2">${header}</div>
                <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-2 mb-3">${startersHtml}</div>
                <div class="mt-2"><span class="font-medium">Bench:</span>
                    <div class="mt-1 flex gap-2 flex-wrap">${benchHtml || '<span class="text-xs text-gray-500">No bench data</span>'}</div>
                </div>
            `;
        }
    </script>

</body>
</html>
"""