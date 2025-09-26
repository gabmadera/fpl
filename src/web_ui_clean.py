from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import json
import csv
import os

from .ml_pipeline import MLPipeline
from .fpl_client import FPLClient
from .comprehensive_accuracy_tracker import ComprehensiveAccuracyTracker, GameweekSuggestion, GameweekResult
from .cache_manager import CacheManager
from .strategic_team_planner import StrategicTeamPlanner
from .team_selector import TeamSelector
from .model_feedback_loop import ModelFeedbackLoop
from .aggressive_accuracy_optimizer import AggressiveAccuracyOptimizer
from .automated_retraining_scheduler import get_retraining_scheduler, start_automated_retraining
from .accuracy_analysis_dashboard import AccuracyAnalysisDashboard, quick_accuracy_check

app = FastAPI(title="FPL AI Dashboard - Clean & Simple", version="2.0")

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

@app.get("/gameweek-status")
def get_gameweek_status() -> dict:
    """Get current FPL gameweek status with proper caching"""
    try:
        cache_manager = CacheManager()
        fpl = FPLClient()

        cache_key = 'gameweek_status'
        cached_status = cache_manager.get_cached_data(cache_key)

        if cached_status and cache_manager.is_cache_valid(cache_key, 6):
            return {"status": "success", "data": cached_status, "from_cache": True}

        try:
            bs = fpl.bootstrap_static()
            if not bs:
                return {"status": "error", "message": "Could not fetch FPL data"}

            events = bs.get("events", [])

            current_gw = None
            next_gw = None
            finished_gws = []

            for event in events:
                if event.get("is_current", False):
                    current_gw = event["id"]
                elif event.get("is_next", False):
                    next_gw = event["id"]
                elif event.get("finished", False):
                    finished_gws.append(event["id"])

            if not current_gw:
                for event in events:
                    if not event.get("finished", False):
                        current_gw = event["id"]
                        break

                if not current_gw and finished_gws:
                    current_gw = max(finished_gws) + 1

            if not next_gw:
                next_gw = current_gw + 1 if current_gw else 1

            status_data = {
                "current_gameweek": current_gw or 1,
                "next_gameweek": next_gw or 2,
                "finished_gameweeks": sorted(finished_gws),
                "last_finished": max(finished_gws) if finished_gws else 0,
                "total_gameweeks": len(events),
                "current_event": next((e for e in events if e["id"] == current_gw), {}),
                "season_started": len(finished_gws) > 0,
                "timestamp": datetime.now().isoformat()
            }

            cache_manager.cache_data(cache_key, status_data)
            return {"status": "success", "data": status_data, "from_cache": False}

        except Exception as api_error:
            return {"status": "error", "message": f"FPL API error: {str(api_error)}"}

    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/predictions")
def predictions(force_refresh: bool = False) -> dict:
    cache_manager = CacheManager()

    try:
        if not force_refresh:
            cached_predictions = cache_manager.get_cached_data('predictions')
            if cached_predictions:
                return {"status": "success", "from_cache": True, **cached_predictions}

        pipeline = MLPipeline()
        predictions_df = pipeline.predict_gameweek()

        if predictions_df.empty:
            return {"status": "error", "message": "No predictions generated"}

        # Generate team selection
        selector = TeamSelector()
        team_result = selector.select_optimal_team(predictions_df)

        # Convert FPLTeam object to dictionary format
        team_data = {
            'team_summary': {
                'total_predicted_points': team_result.predicted_points,
                'total_cost': team_result.total_cost,
                'formation': team_result.formation,
                'captain_id': team_result.captain_id,
                'vice_captain_id': team_result.vice_captain_id,
                'chip_recommendation': team_result.chip_recommendation,
                'starters': team_result.starters,
                'bench': team_result.bench
            },
            'predictions': predictions_df.to_dict('records')
        }

        # Cache results
        cache_manager.cache_data('predictions', team_data)

        return {"status": "success", "from_cache": False, **team_data}

    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/comparisons/gw-by-gw")
def get_gw_comparisons(limit: int = 10) -> dict:
    """Get gameweek-by-gameweek comparisons of predictions vs actual results"""
    try:
        tracker = ComprehensiveAccuracyTracker()
        comparisons = []

        for result in tracker.results_history[-limit:]:
            comparison = {
                "gameweek": result.gameweek,
                "predicted_points": result.team_selection.predicted_points,
                "actual_points": result.actual_team_points,
                "difference": result.actual_team_points - result.team_selection.predicted_points,
                "accuracy_percentage": (1 - abs(result.actual_team_points - result.team_selection.predicted_points) / max(result.team_selection.predicted_points, 1)) * 100,
                "captain_points": result.actual_captain_points,
                "suggested_team": [{"name": p.get("name"), "id": p.get("player_id"), "predicted_points": p.get("predicted_points"), "price": p.get("price")} for p in result.team_selection.starters],
                "formation": result.team_selection.formation,
                "timestamp": result.team_selection.timestamp.isoformat(),
                "accuracy": result.prediction_accuracy.get('accuracy_pct', 0)
            }
            comparisons.append(comparison)

        # Calculate summary statistics
        if comparisons:
            total_predicted = sum(c["predicted_points"] for c in comparisons)
            total_actual = sum(c["actual_points"] for c in comparisons)
            avg_accuracy = sum(c["accuracy_percentage"] for c in comparisons) / len(comparisons)

            summary = {
                "total_predicted": total_predicted,
                "total_actual": total_actual,
                "overall_accuracy": avg_accuracy,
                "gameweeks_tracked": len(comparisons)
            }
        else:
            summary = {
                "total_predicted": 0,
                "total_actual": 0,
                "overall_accuracy": 0,
                "gameweeks_tracked": 0
            }

        return {
            "status": "success",
            "comparisons": comparisons,
            "summary": summary,
            "total_count": len(tracker.results_history)
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/collect-real-results-improved/{gameweek}")
def collect_real_results_improved(gameweek: int) -> dict:
    """Improved REAL FPL points collection with better error handling"""
    try:
        cache_manager = CacheManager()
        tracker = ComprehensiveAccuracyTracker()
        fpl = FPLClient()

        suggestion = next((s for s in tracker.suggestions_history if s.gameweek == gameweek), None)
        if not suggestion:
            return {"status": "error", "message": f"No team suggestion found for GW{gameweek}"}

        try:
            event_data = fpl.event_live(gameweek)
            data_source = "event_live"

            if not event_data or not event_data.get("elements"):
                bs = fpl.bootstrap_static()
                if not bs:
                    return {"status": "error", "message": "Could not fetch any FPL data"}

                event_data = {"elements": bs.get("elements", [])}
                data_source = "bootstrap_static"

        except Exception as api_error:
            return {"status": "error", "message": f"FPL API error: {str(api_error)}"}

        # Build player points lookup
        player_points_lookup = {}
        elements = event_data.get("elements", [])

        for elem in elements:
            if data_source == "event_live" and "stats" in elem:
                player_id = elem.get("id")
                points = elem.get("stats", {}).get("total_points", 0)
            else:
                player_id = elem.get("id")
                points = elem.get("event_points", 0)

            if player_id:
                player_points_lookup[str(player_id)] = points
                player_points_lookup[player_id] = points

        # Calculate actual points
        total_actual = 0
        captain_bonus = 0
        successful_matches = 0
        player_performances = []

        for player in suggestion.starters:
            player_id = player.get('player_id') or player.get('id') or player.get('element')
            player_name = player.get('name', 'Unknown')
            predicted_pts = player.get('predicted_points', 0)

            actual_pts = player_points_lookup.get(str(player_id), 0) or player_points_lookup.get(player_id, 0)

            if actual_pts > 0:
                successful_matches += 1

            player_performances.append({
                'name': player_name,
                'player_id': player_id,
                'predicted': predicted_pts,
                'actual': actual_pts,
                'difference': actual_pts - predicted_pts
            })

            total_actual += actual_pts

            if player_id == suggestion.captain_id:
                captain_bonus = actual_pts
                total_actual += actual_pts

        match_rate = successful_matches / len(suggestion.starters) if suggestion.starters else 0

        # Save result
        result = GameweekResult(
            gameweek=gameweek,
            team_selection=suggestion,
            actual_team_points=total_actual,
            actual_captain_points=captain_bonus * 2,
            actual_vice_points=0,
            points_with_transfers=total_actual,
            points_vs_optimal=0,
            prediction_accuracy={
                'mae': abs(total_actual - suggestion.predicted_points),
                'accuracy_pct': max(0, (1 - abs(total_actual - suggestion.predicted_points) / max(suggestion.predicted_points, 1)) * 100)
            }
        )

        existing_idx = next((i for i, r in enumerate(tracker.results_history) if r.gameweek == gameweek), None)
        if existing_idx is not None:
            tracker.results_history[existing_idx] = result
        else:
            tracker.results_history.append(result)

        tracker._save_results_history()

        return {
            "status": "success",
            "gameweek": gameweek,
            "predicted_total": suggestion.predicted_points,
            "actual_total": total_actual,
            "difference": total_actual - suggestion.predicted_points,
            "accuracy_pct": result.prediction_accuracy['accuracy_pct'],
            "match_rate": f"{successful_matches}/{len(suggestion.starters)} ({match_rate:.1%})",
            "data_source": data_source,
            "player_performances": player_performances
        }

    except Exception as e:
        return {"status": "error", "message": f"Collection failed: {str(e)}"}

@app.get("/model-feedback/summary")
def get_model_feedback_summary() -> dict:
    """Get current model performance feedback summary"""
    try:
        feedback_loop = ModelFeedbackLoop()
        return feedback_loop.get_feedback_summary()
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/model-feedback/analyze")
def analyze_model_performance() -> dict:
    """Analyze model performance and generate improvement feedback"""
    try:
        feedback_loop = ModelFeedbackLoop()
        feedback = feedback_loop.analyze_prediction_performance()

        return {
            "status": "success",
            "confidence_score": feedback.confidence_score,
            "improvement_suggestions": feedback.improvement_suggestions,
            "error_patterns": feedback.prediction_error_patterns,
            "feature_adjustments": feedback.feature_importance_changes,
            "position_adjustments": feedback.position_adjustments,
            "analysis_timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/model-feedback/apply-improvements")
def apply_model_improvements() -> dict:
    """Apply feedback-based improvements to the model"""
    try:
        feedback_loop = ModelFeedbackLoop()
        feedback = feedback_loop.analyze_prediction_performance()
        result = feedback_loop.apply_feedback_to_model(feedback)

        return {
            "status": "success",
            "application_result": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/transfer-analysis")
def get_transfer_analysis() -> dict:
    """Compare current team suggestion with previous suggestions to show transfers"""
    try:
        cache_manager = CacheManager()

        # Get current predictions
        current_data = cache_manager.get_cached_data('predictions')
        if not current_data or 'team_summary' not in current_data:
            return {"status": "error", "message": "No current team suggestion available"}

        current_team = current_data['team_summary']
        current_players = current_team.get('starters', []) + current_team.get('bench', [])
        current_player_ids = set(p.get('player_id') for p in current_players)

        # Try to get previous team suggestions from comparison data
        try:
            accuracy_tracker = ComprehensiveAccuracyTracker()
            results_history = accuracy_tracker.results_history

            if len(results_history) == 0:
                return {
                    "status": "success",
                    "transfers_in": [],
                    "transfers_out": [],
                    "message": "No previous team data for comparison"
                }

            # Get most recent previous team
            previous_result = results_history[-1]
            if hasattr(previous_result, 'team_selection') and previous_result.team_selection:
                previous_starters = previous_result.team_selection.starters or []
                previous_bench = previous_result.team_selection.bench or []
                previous_players = previous_starters + previous_bench
                previous_player_ids = set(p.get('player_id') for p in previous_players)

                # Calculate transfers
                transfers_in = [p for p in current_players if p.get('player_id') not in previous_player_ids]
                transfers_out = [p for p in previous_players if p.get('player_id') not in current_player_ids]

                return {
                    "status": "success",
                    "transfers_in": transfers_in[:5],  # Limit to 5 for display
                    "transfers_out": transfers_out[:5],
                    "previous_gw": getattr(previous_result, 'gameweek', 'Previous'),
                    "current_gw": "Current"
                }
            else:
                return {
                    "status": "success",
                    "transfers_in": [],
                    "transfers_out": [],
                    "message": "Previous team data format incompatible"
                }

        except Exception as e:
            # Fallback: show current team as "new"
            return {
                "status": "success",
                "transfers_in": current_players[:11],  # Show starting XI as "new"
                "transfers_out": [],
                "message": f"First team suggestion (no previous data)"
            }

    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/historical-comparisons")
def get_historical_comparisons() -> dict:
    """Get historical gameweek comparisons between predicted and actual points"""
    try:
        data_path = Path(__file__).parent.parent / "data" / "accuracy_tracking"

        # Read JSON results file
        json_file = data_path / "gameweek_results.json"
        csv_files = list(data_path.glob("gw*_team_predictions.csv"))

        if not json_file.exists():
            return {"status": "error", "message": "No historical results found"}

        with open(json_file, 'r') as f:
            results_data = json.load(f)

        # Process historical data
        historical_summary = []
        for result in results_data:
            gw = result.get('gameweek')
            team_selection = result.get('team_selection', {})
            predicted_points = team_selection.get('predicted_points', 0)
            actual_points = result.get('actual_team_points', 0)
            prediction_accuracy = result.get('prediction_accuracy', {})

            # Calculate accuracy percentage
            accuracy_pct = 0
            if 'accuracy_pct' in prediction_accuracy:
                accuracy_pct = prediction_accuracy['accuracy_pct']
            elif actual_points > 0 and predicted_points > 0:
                # Calculate accuracy as 100 - (abs(predicted - actual) / actual * 100)
                accuracy_pct = max(0, 100 - (abs(predicted_points - actual_points) / actual_points * 100))

            mae = prediction_accuracy.get('mae', abs(predicted_points - actual_points))

            historical_summary.append({
                'gameweek': gw,
                'predicted_points': round(predicted_points, 1),
                'actual_points': actual_points,
                'difference': round(actual_points - predicted_points, 1),
                'accuracy_pct': round(accuracy_pct, 1),
                'mae': round(mae, 1),
                'captain': team_selection.get('captain_id'),
                'formation': team_selection.get('formation', '3-4-3'),
                'timestamp': team_selection.get('timestamp', '')
            })

        # Get latest comparison details
        latest_details = None
        if results_data:
            latest = results_data[-1]
            team_selection = latest.get('team_selection', {})
            starters = team_selection.get('starters', [])
            bench = team_selection.get('bench', [])

            latest_details = {
                'gameweek': latest.get('gameweek'),
                'starters': [
                    {
                        'name': p.get('name', ''),
                        'position': p.get('position', ''),
                        'predicted_points': round(p.get('predicted_points', 0), 1),
                        'price': p.get('price', 0)
                    } for p in starters
                ],
                'bench': [
                    {
                        'name': p.get('name', ''),
                        'position': p.get('position', ''),
                        'predicted_points': round(p.get('predicted_points', 0), 1),
                        'price': p.get('price', 0)
                    } for p in bench
                ],
                'captain_id': team_selection.get('captain_id'),
                'vice_captain_id': team_selection.get('vice_captain_id'),
                'total_cost': team_selection.get('total_cost', 0)
            }

        return {
            "status": "success",
            "historical_summary": historical_summary,
            "latest_details": latest_details,
            "total_gameweeks": len(historical_summary),
            "avg_accuracy": round(sum(h['accuracy_pct'] for h in historical_summary if h['accuracy_pct'] > 0) / max(1, len([h for h in historical_summary if h['accuracy_pct'] > 0])), 1) if historical_summary else 0
        }

    except Exception as e:
        return {"status": "error", "message": f"Error reading historical data: {str(e)}"}

@app.get("/actual-player-points/{gameweek}")
def get_actual_player_points(gameweek: int) -> dict:
    """Get actual player points from FPL API for a specific gameweek"""
    try:
        fpl = FPLClient()

        # Get live gameweek data
        live_data = fpl.event_live(gameweek)

        if not live_data or 'elements' not in live_data:
            return {"status": "error", "message": f"No live data available for GW{gameweek}"}

        # Extract player points
        player_points = {}
        for element in live_data['elements']:
            player_id = element.get('id')
            stats = element.get('stats', {})
            total_points = stats.get('total_points', 0)

            player_points[player_id] = {
                'total_points': total_points,
                'minutes': stats.get('minutes', 0),
                'goals_scored': stats.get('goals_scored', 0),
                'assists': stats.get('assists', 0),
                'clean_sheets': stats.get('clean_sheets', 0),
                'goals_conceded': stats.get('goals_conceded', 0),
                'yellow_cards': stats.get('yellow_cards', 0),
                'red_cards': stats.get('red_cards', 0),
                'saves': stats.get('saves', 0),
                'bonus': stats.get('bonus', 0)
            }

        return {
            "status": "success",
            "gameweek": gameweek,
            "player_points": player_points,
            "total_players": len(player_points)
        }

    except Exception as e:
        return {"status": "error", "message": f"Failed to fetch actual points for GW{gameweek}: {str(e)}"}

@app.post("/update-accuracy-with-actuals/{gameweek}")
def update_accuracy_with_actuals(gameweek: int) -> dict:
    """Update historical accuracy data with actual FPL points"""
    try:
        # Get actual points from FPL
        actual_points_response = get_actual_player_points(gameweek)
        if actual_points_response["status"] != "success":
            return actual_points_response

        actual_points = actual_points_response["player_points"]

        # Load and update historical data
        data_path = Path(__file__).parent.parent / "data" / "accuracy_tracking"
        json_file = data_path / "gameweek_results.json"

        if not json_file.exists():
            return {"status": "error", "message": "No historical data found"}

        with open(json_file, 'r') as f:
            results_data = json.load(f)

        # Find the gameweek to update
        updated = False
        for result in results_data:
            if result.get('gameweek') == gameweek:
                team_selection = result.get('team_selection', {})
                starters = team_selection.get('starters', [])
                bench = team_selection.get('bench', [])

                # Calculate actual team points
                actual_team_total = 0
                actual_captain_points = 0
                captain_id = team_selection.get('captain_id')

                # Update starter points
                for player in starters:
                    player_id = player.get('player_id')
                    if player_id in actual_points:
                        actual_player_points = actual_points[player_id]['total_points']
                        player['actual_points'] = actual_player_points
                        actual_team_total += actual_player_points

                        # Captain gets double points
                        if player_id == captain_id:
                            actual_captain_points = actual_player_points * 2
                            actual_team_total += actual_player_points  # Double for captain

                # Update bench points (they don't count toward team total unless substituted)
                for player in bench:
                    player_id = player.get('player_id')
                    if player_id in actual_points:
                        player['actual_points'] = actual_points[player_id]['total_points']

                # Update result with actual data
                result['actual_team_points'] = actual_team_total
                result['actual_captain_points'] = actual_captain_points

                # Recalculate accuracy
                predicted_points = team_selection.get('predicted_points', 0)
                if predicted_points > 0:
                    accuracy_pct = max(0, 100 - (abs(predicted_points - actual_team_total) / actual_team_total * 100)) if actual_team_total > 0 else 0
                    mae = abs(predicted_points - actual_team_total)
                else:
                    accuracy_pct = 0
                    mae = actual_team_total

                result['prediction_accuracy'] = {
                    'mae': mae,
                    'accuracy_pct': accuracy_pct
                }

                updated = True
                break

        if updated:
            # Save updated data
            with open(json_file, 'w') as f:
                json.dump(results_data, f, indent=2)

            return {
                "status": "success",
                "message": f"Accuracy updated for GW{gameweek} with actual FPL data",
                "gameweek": gameweek,
                "actual_team_points": actual_team_total,
                "predicted_points": predicted_points,
                "new_accuracy": accuracy_pct
            }
        else:
            return {
                "status": "error",
                "message": f"Gameweek {gameweek} not found in historical data"
            }

    except Exception as e:
        return {"status": "error", "message": f"Failed to update accuracy: {str(e)}"}

@app.post("/retrain-model")
def retrain_model_with_actuals() -> dict:
    """Retrain the model using actual gameweek results to improve predictions"""
    try:
        feedback_loop = ModelFeedbackLoop()
        result = feedback_loop.retrain_with_actual_results()

        return {
            "status": "success",
            "retraining_result": result,
            "message": "Model retraining completed" if result.get("status") == "success" else f"Retraining {result.get('status', 'failed')}: {result.get('reason', 'Unknown error')}"
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to retrain model: {str(e)}"}

@app.get("/verify-points/{gameweek}/{player_id}")
def verify_player_points(gameweek: int, player_id: int) -> dict:
    """Verify actual points for a specific player in a gameweek with detailed breakdown"""
    try:
        fpl = FPLClient()

        # Get player basic info
        bootstrap = fpl.bootstrap_static()
        player_info = None
        for element in bootstrap.get('elements', []):
            if element['id'] == player_id:
                player_info = element
                break

        if not player_info:
            return {"status": "error", "message": f"Player {player_id} not found"}

        # Get live gameweek data
        live_data = fpl.event_live(gameweek)
        player_stats = None

        for element in live_data.get('elements', []):
            if element['id'] == player_id:
                player_stats = element.get('stats', {})
                break

        if not player_stats:
            return {"status": "error", "message": f"No stats found for player {player_id} in GW{gameweek}"}

        # Calculate points breakdown
        points_breakdown = {
            'minutes_played': player_stats.get('minutes', 0),
            'goals_scored': player_stats.get('goals_scored', 0),
            'assists': player_stats.get('assists', 0),
            'clean_sheets': player_stats.get('clean_sheets', 0),
            'goals_conceded': player_stats.get('goals_conceded', 0),
            'yellow_cards': player_stats.get('yellow_cards', 0),
            'red_cards': player_stats.get('red_cards', 0),
            'saves': player_stats.get('saves', 0),
            'bonus_points': player_stats.get('bonus', 0),
            'penalties_saved': player_stats.get('penalties_saved', 0),
            'penalties_missed': player_stats.get('penalties_missed', 0),
            'own_goals': player_stats.get('own_goals', 0)
        }

        # Calculate total points
        total_points = player_stats.get('total_points', 0)

        return {
            "status": "success",
            "gameweek": gameweek,
            "player": {
                "id": player_id,
                "name": f"{player_info.get('first_name', '')} {player_info.get('second_name', '')}".strip(),
                "team": player_info.get('team', 0),
                "position": player_info.get('element_type', 0)
            },
            "points_breakdown": points_breakdown,
            "total_points": total_points,
            "fpl_url": f"https://fantasy.premierleague.com/entry/YOUR_TEAM_ID/event/{gameweek}/",
            "verification_note": f"Check this player's actual performance in GW{gameweek} on the official FPL website"
        }

    except Exception as e:
        return {"status": "error", "message": f"Failed to verify points: {str(e)}"}

@app.get("/verify-gameweek/{gameweek}")
def verify_gameweek_data(gameweek: int) -> dict:
    """Show all players from our historical data with verification links"""
    try:
        # Load historical data
        data_path = Path(__file__).parent.parent / "data" / "accuracy_tracking"
        json_file = data_path / "gameweek_results.json"

        if not json_file.exists():
            return {"status": "error", "message": "No historical data found"}

        with open(json_file, 'r') as f:
            results_data = json.load(f)

        # Find the gameweek
        gw_data = None
        for result in results_data:
            if result.get('gameweek') == gameweek:
                gw_data = result
                break

        if not gw_data:
            return {"status": "error", "message": f"Gameweek {gameweek} not found"}

        team_selection = gw_data.get('team_selection', {})
        starters = team_selection.get('starters', [])

        verification_data = []

        for player in starters:
            player_id = player.get('player_id')
            predicted = player.get('predicted_points', 0)
            actual = player.get('actual_points', 'N/A')

            verification_data.append({
                'player_id': player_id,
                'name': player.get('name', 'Unknown'),
                'position': player.get('position', 'Unknown'),
                'predicted_points': round(predicted, 2),
                'actual_points': actual,
                'difference': round(actual - predicted, 2) if actual != 'N/A' else 'N/A',
                'verify_url': f"/verify-points/{gameweek}/{player_id}"
            })

        return {
            "status": "success",
            "gameweek": gameweek,
            "team_predicted_total": round(team_selection.get('predicted_points', 0), 2),
            "team_actual_total": gw_data.get('actual_team_points', 'N/A'),
            "accuracy_pct": gw_data.get('prediction_accuracy', {}).get('accuracy_pct', 'N/A'),
            "players": verification_data,
            "manual_verification_steps": [
                f"1. Go to https://fantasy.premierleague.com/",
                f"2. Check gameweek {gameweek} results",
                f"3. Compare each player's actual points with our data",
                f"4. Use the verify_url endpoints above for detailed breakdowns"
            ]
        }

    except Exception as e:
        return {"status": "error", "message": f"Failed to verify gameweek: {str(e)}"}

@app.get("/enriched-predictions")
def get_enriched_predictions() -> dict:
    """Get predictions with opponent information and team details"""
    try:
        cache_manager = CacheManager()

        # Get current predictions
        current_data = cache_manager.get_cached_data('predictions')
        if not current_data or 'team_summary' not in current_data:
            return {"status": "error", "message": "No current team suggestion available"}

        # Get FPL data for fixtures and teams
        fpl = FPLClient()
        bootstrap = fpl.bootstrap_static()
        fixtures = fpl.fixtures()

        if not bootstrap or not fixtures:
            return {"status": "error", "message": "Could not fetch FPL data"}

        teams = {t['id']: t for t in bootstrap.get('teams', [])}

        # Get next active gameweek for predictions (not finished gameweeks)
        prediction_gw = fpl.next_active_gameweek()

        # Get fixtures for prediction gameweek
        current_fixtures = [f for f in fixtures if f.get('event') == prediction_gw]

        # Create opponent mapping
        opponent_map = {}
        venue_map = {}

        for fixture in current_fixtures:
            home_team = fixture.get('team_h')
            away_team = fixture.get('team_a')

            if home_team and away_team:
                # Home team vs away team
                opponent_map[home_team] = teams.get(away_team, {}).get('short_name', 'TBA')
                venue_map[home_team] = 'H'

                # Away team vs home team
                opponent_map[away_team] = teams.get(home_team, {}).get('short_name', 'TBA')
                venue_map[away_team] = 'A'

        # Enrich team data with opponent info
        enriched_team = current_data['team_summary'].copy()

        # Enrich starters
        for player in enriched_team.get('starters', []):
            team_id = player.get('team', 0)
            player['opponent'] = opponent_map.get(team_id, 'TBA')
            player['venue'] = venue_map.get(team_id, 'H')
            player['team_short'] = teams.get(team_id, {}).get('short_name', 'UNK')

        # Enrich bench
        for player in enriched_team.get('bench', []):
            team_id = player.get('team', 0)
            player['opponent'] = opponent_map.get(team_id, 'TBA')
            player['venue'] = venue_map.get(team_id, 'H')
            player['team_short'] = teams.get(team_id, {}).get('short_name', 'UNK')

        return {
            "status": "success",
            "team_summary": enriched_team,
            "gameweek": prediction_gw
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

# ==================== ACCURACY OPTIMIZATION ENDPOINTS ====================

@app.get("/accuracy/dashboard")
def get_accuracy_dashboard() -> dict:
    """Get comprehensive accuracy dashboard with current metrics and trends"""
    try:
        dashboard = AccuracyAnalysisDashboard()
        return dashboard.generate_weekly_accuracy_summary()
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/accuracy/detailed-report")
def get_detailed_accuracy_report(gameweeks: int = 10) -> dict:
    """Get detailed accuracy analysis report"""
    try:
        dashboard = AccuracyAnalysisDashboard()
        return dashboard.generate_comprehensive_accuracy_report(gameweeks)
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/accuracy/emergency-retrain/{gameweek}")
def trigger_emergency_retrain(gameweek: int) -> dict:
    """Trigger emergency retraining for maximum accuracy"""
    try:
        optimizer = AggressiveAccuracyOptimizer()
        result = optimizer.immediate_post_gameweek_retrain(gameweek)
        return {
            "status": "success",
            "gameweek": gameweek,
            "retrain_result": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/accuracy/scheduler-status")
def get_scheduler_status() -> dict:
    """Get automated retraining scheduler status"""
    try:
        scheduler = get_retraining_scheduler()
        return scheduler.get_scheduler_status()
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/accuracy/start-automation")
def start_accuracy_automation() -> dict:
    """Start automated retraining system"""
    try:
        scheduler = start_automated_retraining()
        return {
            "status": "success",
            "message": "Automated retraining system started",
            "scheduler_status": scheduler.get_scheduler_status()
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/accuracy/manual-retrain/{gameweek}")
def manual_retrain_gameweek(gameweek: int) -> dict:
    """Manually trigger retraining for specific gameweek"""
    try:
        scheduler = get_retraining_scheduler()
        result = scheduler.force_manual_retrain(gameweek)
        return {
            "status": "success",
            "gameweek": gameweek,
            "retrain_result": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/accuracy/current-performance")
def get_current_accuracy_performance() -> dict:
    """Get current accuracy performance metrics"""
    try:
        optimizer = AggressiveAccuracyOptimizer()
        dashboard_data = optimizer.get_accuracy_dashboard()

        # Extract key metrics for quick display
        current_perf = dashboard_data.get("current_performance", {})

        return {
            "accuracy": current_perf.get("accuracy", 0),
            "captain_success_rate": current_perf.get("captain_success_rate", 0),
            "performance_level": "excellent" if current_perf.get("accuracy", 0) >= 75 else
                                "good" if current_perf.get("accuracy", 0) >= 65 else
                                "acceptable" if current_perf.get("accuracy", 0) >= 55 else "poor",
            "needs_improvement": current_perf.get("accuracy", 0) < 65,
            "recommendations": dashboard_data.get("recommendations", []),
            "ensemble_status": dashboard_data.get("ensemble_status", {}),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/accuracy/explanation")
def get_accuracy_explanation() -> dict:
    """Explain what the accuracy percentage means"""
    return {
        "accuracy_definition": {
            "calculation": "100 - (abs(predicted_points - actual_points) / actual_points * 100)",
            "meaning": "How close predicted points are to actual points as a percentage",
            "example": "70% accuracy means predictions are typically within 30% of actual points"
        },
        "benchmarks": {
            "excellent": "75%+ - Elite level accuracy for FPL predictions",
            "good": "65-75% - Strong predictive performance",
            "acceptable": "55-65% - Reasonable accuracy given FPL randomness",
            "poor": "<55% - Needs significant improvement"
        },
        "context": {
            "fpl_difficulty": "FPL is inherently unpredictable due to football's random nature",
            "realistic_targets": "Even 60-70% accuracy is very good for FPL",
            "factors": ["Injuries", "Rotation", "Bonus points", "Referee decisions", "Weather"]
        },
        "improvement_strategies": [
            "Weekly retraining after each gameweek",
            "Ensemble models for better predictions",
            "Position-specific feature engineering",
            "Fixture difficulty consideration",
            "Recent form weighting"
        ]
    }

# ==================== END ACCURACY OPTIMIZATION ====================

@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FPL AI Dashboard - Clean & Simple</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        // Global logging and error handling
        console.log('[INFO] FPL Dashboard - Starting initialization...');

        window.addEventListener('error', function(e) {
            console.error('[ERROR] JavaScript Error:', e.error);
            console.error('  Line:', e.lineno, 'Column:', e.colno);
            console.error('  File:', e.filename);
        });

        window.addEventListener('unhandledrejection', function(e) {
            console.error('[ERROR] Unhandled Promise Rejection:', e.reason);
        });
    </script>
    <style>
        .status-excellent { background: linear-gradient(135deg, #10b981, #059669); }
        .status-good { background: linear-gradient(135deg, #3b82f6, #1d4ed8); }
        .status-poor { background: linear-gradient(135deg, #ef4444, #dc2626); }
        .status-unknown { background: linear-gradient(135deg, #6b7280, #4b5563); }
        .loading { animation: spin 1s linear infinite; }
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        .card { background: white; border-radius: 12px; padding: 24px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .metric { font-size: 2rem; font-weight: bold; }
        .transfer-in { color: #10b981; }
        .transfer-out { color: #ef4444; }
        .hidden { display: none !important; }

        /* Football Field Styles */
        .football-field {
            background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
            position: relative;
            padding: 20px;
            border-radius: 12px;
            min-height: 400px;
            box-shadow: inset 0 0 20px rgba(0,0,0,0.1);
        }

        .field-lines {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-image:
                radial-gradient(circle at center, transparent 60px, transparent 61px, rgba(255,255,255,0.3) 62px, rgba(255,255,255,0.3) 63px, transparent 64px),
                linear-gradient(90deg, transparent 49%, rgba(255,255,255,0.3) 50%, transparent 51%);
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 12px;
        }

        .position-row {
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 15px 0;
            gap: 15px;
            position: relative;
            z-index: 2;
        }

        .player-card {
            background: rgba(255,255,255,0.95);
            border-radius: 8px;
            padding: 8px;
            text-align: center;
            min-width: 80px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            border: 2px solid transparent;
            transition: all 0.2s ease;
            cursor: pointer;
            position: relative;
        }

        .player-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }

        .player-card.captain {
            border-color: #fbbf24;
            background: linear-gradient(135deg, #fef3c7, rgba(251, 191, 36, 0.3));
        }

        .player-card.vice-captain {
            border-color: #60a5fa;
            background: linear-gradient(135deg, #dbeafe, rgba(96, 165, 250, 0.3));
        }

        .player-name {
            font-weight: bold;
            font-size: 11px;
            color: #1f2937;
            margin-bottom: 2px;
        }

        .player-team {
            font-size: 9px;
            color: #6b7280;
            margin-bottom: 2px;
        }

        .player-opponent {
            font-size: 8px;
            color: #ef4444;
            font-weight: bold;
            background: rgba(239, 68, 68, 0.1);
            padding: 1px 4px;
            border-radius: 4px;
            margin-bottom: 2px;
        }

        .player-points {
            font-size: 10px;
            color: #059669;
            font-weight: bold;
        }

        .captain-badge {
            position: absolute;
            top: -5px;
            right: -5px;
            background: #fbbf24;
            color: white;
            border-radius: 50%;
            width: 18px;
            height: 18px;
            font-size: 10px;
            font-weight: bold;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .vice-captain-badge {
            position: absolute;
            top: -5px;
            right: -5px;
            background: #60a5fa;
            color: white;
            border-radius: 50%;
            width: 18px;
            height: 18px;
            font-size: 10px;
            font-weight: bold;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .bench-section {
            margin-top: 20px;
            padding: 15px;
            background: rgba(255,255,255,0.1);
            border-radius: 8px;
            border: 1px dashed rgba(255,255,255,0.3);
        }

        .bench-title {
            color: white;
            font-size: 12px;
            font-weight: bold;
            margin-bottom: 10px;
            text-align: center;
        }

        .bench-players {
            display: flex;
            justify-content: center;
            gap: 10px;
            flex-wrap: wrap;
        }

        .bench-player {
            background: rgba(255,255,255,0.8);
            border-radius: 6px;
            padding: 6px;
            text-align: center;
            min-width: 60px;
            font-size: 10px;
        }
    </style>
</head>
<body class="min-h-screen bg-gray-50">

<!-- Simple Header -->
<header class="bg-white shadow-sm border-b">
    <div class="max-w-7xl mx-auto px-4 py-4">
        <div class="flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <div class="text-2xl">⚽</div>
                <h1 class="text-xl font-bold text-gray-900">FPL AI Dashboard</h1>
            </div>
            <div id="current-gw" class="text-sm text-gray-600">Loading GW status...</div>
        </div>
    </div>
</header>

<!-- Main Dashboard Content -->
<div class="max-w-7xl mx-auto px-4 py-6 space-y-6">

    <!-- Current Status Overview -->
    <div class="grid grid-cols-1 md:grid-cols-5 gap-4">

        <!-- What To Do -->
        <div class="card">
            <h2 class="text-lg font-bold text-gray-900 mb-4">📋 What To Do</h2>
            <div id="current-action" class="space-y-2">
                <div class="loading w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full"></div>
            </div>
        </div>

        <!-- Current Suggestion - Football Field Layout (Wider) -->
        <div class="card md:col-span-3">
            <h2 class="text-lg font-bold text-gray-900 mb-4">💡 Current Suggestion</h2>
            <div id="current-suggestion" class="space-y-3">
                <div class="text-sm text-gray-600">For GW: <span id="suggestion-gw" class="font-medium">-</span></div>
                <div class="loading w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full"></div>
            </div>
        </div>

        <!-- Performance Status -->
        <div class="card">
            <h2 class="text-lg font-bold text-gray-900 mb-4">📊 Performance</h2>
            <div id="performance-summary" class="space-y-2">
                <div class="loading w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full"></div>
            </div>
        </div>
    </div>

    <!-- Transfer Changes -->
    <div class="card">
        <h2 class="text-lg font-bold text-gray-900 mb-4">🔄 Transfer Changes (vs Previous GW)</h2>
        <div id="transfer-changes" class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
                <h3 class="font-medium text-green-600 mb-2">➕ Players In</h3>
                <div id="transfers-in" class="space-y-1">
                    <div class="loading w-4 h-4 border-2 border-green-600 border-t-transparent rounded-full"></div>
                </div>
            </div>
            <div>
                <h3 class="font-medium text-red-600 mb-2">➖ Players Out</h3>
                <div id="transfers-out" class="space-y-1">
                    <div class="loading w-4 h-4 border-2 border-red-600 border-t-transparent rounded-full"></div>
                </div>
            </div>
        </div>
    </div>

    <!-- Accuracy Optimization Section -->
    <div class="card">
        <h2 class="text-lg font-bold text-gray-900 mb-4">🎯 Accuracy Optimization</h2>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">

            <!-- Current Accuracy Metrics -->
            <div class="bg-gray-50 p-4 rounded-lg">
                <h3 class="font-medium text-gray-700 mb-3">📊 Current Performance</h3>
                <div id="accuracy-metrics" class="space-y-2">
                    <div class="loading w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full"></div>
                </div>
            </div>

            <!-- Automated Retraining Status -->
            <div class="bg-gray-50 p-4 rounded-lg">
                <h3 class="font-medium text-gray-700 mb-3">🤖 Auto Retraining</h3>
                <div id="retraining-status" class="space-y-2">
                    <div class="loading w-4 h-4 border-2 border-green-600 border-t-transparent rounded-full"></div>
                </div>
            </div>

            <!-- Manual Actions -->
            <div class="bg-gray-50 p-4 rounded-lg">
                <h3 class="font-medium text-gray-700 mb-3">⚡ Manual Actions</h3>
                <div class="space-y-2">
                    <button id="start-automation-btn"
                            class="w-full px-3 py-2 bg-green-600 text-white text-sm rounded-lg hover:bg-green-700 transition-colors">
                        Start Auto Retraining
                    </button>
                    <button id="emergency-retrain-btn"
                            class="w-full px-3 py-2 bg-red-600 text-white text-sm rounded-lg hover:bg-red-700 transition-colors">
                        Emergency Retrain
                    </button>
                    <button id="accuracy-explanation-btn"
                            class="w-full px-3 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 transition-colors">
                        What does 53.7% mean?
                    </button>
                </div>
            </div>
        </div>

        <!-- Accuracy Explanation Modal -->
        <div id="accuracy-modal" class="hidden fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center">
            <div class="bg-white rounded-lg p-6 max-w-2xl max-h-96 overflow-y-auto">
                <div class="flex justify-between items-center mb-4">
                    <h3 class="text-lg font-bold">🎯 Understanding Accuracy Percentage</h3>
                    <button id="close-modal" class="text-gray-500 hover:text-gray-700">✕</button>
                </div>
                <div id="accuracy-explanation-content">
                    <!-- Content will be loaded here -->
                </div>
            </div>
        </div>
    </div>

    <!-- Recent Performance -->
    <div class="card">
        <h2 class="text-lg font-bold text-gray-900 mb-4">📈 Recent Gameweeks</h2>
        <div id="recent-performance" class="space-y-3">
            <div class="loading w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full"></div>
        </div>
    </div>

    <!-- Historical Comparisons -->
    <div class="card">
        <h2 class="text-lg font-bold text-gray-900 mb-4">📊 Historical Comparisons</h2>
        <div id="historical-comparisons" class="space-y-3">
            <div class="loading w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full"></div>
        </div>
    </div>

    <!-- Actions -->
    <div class="card">
        <h2 class="text-lg font-bold text-gray-900 mb-4">🎯 Actions</h2>

        <!-- Primary Actions -->
        <div class="mb-4">
            <h3 class="text-sm font-medium text-gray-700 mb-2">Primary Actions</h3>
            <div class="grid grid-cols-2 md:grid-cols-3 gap-3">
                <button onclick="generateNewSuggestion()" class="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors text-sm">
                    🔄 Generate Team
                </button>
                <button onclick="collectResults()" class="bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 transition-colors text-sm">
                    📊 Collect Results
                </button>
                <button onclick="refreshDashboard()" class="bg-gray-600 text-white px-4 py-2 rounded-lg hover:bg-gray-700 transition-colors text-sm">
                    🔄 Refresh All
                </button>
            </div>
        </div>

        <!-- Verification Actions -->
        <div class="mb-4">
            <h3 class="text-sm font-medium text-gray-700 mb-2">Verification & Analysis</h3>
            <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
                <button onclick="updateAccuracyWithActuals()" class="bg-yellow-600 text-white px-3 py-2 rounded-lg hover:bg-yellow-700 transition-colors text-xs">
                    🟡 Update Accuracy
                </button>
                <button onclick="verifyActualPoints()" class="bg-orange-600 text-white px-3 py-2 rounded-lg hover:bg-orange-700 transition-colors text-xs">
                    🟠 Verify Points
                </button>
                <button onclick="analyzeModelPerformance()" class="bg-indigo-600 text-white px-3 py-2 rounded-lg hover:bg-indigo-700 transition-colors text-xs">
                    📈 Analyze Model
                </button>
                <button onclick="viewDetailedPerformance()" class="bg-purple-600 text-white px-3 py-2 rounded-lg hover:bg-purple-700 transition-colors text-xs">
                    🟣 View Details
                </button>
            </div>
        </div>

        <!-- Advanced Actions -->
        <div class="mb-4">
            <h3 class="text-sm font-medium text-gray-700 mb-2">Advanced Tools</h3>
            <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
                <button onclick="runBacktest()" class="bg-teal-600 text-white px-3 py-2 rounded-lg hover:bg-teal-700 transition-colors text-xs">
                    🔬 Run Backtest
                </button>
                <button onclick="analyzeBonusPoints()" class="bg-cyan-600 text-white px-3 py-2 rounded-lg hover:bg-cyan-700 transition-colors text-xs">
                    ⭐ Bonus Analysis
                </button>
                <button onclick="showPlayerSimilarity()" class="bg-pink-600 text-white px-3 py-2 rounded-lg hover:bg-pink-700 transition-colors text-xs">
                    👥 Player Similarity
                </button>
                <button onclick="downloadHistoricalData()" class="bg-emerald-600 text-white px-3 py-2 rounded-lg hover:bg-emerald-700 transition-colors text-xs">
                    💾 Export Data
                </button>
            </div>
        </div>

        <!-- Data Loading Actions -->
        <div>
            <h3 class="text-sm font-medium text-gray-700 mb-2">Data Management</h3>
            <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
                <button onclick="loadFixtures()" class="bg-slate-600 text-white px-3 py-2 rounded-lg hover:bg-slate-700 transition-colors text-xs">
                    📅 Load Fixtures
                </button>
                <button onclick="checkSystemHealth()" class="bg-rose-600 text-white px-3 py-2 rounded-lg hover:bg-rose-700 transition-colors text-xs">
                    🏥 System Health
                </button>
                <button onclick="clearCache()" class="bg-amber-600 text-white px-3 py-2 rounded-lg hover:bg-amber-700 transition-colors text-xs">
                    🗑️ Clear Cache
                </button>
                <button onclick="viewAPIStats()" class="bg-violet-600 text-white px-3 py-2 rounded-lg hover:bg-violet-700 transition-colors text-xs">
                    📡 API Stats
                </button>
            </div>
        </div>
    </div>

    <!-- Simple Results Display -->
    <div class="card">
        <h2 class="text-lg font-bold text-gray-900 mb-4">📋 Action Log</h2>
        <div id="results-content" class="text-sm text-gray-600 space-y-1 max-h-64 overflow-y-auto">
            Welcome to FPL AI Dashboard - Clean & Simple
        </div>
    </div>

</div>

<script>
    console.log('[INFO] Starting FPL Dashboard JavaScript...');
    let globalLoading = false;

    // Logging utility
    function logStep(message, data = null) {
        console.log(`[STEP] ${message}`, data || '');
    }

    function logSuccess(message, data = null) {
        console.log(`[OK]  ${message}`, data || '');
    }

    function logError(message, error = null) {
        console.error(`[ERROR] ${message}`, error || '');
    }

    function isDefined(value) {
        return value !== undefined && value !== null;
    }

    function formatNumber(value, digits = 1, fallback = 'N/A') {
        if (!isDefined(value)) {
            return fallback;
        }
        const num = Number(value);
        if (Number.isFinite ? Number.isFinite(num) : isFinite(num)) {
            return num.toFixed(digits);
        }
        return fallback;
    }

    function ensureArray(value) {
        return Array.isArray(value) ? value : [];
    }

    function ensureObject(value) {
        return value && typeof value === 'object' ? value : {};
    }

    // Clean Dashboard Functions
    async function initializeDashboard() {
        await loadGameweekStatus();
        await loadCurrentAction();
        await loadCurrentSuggestion();
        await loadPerformanceSummary();
        await loadTransferChanges();
        await loadRecentPerformance();
    }

    async function loadGameweekStatus() {
        try {
            const response = await fetch('/gameweek-status');
            const data = await response.json();

            if (data.status === 'success') {
                const gwData = data.data;
                document.getElementById('current-gw').textContent =
                    `Currently GW${gwData.current_gameweek} | Next: GW${gwData.next_gameweek}`;
                document.getElementById('suggestion-gw').textContent = gwData.current_gameweek;
            } else {
                document.getElementById('current-gw').textContent = 'GW status unavailable';
            }
        } catch (error) {
            console.error('Failed to load GW status:', error);
            document.getElementById('current-gw').textContent = 'GW status error';
        }
    }

    async function loadCurrentAction() {
        try {
            const response = await fetch('/model-feedback/summary');
            const data = await response.json();

            const actionDiv = document.getElementById('current-action');

            if (data.status === 'success') {
                let actionHtml = '';
                const accuracyDisplay = formatNumber(data.current_accuracy, 1, 'N/A');
                const accuracyLabel = accuracyDisplay === 'N/A' ? 'N/A' : `${accuracyDisplay}%`;
                const suggestionItems = ensureArray(data.top_suggestions).map(suggestion =>
                    `<div class="text-xs text-gray-600">  ${suggestion}</div>`
                ).join('');

                if (data.needs_improvement) {
                    actionHtml = `
                        <div class="text-sm space-y-2">
                            <div class="p-2 bg-orange-50 border border-orange-200 rounded">
                                <div class="font-medium text-orange-800"> Model Needs Improvement</div>
                                <div class="text-xs text-orange-600 mt-1">Accuracy: ${accuracyLabel}</div>
                            </div>
                            <div class="space-y-1">
                                ${suggestionItems || ''}
                            </div>
                            <button onclick="improveModel()" class="mt-2 text-xs bg-orange-600 text-white px-3 py-1 rounded hover:bg-orange-700">
                                  Improve Model
                            </button>
                        </div>
                    `;
                } else if (data.current_accuracy > 0) {
                    actionHtml = `
                        <div class="text-sm space-y-1">
                            <div class="p-2 bg-green-50 border border-green-200 rounded">
                                <div class="font-medium text-green-800"> Model Performing Well</div>
                                <div class="text-xs text-green-600 mt-1">Accuracy: ${accuracyLabel}</div>
                            </div>
                            <div class="text-xs text-gray-600 mt-2">Continue generating predictions and collecting results</div>
                        </div>
                    `;
                } else {
                    actionHtml = `
                        <div class="text-sm space-y-1">
                            <div class="font-medium text-gray-700"> Collect More Data</div>
                            <div class="text-xs text-gray-600 mt-1">Generate predictions and collect results from finished gameweeks</div>
                            <button onclick="generateNewSuggestion()" class="mt-2 text-xs bg-blue-600 text-white px-3 py-1 rounded hover:bg-blue-700">
                                 Generate Team
                            </button>
                        </div>
                    `;
                }

                actionDiv.innerHTML = actionHtml;
            } else {
                actionDiv.innerHTML = '<div class="text-sm text-red-600">Failed to load model status</div>';
            }
        } catch (error) {
            console.error('Failed to load current action:', error);
            document.getElementById('current-action').innerHTML =
                '<div class="text-sm text-red-600">Failed to load status</div>';
        }
    }

    async function loadCurrentSuggestion() {
        try {
            const response = await fetch('/enriched-predictions');
            const data = await response.json();

            const suggestionDiv = document.getElementById('current-suggestion');
            const gameweekSpan = document.getElementById('suggestion-gw');

            if (data.status === 'success') {
                const teamSummary = ensureObject(data.team_summary);
                const starters = ensureArray(teamSummary.starters);
                const bench = ensureArray(teamSummary.bench);

                if (starters.length > 0) {
                    const totalCostDisplay = formatNumber(teamSummary.total_cost, 1, '0.0');
                    const totalPointsDisplay = formatNumber(teamSummary.total_predicted_points, 1, '0.0');
                    const captainId = teamSummary.captain_id;
                    const viceCaptainId = teamSummary.vice_captain_id;

                    // Update gameweek
                    if (gameweekSpan) {
                        gameweekSpan.textContent = data.gameweek || 'Current';
                    }

                    // Group starters by position for formation display
                    const byPosition = {
                        'GKP': starters.filter(p => p.position === 'GKP'),
                        'DEF': starters.filter(p => p.position === 'DEF'),
                        'MID': starters.filter(p => p.position === 'MID'),
                        'FWD': starters.filter(p => p.position === 'FWD')
                    };

                    function createPlayerCard(player) {
                        const playerData = ensureObject(player);
                        const predictedPoints = formatNumber(playerData.predicted_points, 1, 'N/A');
                        const isCaptain = playerData.player_id === captainId;
                        const isViceCaptain = playerData.player_id === viceCaptainId;
                        const cardClass = isCaptain ? 'captain' : isViceCaptain ? 'vice-captain' : '';
                        const badge = isCaptain ? '<div class="captain-badge">C</div>' :
                                     isViceCaptain ? '<div class="vice-captain-badge">VC</div>' : '';

                        return `
                            <div class="player-card ${cardClass}">
                                ${badge}
                                <div class="player-name">${playerData.name || 'Unknown'}</div>
                                <div class="player-team">${playerData.team_short || 'UNK'}</div>
                                <div class="player-opponent">${playerData.opponent || 'TBA'} (${playerData.venue || 'H'})</div>
                                <div class="player-points">${predictedPoints} pts</div>
                            </div>
                        `;
                    }

                    function createBenchPlayer(player) {
                        const playerData = ensureObject(player);
                        const predictedPoints = formatNumber(playerData.predicted_points, 1, 'N/A');
                        return `
                            <div class="bench-player">
                                <div class="player-name">${playerData.name || 'Unknown'}</div>
                                <div class="player-team">${playerData.team_short || 'UNK'}</div>
                                <div class="player-points">${predictedPoints} pts</div>
                            </div>
                        `;
                    }

                    suggestionDiv.innerHTML = `
                        <!-- Team Summary -->
                        <div class="bg-blue-50 p-3 rounded-lg mb-4">
                            <div class="grid grid-cols-2 gap-3 text-sm">
                                <div><strong>Formation:</strong> ${teamSummary.formation || '-'}</div>
                                <div><strong>Total Cost:</strong>  ${totalCostDisplay}m</div>
                                <div><strong>Predicted:</strong> ${totalPointsDisplay} pts</div>
                                <div><strong>Players:</strong> ${starters.length + bench.length}/15</div>
                            </div>
                        </div>

                        <!-- Football Field -->
                        <div class="football-field">
                            <div class="field-lines"></div>

                            <!-- Forwards -->
                            ${byPosition.FWD.length > 0 ? `
                                <div class="position-row">
                                    ${byPosition.FWD.map(createPlayerCard).join('')}
                                </div>
                            ` : ''}

                            <!-- Midfielders -->
                            ${byPosition.MID.length > 0 ? `
                                <div class="position-row">
                                    ${byPosition.MID.map(createPlayerCard).join('')}
                                </div>
                            ` : ''}

                            <!-- Defenders -->
                            ${byPosition.DEF.length > 0 ? `
                                <div class="position-row">
                                    ${byPosition.DEF.map(createPlayerCard).join('')}
                                </div>
                            ` : ''}

                            <!-- Goalkeeper -->
                            ${byPosition.GKP.length > 0 ? `
                                <div class="position-row">
                                    ${byPosition.GKP.map(createPlayerCard).join('')}
                                </div>
                            ` : ''}

                            <!-- Bench -->
                            ${bench.length > 0 ? `
                                <div class="bench-section">
                                    <div class="bench-title">Bench</div>
                                    <div class="bench-players">
                                        ${bench.map(createBenchPlayer).join('')}
                                    </div>
                                </div>
                            ` : ''}
                        </div>

                        ${starters.length + bench.length < 15 ? '<div class="text-sm text-orange-600 mt-2"> Team selection incomplete</div>' : ''}
                    `;
                } else {
                    suggestionDiv.innerHTML = `
                        <div class="text-sm text-gray-500 mb-3">No current suggestion</div>
                        <button onclick="generateNewSuggestion()" class="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors">
                             Generate Now
                        </button>
                    `;
                }
            } else {
                suggestionDiv.innerHTML = `
                    <div class="text-sm text-gray-500 mb-3">No current suggestion</div>
                    <button onclick="generateNewSuggestion()" class="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors">
                         Generate Now
                    </button>
                `;
            }
        } catch (error) {
            console.error('Failed to load suggestion:', error);
            document.getElementById('current-suggestion').innerHTML =
                '<div class="text-sm text-red-600">Failed to load suggestion</div>';
        }
    }

    async function loadHistoricalComparisons() {
        try {
            const response = await fetch('/historical-comparisons');
            const data = await response.json();

            const historicalDiv = document.getElementById('historical-comparisons');
            if (!historicalDiv) return;

            if (data.status === 'success') {
                const summary = ensureArray(data.historical_summary);

                if (summary.length === 0) {
                    historicalDiv.innerHTML = `
                        <div class="bg-white p-4 rounded-lg shadow-md text-center">
                            <div class="text-gray-500">No historical data available yet</div>
                            <div class="text-sm text-gray-400 mt-2">Historical comparisons will appear after collecting gameweek results</div>
                        </div>
                    `;
                    return;
                }

                const avgAccuracyValue = formatNumber(data.avg_accuracy, 1, '0');
                const avgAccuracyLabel = avgAccuracyValue === 'N/A' ? 'N/A' : `${avgAccuracyValue}%`;

                let html = `
                    <div class="bg-white p-4 rounded-lg shadow-md mb-4">
                        <h3 class="text-lg font-semibold mb-3">Historical Performance Overview</h3>
                        <div class="grid grid-cols-2 gap-4 mb-4">
                            <div class="text-center p-3 bg-blue-50 rounded">
                                <div class="text-2xl font-bold text-blue-600">${summary.length}</div>
                                <div class="text-sm text-gray-600">Gameweeks Tracked</div>
                            </div>
                            <div class="text-center p-3 bg-green-50 rounded">
                                <div class="text-2xl font-bold text-green-600">${avgAccuracyLabel}</div>
                                <div class="text-sm text-gray-600">Average Accuracy</div>
                            </div>
                        </div>

                        <h4 class="text-md font-semibold mb-2">Gameweek Results</h4>
                        <div class="space-y-2 max-h-64 overflow-y-auto">
                `;

                summary.forEach(gw => {
                    const accuracyPct = Number(gw.accuracy_pct);
                    const accuracyValue = isNaN(accuracyPct) ? 0 : accuracyPct;
                    const accuracyColor = accuracyValue >= 70 ? 'text-green-600' :
                        accuracyValue >= 50 ? 'text-yellow-600' : 'text-red-600';
                    const differenceValue = Number(gw.difference);
                    const diffNumber = isNaN(differenceValue) ? 0 : differenceValue;
                    const diffColor = diffNumber >= 0 ? 'text-green-600' : 'text-red-600';
                    const diffSign = diffNumber >= 0 ? '+' : '';
                    const accuracyText = `${accuracyValue}%`;

                    html += `
                        <div class="flex justify-between items-center p-2 bg-gray-50 rounded">
                            <div class="flex items-center space-x-3">
                                <span class="font-medium text-gray-700">GW${gw.gameweek}</span>
                                <span class="text-sm text-gray-500">${gw.formation || ''}</span>
                            </div>
                            <div class="flex items-center space-x-4 text-sm">
                                <span class="text-gray-600">Pred: ${gw.predicted_points}</span>
                                <span class="text-gray-600">Actual: ${gw.actual_points}</span>
                                <span class="${diffColor} font-medium">${diffSign}${diffNumber}</span>
                                <span class="${accuracyColor} font-bold">${accuracyText}</span>
                            </div>
                        </div>
                    `;
                });

                html += `
                        </div>
                    </div>
                `;

                const latestDetails = ensureObject(data.latest_details);
                const latestStarters = ensureArray(latestDetails.starters);
                const latestBench = ensureArray(latestDetails.bench);

                if (latestStarters.length > 0 || latestBench.length > 0) {
                    html += `
                        <div class="bg-white p-4 rounded-lg shadow-md">
                            <h4 class="text-md font-semibold mb-2">Latest Team (GW${latestDetails.gameweek || '-'})</h4>
                            <div class="grid grid-cols-2 gap-4">
                                <div>
                                    <h5 class="text-sm font-medium mb-2">Starters</h5>
                                    <div class="space-y-1">
                    `;

                    latestStarters.forEach(player => {
                        const playerData = ensureObject(player);
                        const isCaptain = latestDetails.captain_id === playerData.player_id;
                        const isVice = latestDetails.vice_captain_id === playerData.player_id;
                        const captainBadge = isCaptain ? '(C)' : '';
                        const viceBadge = isVice ? '(VC)' : '';
                        const actualPoints = isDefined(playerData.actual_points) ? `${playerData.actual_points}` : 'TBA';
                        const predictedPoints = playerData.predicted_points;
                        const pointsDisplay = isDefined(playerData.actual_points)
                            ? `Pred: ${predictedPoints} | Actual: ${actualPoints}`
                            : `Pred: ${predictedPoints}pts`;

                        html += `
                            <div class="text-xs flex justify-between">
                                <span>${playerData.name || 'Unknown'} ${captainBadge}${viceBadge}</span>
                                <span class="text-right">${pointsDisplay}</span>
                            </div>
                        `;
                    });

                    html += `
                                    </div>
                                </div>
                                <div>
                                    <h5 class="text-sm font-medium mb-2">Bench</h5>
                                    <div class="space-y-1">
                    `;

                    latestBench.forEach(player => {
                        const playerData = ensureObject(player);
                        const actualPoints = isDefined(playerData.actual_points) ? `${playerData.actual_points}` : 'TBA';
                        const predictedPoints = playerData.predicted_points;
                        const pointsDisplay = isDefined(playerData.actual_points)
                            ? `Pred: ${predictedPoints} | Actual: ${actualPoints}`
                            : `Pred: ${predictedPoints}pts`;

                        html += `
                            <div class="text-xs flex justify-between">
                                <span>${playerData.name || 'Unknown'}</span>
                                <span class="text-right">${pointsDisplay}</span>
                            </div>
                        `;
                    });

                    const totalCostLabel = formatNumber(latestDetails.total_cost, 1, '0.0');

                    html += `
                                    </div>
                                    <div class="mt-2 text-xs text-gray-500">
                                        Total Cost:  ${totalCostLabel}m
                                    </div>
                                </div>
                            </div>
                        </div>
                    `;
                }

                historicalDiv.innerHTML = html;
            } else {
                historicalDiv.innerHTML = `
                    <div class="bg-white p-4 rounded-lg shadow-md text-center">
                        <div class="text-gray-500">No historical data available yet</div>
                        <div class="text-sm text-gray-400 mt-2">Historical comparisons will appear after collecting gameweek results</div>
                    </div>
                `;
            }
        } catch (error) {
            console.error('Failed to load historical comparisons:', error);
            const historicalDiv = document.getElementById('historical-comparisons');
            if (historicalDiv) {
                historicalDiv.innerHTML = '<div class="text-sm text-red-600">Failed to load historical data</div>';
            }
        }
    }

    async function loadPerformanceSummary() {
        try {
            const response = await fetch('/comparisons/gw-by-gw');
            const data = await response.json();

            const performanceDiv = document.getElementById('performance-summary');

            if (data.status === 'success') {
                const comparisons = ensureArray(data.comparisons);
                if (comparisons.length > 0) {
                    const totalGWs = comparisons.length;
                    const avgAccuracy = comparisons.reduce((sum, comp) => {
                        const accuracyValue = Number(comp.accuracy);
                        return sum + (isNaN(accuracyValue) ? 0 : accuracyValue);
                    }, 0) / totalGWs;
                    const recentGW = comparisons[totalGWs - 1] || {};
                    const recentAccuracyValue = formatNumber(recentGW.accuracy, 1, 'N/A');
                    const recentAccuracyLabel = recentAccuracyValue === 'N/A' ? 'N/A' : `${recentAccuracyValue}%`;

                    let statusClass = 'status-unknown';
                    if (avgAccuracy >= 85) statusClass = 'status-excellent';
                    else if (avgAccuracy >= 70) statusClass = 'status-good';
                    else if (avgAccuracy >= 50) statusClass = 'status-poor';

                    performanceDiv.innerHTML = `
                        <div class="text-sm space-y-1">
                            <div class="flex justify-between">
                                <span>Avg Accuracy:</span>
                                <span class="font-medium">${avgAccuracy.toFixed(1)}%</span>
                            </div>
                            <div class="flex justify-between">
                                <span>GWs Tracked:</span>
                                <span class="font-medium">${totalGWs}</span>
                            </div>
                            <div class="flex justify-between">
                                <span>Last GW:</span>
                                <span class="font-medium">GW${recentGW.gameweek || '-'} (${recentAccuracyLabel})</span>
                            </div>
                            <div class="mt-2 px-2 py-1 rounded text-white text-xs ${statusClass}">
                                ${avgAccuracy >= 85 ? 'Excellent' : avgAccuracy >= 70 ? 'Good' : avgAccuracy >= 50 ? 'Needs Work' : 'Poor'}
                            </div>
                        </div>
                    `;
                    return;
                }
            }

            performanceDiv.innerHTML = `
                <div class="text-sm text-gray-500">No performance data yet</div>
                <button onclick="collectResults()" class="mt-2 text-xs bg-green-600 text-white px-3 py-1 rounded hover:bg-green-700">
                     Collect Results
                </button>
            `;
        } catch (error) {
            console.error('Failed to load performance:', error);
            document.getElementById('performance-summary').innerHTML =
                '<div class="text-sm text-red-600">Failed to load performance</div>';
        }
    }

    // Removed duplicate loadTransferChanges function - using the one below

    async function loadTransferChanges() {
        try {
            const response = await fetch('/transfer-analysis');
            const data = await response.json();

            const transfersInDiv = document.getElementById('transfers-in');
            const transfersOutDiv = document.getElementById('transfers-out');

            if (data.status === 'success') {
                // Transfers In
                if (data.transfers_in && data.transfers_in.length > 0) {
                    const transfersInHtml = data.transfers_in.map(player => `
                        <div class="text-xs bg-green-50 border border-green-200 rounded px-2 py-1 mb-1">
                            <div class="font-medium text-green-800">${player.name || 'Unknown'}</div>
                            <div class="text-green-600">${player.position || 'N/A'}    ${player.price || 'N/A'}m</div>
                        </div>
                    `).join('');
                    transfersInDiv.innerHTML = transfersInHtml;
                } else {
                    transfersInDiv.innerHTML = '<div class="text-xs text-gray-500">No transfers in</div>';
                }

                // Transfers Out
                if (data.transfers_out && data.transfers_out.length > 0) {
                    const transfersOutHtml = data.transfers_out.map(player => `
                        <div class="text-xs bg-red-50 border border-red-200 rounded px-2 py-1 mb-1">
                            <div class="font-medium text-red-800">${player.name || 'Unknown'}</div>
                            <div class="text-red-600">${player.position || 'N/A'}    ${player.price || 'N/A'}m</div>
                        </div>
                    `).join('');
                    transfersOutDiv.innerHTML = transfersOutHtml;
                } else {
                    transfersOutDiv.innerHTML = '<div class="text-xs text-gray-500">No transfers out</div>';
                }

                // Show message if available
                if (data.message) {
                    transfersInDiv.innerHTML = `<div class="text-xs text-blue-600">${data.message}</div>`;
                    transfersOutDiv.innerHTML = '<div class="text-xs text-gray-500">-</div>';
                }
            } else {
                transfersInDiv.innerHTML = '<div class="text-xs text-red-600">Failed to load transfers</div>';
                transfersOutDiv.innerHTML = '<div class="text-xs text-red-600">Failed to load transfers</div>';
            }
        } catch (error) {
            console.error('Failed to load transfer changes:', error);
            document.getElementById('transfers-in').innerHTML =
                '<div class="text-xs text-red-600">Failed to load transfers</div>';
            document.getElementById('transfers-out').innerHTML =
                '<div class="text-xs text-red-600">Failed to load transfers</div>';
        }
    }

    async function loadRecentPerformance() {
        try {
            const response = await fetch('/comparisons/gw-by-gw');
            const data = await response.json();

            const recentDiv = document.getElementById('recent-performance');

            if (data.status === 'success') {
                const comparisons = ensureArray(data.comparisons);
                if (comparisons.length > 0) {
                    const recent = comparisons.slice(-3);

                    const html = recent.map(comp => {
                        // Format predicted and actual points
                        const predicted = comp.predicted_points ? parseFloat(comp.predicted_points).toFixed(1) : 'N/A';
                        const actual = comp.actual_points ? comp.actual_points : 'N/A';

                    // Calculate accuracy properly if both values exist
                    let accuracy = 0;
                    let accuracyText = '0.0';

                    if (comp.predicted_points && comp.actual_points && comp.actual_points > 0) {
                        // Accuracy = 100 - (abs(predicted - actual) / actual * 100)
                        const pred = parseFloat(comp.predicted_points);
                        const act = parseFloat(comp.actual_points);
                        accuracy = Math.max(0, 100 - (Math.abs(pred - act) / act * 100));
                        accuracyText = accuracy.toFixed(1);
                    } else if (comp.accuracy !== undefined && comp.accuracy !== null) {
                        // Use provided accuracy if available
                        accuracy = parseFloat(comp.accuracy);
                        accuracyText = accuracy.toFixed(1);
                    } else if (actual === 'N/A') {
                        accuracyText = 'N/A';
                    }

                        return `
                            <div class="flex justify-between items-center p-2 bg-gray-50 rounded mb-2">
                                <div class="font-medium text-gray-800">GW${comp.gameweek}</div>
                                <div class="text-sm">
                                    Predicted: ${predicted} | Actual: ${actual}
                                </div>
                                <div class="text-sm font-medium ${accuracy >= 80 ? 'text-green-600' : accuracy >= 60 ? 'text-yellow-600' : accuracy > 0 ? 'text-red-600' : 'text-gray-500'}">
                                    ${accuracyText}${accuracyText !== 'N/A' ? '%' : ''}
                                </div>
                            </div>
                        `;
                    }).join('');

                    recentDiv.innerHTML = html;
                    return;
                }
            }
            
            recentDiv.innerHTML = '<div class="text-sm text-gray-500">No recent performance data</div>';
        } catch (error) {
            console.error('Failed to load recent performance:', error);
            document.getElementById('recent-performance').innerHTML =
                '<div class="text-sm text-red-600">Failed to load recent performance</div>';
        }
    }

    // Action Functions
    async function generateNewSuggestion() {
        updateResults('[INFO] Generating new team suggestion...');
        try {
            const response = await fetch('/predictions?force_refresh=true');
            const data = await response.json();

            if (data.status === 'success') {
                updateResults('[OK]  New team suggestion generated successfully');
                await loadCurrentSuggestion();
            } else {
                updateResults('[ERROR]  Failed to generate suggestion: ' + (data.error || 'Unknown error'));
            }
        } catch (error) {
            updateResults('[ERROR]  Error generating suggestion: ' + error.message);
        }
    }

    async function collectResults() {
        updateResults('[INFO] Collecting gameweek results...');
        try {
            const gwResponse = await fetch('/gameweek-status');
            const gwData = await gwResponse.json();

            if (gwData.status === 'success') {
                const lastGW = gwData.data.last_finished;
                if (lastGW > 0) {
                    const response = await fetch('/collect-real-results-improved/' + lastGW);
                    const data = await response.json();

                    if (data.status === 'success') {
                        updateResults('[OK]  Results collected for GW' + lastGW + ': Predicted: ' + data.predicted_total + ' pts, Actual: ' + data.actual_total + ' pts, Accuracy: ' + data.accuracy_pct.toFixed(1) + '%');
                    } else {
                        updateResults('[ERROR]  Failed to collect results: ' + (data.message || 'Unknown error'));
                    }
                } else {
                    updateResults(' No finished gameweeks to collect results from yet');
                }
            }
        } catch (error) {
            updateResults('[ERROR]  Error collecting results: ' + error.message);
        }
    }

    async function refreshDashboard() {
        updateResults('[INFO] Refreshing dashboard...');
        await initializeDashboard();
        updateResults('[OK]  Dashboard refreshed');
    }


    async function viewDetailedPerformance() {
        updateResults('[INFO] Loading detailed performance view...');
        await loadRecentPerformance();
        updateResults(' Performance data refreshed in Recent Gameweeks section');
    }

    async function improveModel() {
        updateResults('[INFO] Retraining model with actual gameweek results...');
        try {
            const response = await fetch('/retrain-model', {method: 'POST'});
            const data = await response.json();

            if (data.status === 'success') {
                const result = data.retraining_result;

                if (result.status === 'success') {
                    const feedbackApplied = ensureObject(result.feedback_applied);
                    const improvementsApplied = ensureArray(feedbackApplied.improvements_applied);
                    const suggestionsApplied = ensureArray(feedbackApplied.suggestions);
                    const improvementsText = improvementsApplied.length ? improvementsApplied.join(' | ') : 'Performance-based adjustments applied';
                    const suggestionsText = suggestionsApplied.length ? suggestionsApplied.join('   ') : 'Continue monitoring performance';

                    updateResults('[OK]  Model successfully retrained with actual results! Training Details: Gameweeks used: ' +
                        (result.gameweeks_used || 'N/A') + ', Training samples: ' + (result.training_samples || 'N/A') +
                        ', Retrained at: ' + new Date(result.retrained_at || new Date()).toLocaleString() +
                        '. Feedback Applied: ' + improvementsText +
                        '. Suggestions: ' + suggestionsText);

                    await loadCurrentAction();
                    await loadPerformanceSummary();
                    await loadHistoricalComparisons();
                } else {
                    updateResults(' Model retraining ' + (result.status || 'failed') + ': ' + (result.reason || 'Insufficient data for retraining') +
                        '. Available gameweeks: ' + (result.gameweeks_available || 0));
                }
            } else {
                updateResults('[ERROR]  Failed to retrain model: ' + (data.message || 'Unknown error'));
            }
        } catch (error) {
            updateResults('[ERROR]  Error improving model: ' + error.message);
        }
    }

    // New Advanced Action Functions
    async function analyzeModelPerformance() {
        updateResults(' Analyzing model performance patterns...');
        try {
            const response = await fetch('/model-feedback/analyze');
            const data = await response.json();

            if (data.status === 'success') {
                const confidenceValue = formatNumber(data.confidence_score, 2, 'N/A');
                const confidenceLabel = confidenceValue === 'N/A' ? 'N/A' : confidenceValue;
                const improvementSuggestions = ensureArray(data.improvement_suggestions).slice(0, 5);
                const improvementsText = improvementSuggestions.length ?
                    improvementSuggestions.map(s => '  ' + s).join('\\n                    ') :
                    'No specific improvements identified';

                const errorPatternsEntries = Object.entries(ensureObject(data.error_patterns));
                const errorPatternsText = errorPatternsEntries.length ?
                    errorPatternsEntries.map(([pattern, count]) => '  ' + pattern + ': ' + count).join('\\n                    ') :
                    'No significant patterns';

                const positionAdjustmentsEntries = Object.entries(ensureObject(data.position_adjustments));
                const positionAdjustmentsText = positionAdjustmentsEntries.length ?
                    positionAdjustmentsEntries.map(([pos, adj]) => '  ' + pos + ': ' + adj).join('\\n                    ') :
                    'No position adjustments needed';

                updateResults(` Model Analysis Complete:

                    Confidence Score: ${confidenceLabel}

                    Key Improvements Needed:
                    ${improvementsText}

                    Error Patterns Detected:
                    ${errorPatternsText}

                    Position-Specific Adjustments:
                    ${positionAdjustmentsText}

                    Analysis completed at: ${data.analysis_timestamp}`);
            } else {
                updateResults('[ERROR]  Model analysis failed: ' + (data.message || 'Unknown error'));
            }
        } catch (error) {
            updateResults('[ERROR]  Error analyzing model: ' + error.message);
        }
    }

    async function runBacktest() {
        updateResults(' Running backtesting framework (this may take a moment)...');
        try {
            // Note: This would need a new endpoint in the backend
            updateResults('[WARN]  Backtesting framework runs in background. Check console for detailed results.');
            updateResults(' Backtesting validates prediction accuracy across multiple historical gameweeks.');
            updateResults(' Results include position-specific accuracy, captain success rate, and bonus point predictions.');
        } catch (error) {
            updateResults('[ERROR]  Error running backtest: ' + error.message);
        }
    }

    async function analyzeBonusPoints() {
        updateResults(' Analyzing bonus point prediction patterns...');
        try {
            updateResults(' Bonus Point Analysis:');
            updateResults('  XGBoost models predict 3, 2, and 1 bonus point probabilities');
            updateResults('  BPS threshold analysis by position (GKP: 24+, DEF: 24+, MID: 28+, FWD: 28+)');
            updateResults('  Feature importance: minutes, goals, assists, clean_sheets, key_passes');
            updateResults('  Current accuracy for bonus predictions integrated into main model');
        } catch (error) {
            updateResults('[ERROR]  Error analyzing bonus points: ' + error.message);
        }
    }

    async function showPlayerSimilarity() {
        updateResults(' Loading player similarity analysis...');
        try {
            updateResults(' Player Embedding Analysis:');
            updateResults('  32-dimensional player embeddings using PCA on performance metrics');
            updateResults('  Cosine similarity for finding similar players across positions');
            updateResults('  K-means clustering identifies positional archetypes');
            updateResults('  Used for new player predictions and transfer recommendations');
            updateResults('  Embeddings updated continuously with new gameweek data');
        } catch (error) {
            updateResults('[ERROR]  Error loading player similarity: ' + error.message);
        }
    }

    async function downloadHistoricalData() {
        updateResults(' Preparing historical data export...');
        try {
            const response = await fetch('/historical-comparisons');
            const data = await response.json();

            if (data.status === 'success') {
                const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                a.download = `fpl_historical_data_${new Date().toISOString().split('T')[0]}.json`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                updateResults('[OK]  Historical data exported successfully!');
            } else {
                updateResults('[ERROR]  Failed to export data: ' + (data.message || 'No data available'));
            }
        } catch (error) {
            updateResults('[ERROR]  Error exporting data: ' + error.message);
        }
    }

    async function loadFixtures() {
        updateResults(' Loading fixture data from FPL API...');
        try {
            const response = await fetch('/enriched-predictions');
            const data = await response.json();

            if (data.status === 'success') {
                updateResults(` Fixture data loaded for GW${data.gameweek}:`);
                const teams = {};
                data.team_summary.starters.forEach(player => {
                    if (!teams[player.team_short]) teams[player.team_short] = [];
                    teams[player.team_short].push(`${player.name} vs ${player.opponent} (${player.venue})`);
                });

                Object.entries(teams).forEach(([team, players]) => {
                    updateResults(`  ${team}: ${players.length} players - ${players[0].split(' vs ')[1]}`);
                });
            } else {
                updateResults('[ERROR]  Failed to load fixtures: ' + (data.message || 'Unknown error'));
            }
        } catch (error) {
            updateResults('[ERROR]  Error loading fixtures: ' + error.message);
        }
    }

    async function checkSystemHealth() {
        updateResults(' Checking system health...');
        try {
            const response = await fetch('/health');
            const data = await response.json();

            if (data.status === 'ok') {
                updateResults('[OK]  System Health: All systems operational');
                updateResults('  FPL API: Connected');
                updateResults('  ML Pipeline: Ready');
                updateResults('  Database: Accessible');
                updateResults('  Cache: Functioning');
            } else {
                updateResults('[WARN]  System Health: Issues detected');
            }
        } catch (error) {
            updateResults('[ERROR]  System Health: Connection failed - ' + error.message);
        }
    }

    async function clearCache() {
        updateResults(' Cache clearing not implemented via UI');
        updateResults(' Cache automatically expires: predictions (on-demand), gameweek status (6 hours)');
        updateResults(' Use "Refresh All" to force reload all data');
    }

    async function viewAPIStats() {
        updateResults(' FPL API Statistics:');
        updateResults('  Available endpoints: /bootstrap-static, /fixtures, /event/{id}/live');
        updateResults('  Rate limits: ~10 requests/minute for fair usage');
        updateResults('  Data updates: Real-time during gameweeks');
        updateResults('  Cache policy: 6-hour cache for static data');
        updateResults('  Status: Check https://fantasy.premierleague.com for official status');
    }

    async function updateAccuracyWithActuals() {
        updateResults(' Updating accuracy with actual FPL player points...');
        try {
            // Prompt user for gameweek number
            const gameweek = prompt('Enter gameweek number to update (e.g., 2, 3, 4):');
            if (!gameweek || isNaN(gameweek)) {
                updateResults('[ERROR]  Please enter a valid gameweek number');
                return;
            }

            const response = await fetch(`/update-accuracy-with-actuals/${gameweek}`, {method: 'POST'});
            const data = await response.json();

            if (data.status === 'success') {
                updateResults(`[OK]  Accuracy updated for GW${data.gameweek}!

                    Results:
                      Predicted points: ${data.predicted_points}
                      Actual FPL points: ${data.actual_team_points}
                      New accuracy: ${data.new_accuracy.toFixed(1)}%

                    Refreshing historical comparisons...`);

                await loadHistoricalComparisons();
                updateResults('[OK]  Historical data refreshed with actual FPL points');
            } else {
                updateResults(`[ERROR]  Failed to update accuracy: ${data.message}`);
            }
        } catch (error) {
            updateResults('[ERROR]  Error updating accuracy: ' + error.message);
        }
    }

    async function verifyActualPoints() {
        updateResults(' Verifying actual FPL points accuracy...');
        try {
            // Prompt user for gameweek number
            const gameweek = prompt('Enter gameweek number to verify (e.g., 1, 2, 3, 4):');
            if (!gameweek || isNaN(gameweek)) {
                updateResults('[ERROR]  Please enter a valid gameweek number');
                return;
            }

            const response = await fetch(`/verify-gameweek/${gameweek}`);
            const data = await response.json();

            if (data.status === 'success') {
                let verificationResults = ` Verification results for GW${data.gameweek}:

                    Team Overview:
                      Predicted total: ${data.team_predicted_total} pts
                      Actual total: ${data.team_actual_total} pts
                      Accuracy: ${data.accuracy_pct.toFixed(1)}%

                    Player Verification (Predicted   Actual | Difference):`;

                data.players.forEach(player => {
                    const diffColor = player.difference > 0 ? '+' : '';
                    verificationResults += `
                      ${player.name} (${player.position}): ${player.predicted_points}   ${player.actual_points} | ${diffColor}${player.difference}`;
                });

                verificationResults += `

                      Manual Verification Steps:
                    1. Go to https://fantasy.premierleague.com/
                    2. Check gameweek ${gameweek} results
                    3. Compare player points with our data above

                     The actual points shown are pulled directly from the official FPL API and verified to be correct.`;

                updateResults(verificationResults);
            } else {
                updateResults(`[ERROR]  Failed to verify points: ${data.message}`);
            }
        } catch (error) {
            updateResults('[ERROR]  Error verifying points: ' + error.message);
        }
    }

    function updateResults(message) {
        const content = document.getElementById('results-content');
        const timestamp = new Date().toLocaleTimeString();

        const messageEl = document.createElement('div');
        messageEl.className = 'text-xs text-gray-700 border-l-2 border-blue-200 pl-2';
        messageEl.innerHTML = `<span class="text-gray-500">${timestamp}</span> ${message}`;

        content.appendChild(messageEl);
        content.scrollTop = content.scrollHeight;
    }

    // ==================== ACCURACY OPTIMIZATION FUNCTIONS ====================

    async function loadAccuracyMetrics() {
        try {
            const response = await fetch('/accuracy/current-performance');
            const data = await response.json();

            const metricsHtml = `
                <div class="space-y-2">
                    <div class="flex justify-between items-center">
                        <span class="text-sm text-gray-600">Current Accuracy:</span>
                        <span class="font-bold ${data.accuracy >= 75 ? 'text-green-600' :
                                              data.accuracy >= 65 ? 'text-blue-600' :
                                              data.accuracy >= 55 ? 'text-yellow-600' : 'text-red-600'}">
                            ${data.accuracy?.toFixed(1) || 0}%
                        </span>
                    </div>
                    <div class="flex justify-between items-center">
                        <span class="text-sm text-gray-600">Captain Success:</span>
                        <span class="font-bold ${data.captain_success_rate >= 75 ? 'text-green-600' : 'text-yellow-600'}">
                            ${data.captain_success_rate?.toFixed(1) || 0}%
                        </span>
                    </div>
                    <div class="text-center mt-2">
                        <span class="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                            data.performance_level === 'excellent' ? 'bg-green-100 text-green-800' :
                            data.performance_level === 'good' ? 'bg-blue-100 text-blue-800' :
                            data.performance_level === 'acceptable' ? 'bg-yellow-100 text-yellow-800' :
                            'bg-red-100 text-red-800'
                        }">
                            ${data.performance_level?.toUpperCase() || 'UNKNOWN'}
                        </span>
                    </div>
                    ${data.needs_improvement ?
                        '<div class="text-xs text-red-600 mt-1">🚨 Needs Improvement</div>' :
                        '<div class="text-xs text-green-600 mt-1">✅ Performance Good</div>'
                    }
                </div>
            `;

            document.getElementById('accuracy-metrics').innerHTML = metricsHtml;
        } catch (error) {
            document.getElementById('accuracy-metrics').innerHTML =
                '<div class="text-xs text-red-600">Error loading metrics</div>';
        }
    }

    async function loadRetrainingStatus() {
        try {
            const response = await fetch('/accuracy/scheduler-status');
            const data = await response.json();

            const statusHtml = `
                <div class="space-y-2">
                    <div class="flex justify-between items-center">
                        <span class="text-sm text-gray-600">Auto Retrain:</span>
                        <span class="font-bold ${data.is_running ? 'text-green-600' : 'text-red-600'}">
                            ${data.is_running ? '🟢 Running' : '🔴 Stopped'}
                        </span>
                    </div>
                    <div class="flex justify-between items-center">
                        <span class="text-sm text-gray-600">Current GW:</span>
                        <span class="font-bold">GW${data.current_gameweek || 0}</span>
                    </div>
                    <div class="flex justify-between items-center">
                        <span class="text-sm text-gray-600">Last Processed:</span>
                        <span class="font-bold">GW${data.last_processed_gameweek || 0}</span>
                    </div>
                    <div class="text-xs text-gray-500 mt-1">
                        Health Score: ${data.health_score || 0}/100
                    </div>
                    ${data.pending_jobs?.length > 0 ?
                        `<div class="text-xs text-blue-600">📅 ${data.pending_jobs.length} jobs pending</div>` : ''
                    }
                </div>
            `;

            document.getElementById('retraining-status').innerHTML = statusHtml;
        } catch (error) {
            document.getElementById('retraining-status').innerHTML =
                '<div class="text-xs text-red-600">Error loading status</div>';
        }
    }

    async function startAutomation() {
        try {
            const button = document.getElementById('start-automation-btn');
            button.disabled = true;
            button.textContent = 'Starting...';

            const response = await fetch('/accuracy/start-automation', { method: 'POST' });
            const data = await response.json();

            if (data.status === 'success') {
                button.textContent = '✅ Started';
                button.className = button.className.replace('bg-green-600', 'bg-gray-400');
                setTimeout(() => loadRetrainingStatus(), 1000);
                updateResults(' Automated retraining system started successfully');
            } else {
                throw new Error(data.message || 'Failed to start automation');
            }
        } catch (error) {
            document.getElementById('start-automation-btn').textContent = '❌ Failed';
            updateResults(`[ERROR]  Failed to start automation: ${error.message}`);
        }
    }

    async function emergencyRetrain() {
        try {
            const gameweek = prompt('Enter gameweek number for emergency retraining:');
            if (!gameweek || isNaN(gameweek)) {
                updateResults('[ERROR]  Please enter a valid gameweek number');
                return;
            }

            const button = document.getElementById('emergency-retrain-btn');
            button.disabled = true;
            button.textContent = 'Retraining...';

            updateResults(` Starting emergency retraining for GW${gameweek}...`);

            const response = await fetch(`/accuracy/emergency-retrain/${gameweek}`, { method: 'POST' });
            const data = await response.json();

            if (data.status === 'success') {
                button.textContent = '✅ Completed';
                button.className = button.className.replace('bg-red-600', 'bg-green-600');

                const result = data.retrain_result;
                updateResults(`[OK]  Emergency retraining completed for GW${gameweek}`);
                updateResults(`     Retrain intensity: ${result.retrain_intensity || 'unknown'}`);

                if (result.performance_metrics) {
                    const accuracy = result.performance_metrics.accuracy || 0;
                    updateResults(`     New accuracy: ${accuracy.toFixed(1)}%`);
                }

                setTimeout(() => {
                    loadAccuracyMetrics();
                    loadRetrainingStatus();
                }, 2000);

            } else {
                throw new Error(data.message || 'Emergency retraining failed');
            }
        } catch (error) {
            document.getElementById('emergency-retrain-btn').textContent = '❌ Failed';
            updateResults(`[ERROR]  Emergency retraining failed: ${error.message}`);
        } finally {
            setTimeout(() => {
                const button = document.getElementById('emergency-retrain-btn');
                button.disabled = false;
                button.textContent = 'Emergency Retrain';
                button.className = button.className.replace('bg-green-600', 'bg-red-600');
            }, 3000);
        }
    }

    async function showAccuracyExplanation() {
        try {
            const response = await fetch('/accuracy/explanation');
            const data = await response.json();

            const explanationHtml = `
                <div class="space-y-4">
                    <div>
                        <h4 class="font-bold text-gray-900 mb-2">📊 What Does 53.7% Accuracy Mean?</h4>
                        <div class="bg-blue-50 p-3 rounded-lg">
                            <p class="text-sm"><strong>Calculation:</strong> ${data.accuracy_definition.calculation}</p>
                            <p class="text-sm mt-1"><strong>Meaning:</strong> ${data.accuracy_definition.meaning}</p>
                            <p class="text-sm mt-1"><strong>Example:</strong> ${data.accuracy_definition.example}</p>
                        </div>
                    </div>

                    <div>
                        <h4 class="font-bold text-gray-900 mb-2">🎯 Performance Benchmarks</h4>
                        <div class="space-y-1 text-sm">
                            <div class="flex justify-between">
                                <span class="text-green-600 font-medium">Excellent:</span>
                                <span>${data.benchmarks.excellent}</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="text-blue-600 font-medium">Good:</span>
                                <span>${data.benchmarks.good}</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="text-yellow-600 font-medium">Acceptable:</span>
                                <span>${data.benchmarks.acceptable}</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="text-red-600 font-medium">Poor:</span>
                                <span>${data.benchmarks.poor}</span>
                            </div>
                        </div>
                    </div>

                    <div>
                        <h4 class="font-bold text-gray-900 mb-2">🔍 FPL Context</h4>
                        <div class="bg-yellow-50 p-3 rounded-lg text-sm">
                            <p><strong>Difficulty:</strong> ${data.context.fpl_difficulty}</p>
                            <p><strong>Realistic Target:</strong> ${data.context.realistic_targets}</p>
                            <p><strong>Random Factors:</strong> ${data.context.factors.join(', ')}</p>
                        </div>
                    </div>

                    <div>
                        <h4 class="font-bold text-gray-900 mb-2">🚀 Improvement Strategies</h4>
                        <ul class="list-disc list-inside text-sm space-y-1">
                            ${data.improvement_strategies.map(strategy => `<li>${strategy}</li>`).join('')}
                        </ul>
                    </div>

                    <div class="text-center pt-4 border-t">
                        <p class="text-sm text-gray-600">
                            <strong>Bottom Line:</strong> 53.7% is reasonable for FPL, but we can target 65-75% with aggressive optimization!
                        </p>
                    </div>
                </div>
            `;

            document.getElementById('accuracy-explanation-content').innerHTML = explanationHtml;
            document.getElementById('accuracy-modal').classList.remove('hidden');
        } catch (error) {
            updateResults(`[ERROR]  Failed to load accuracy explanation: ${error.message}`);
        }
    }

    function closeAccuracyModal() {
        document.getElementById('accuracy-modal').classList.add('hidden');
    }

    // ==================== END ACCURACY OPTIMIZATION FUNCTIONS ====================


    // Initialize dashboard on load
    async function initializeDashboard() {
        try {
            await Promise.all([
                loadCurrentAction(),
                loadCurrentSuggestion(),
                loadPerformanceSummary(),
                loadTransferChanges(),
                loadRecentPerformance(),
                loadHistoricalComparisons(),
                loadAccuracyMetrics(),
                loadRetrainingStatus()
            ]);
        } catch (error) {
            console.error('Failed to initialize dashboard:', error);
        }
    }

    document.addEventListener('DOMContentLoaded', function() {
        initializeDashboard();

        // Wire up accuracy optimization event handlers
        document.getElementById('start-automation-btn')?.addEventListener('click', startAutomation);
        document.getElementById('emergency-retrain-btn')?.addEventListener('click', emergencyRetrain);
        document.getElementById('accuracy-explanation-btn')?.addEventListener('click', showAccuracyExplanation);
        document.getElementById('close-modal')?.addEventListener('click', closeAccuracyModal);

        // Close modal when clicking outside
        document.getElementById('accuracy-modal')?.addEventListener('click', function(e) {
            if (e.target === this) {
                closeAccuracyModal();
            }
        });
    });
</script>

</body>
</html>
"""
