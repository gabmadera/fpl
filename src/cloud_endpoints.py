"""
Cloud-specific endpoints for Render deployment
Handles initialization, health checks, and fallback systems
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import os
import logging
from typing import Dict, Any
from .cloud_init import CloudInitializer
from .cloud_fpl_client import CloudFPLClient

router = APIRouter()
logger = logging.getLogger("CloudEndpoints")

# Global instances
cloud_init = CloudInitializer()
cloud_fpl = CloudFPLClient()

@router.get("/cloud/health")
async def cloud_health() -> Dict[str, Any]:
    """Comprehensive health check for cloud deployment"""
    try:
        # Check FPL API accessibility
        fpl_health = cloud_fpl.health_check()

        # Check initialization status
        init_status = cloud_init.get_initialization_status()

        # Check environment
        environment = {
            "is_render": bool(os.getenv('RENDER')),
            "port": os.getenv('PORT', 'not_set'),
            "python_version": os.getenv('PYTHON_VERSION', 'unknown')
        }

        return {
            "status": "healthy",
            "timestamp": "2024-09-26T22:15:00Z",
            "components": {
                "fpl_api": fpl_health,
                "initialization": init_status,
                "environment": environment
            },
            "ready_for_predictions": init_status["is_initialized"] and
                                   (fpl_health["api_accessible"] or fpl_health.get("fallback_available", False))
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": "2024-09-26T22:15:00Z"
        }

@router.post("/cloud/initialize")
async def initialize_cloud_system() -> Dict[str, Any]:
    """Initialize the cloud system with bootstrap data"""
    try:
        if cloud_init.is_initialized():
            return {
                "status": "already_initialized",
                "message": "System already initialized",
                "timestamp": "2024-09-26T22:15:00Z"
            }

        result = cloud_init.initialize_system()

        if result["status"] == "success":
            logger.info("Cloud system initialized successfully")
        else:
            logger.error(f"Cloud initialization failed: {result['message']}")

        return result

    except Exception as e:
        logger.error(f"Initialization error: {e}")
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")

@router.get("/cloud/status")
async def get_cloud_status() -> Dict[str, Any]:
    """Get detailed cloud deployment status"""
    try:
        return {
            "deployment": {
                "platform": "render",
                "is_cloud": bool(os.getenv('RENDER') or os.getenv('PORT')),
                "initialized": cloud_init.is_initialized()
            },
            "api_status": cloud_fpl.health_check(),
            "initialization": cloud_init.get_initialization_status(),
            "environment": {
                "render": os.getenv('RENDER', 'false'),
                "port": os.getenv('PORT', 'not_set'),
                "python_version": os.getenv('PYTHON_VERSION', 'unknown')
            }
        }

    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return {
            "error": str(e),
            "status": "error"
        }

@router.get("/cloud/bootstrap-data")
async def get_bootstrap_data() -> Dict[str, Any]:
    """Get essential bootstrap data for frontend"""
    try:
        import json
        from pathlib import Path

        data_dir = Path("data/bootstrap")

        # Load gameweeks
        gameweeks_file = data_dir / "gameweeks.json"
        gameweeks = {}
        if gameweeks_file.exists():
            with open(gameweeks_file, 'r') as f:
                gameweeks = json.load(f)

        # Load teams
        teams_file = data_dir / "teams.json"
        teams = {}
        if teams_file.exists():
            with open(teams_file, 'r') as f:
                teams = json.load(f)

        return {
            "status": "success",
            "data": {
                "gameweeks": gameweeks,
                "teams": teams,
                "source": "bootstrap_files"
            }
        }

    except Exception as e:
        logger.error(f"Bootstrap data error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "data": {
                "gameweeks": {"current_gameweek": 6},
                "teams": {"teams": []},
                "source": "fallback"
            }
        }

@router.get("/cloud/predictions-fallback")
async def get_fallback_predictions() -> Dict[str, Any]:
    """Get predictions using fallback system when FPL API is blocked"""
    try:
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))
        from models.minimal.simple_predictor import SimplePredictor

        predictor = SimplePredictor()

        # Generate fallback team
        team_prediction = predictor._generate_fallback_team()

        return {
            "status": "success",
            "gameweek": 6,
            "team_summary": {
                "starters": team_prediction["starters"],
                "bench": team_prediction["bench"],
                "formation": team_prediction["formation"],
                "total_predicted_points": team_prediction["total_predicted_points"],
                "captain_id": team_prediction["captain_id"],
                "vice_captain_id": team_prediction["vice_captain_id"],
                "total_cost": team_prediction["total_cost"]
            },
            "source": "fallback_predictor",
            "message": "Using fallback predictions - FPL API unavailable"
        }

    except Exception as e:
        logger.error(f"Fallback predictions error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "message": "Fallback predictions failed"
        }

@router.post("/cloud/force-refresh")
async def force_refresh_system() -> Dict[str, Any]:
    """Force refresh all system components"""
    try:
        # Clear caches
        cloud_fpl.cache.clear() if hasattr(cloud_fpl.cache, 'clear') else None

        # Check API status
        api_health = cloud_fpl.health_check()

        # Re-initialize if needed
        if not cloud_init.is_initialized():
            init_result = cloud_init.initialize_system()
        else:
            init_result = {"status": "already_initialized"}

        return {
            "status": "success",
            "refresh_time": "2024-09-26T22:15:00Z",
            "components_refreshed": {
                "api_cache": "cleared",
                "api_status": api_health,
                "initialization": init_result
            }
        }

    except Exception as e:
        logger.error(f"Force refresh error: {e}")
        return {
            "status": "error",
            "error": str(e)
        }