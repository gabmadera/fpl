"""
Cloud Data Fetcher for FPL AI Dashboard
Fetches and caches FPL data on cloud deployment startup
"""

import os
import json
import requests
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import time

logger = logging.getLogger("CloudDataFetcher")

class CloudDataFetcher:
    """Fetches and processes FPL data for cloud deployment"""

    def __init__(self):
        self.is_cloud = bool(os.getenv('RENDER') or os.getenv('PORT'))
        self.base_url = "https://fantasy.premierleague.com/api"
        self.data_dir = Path("data")
        self.cache_dir = self.data_dir / "cache"
        self.processed_dir = self.data_dir / "processed"

        # Create directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Headers for FPL API
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    def fetch_bootstrap_data(self) -> Optional[Dict[str, Any]]:
        """Fetch bootstrap-static data from FPL API"""
        try:
            logger.info("Fetching FPL bootstrap data...")
            response = requests.get(
                f"{self.base_url}/bootstrap-static/",
                headers=self.headers,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()

                # Cache the data
                cache_file = self.cache_dir / "bootstrap_static.json"
                with open(cache_file, 'w') as f:
                    json.dump(data, f, indent=2)

                logger.info(f"Successfully cached bootstrap data ({len(data.get('elements', []))} players)")
                return data
            else:
                logger.warning(f"FPL API returned {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Failed to fetch bootstrap data: {e}")
            return None

    def fetch_fixtures_data(self) -> Optional[List[Dict[str, Any]]]:
        """Fetch fixtures data from FPL API"""
        try:
            logger.info("Fetching FPL fixtures data...")
            response = requests.get(
                f"{self.base_url}/fixtures/",
                headers=self.headers,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()

                # Cache the data
                cache_file = self.cache_dir / "fixtures.json"
                with open(cache_file, 'w') as f:
                    json.dump(data, f, indent=2)

                logger.info(f"Successfully cached fixtures data ({len(data)} fixtures)")
                return data
            else:
                logger.warning(f"FPL API returned {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Failed to fetch fixtures data: {e}")
            return None

    def process_bootstrap_data(self, bootstrap_data: Dict[str, Any]) -> bool:
        """Process bootstrap data into usable format"""
        try:
            logger.info("Processing bootstrap data...")

            # Extract key components
            elements = bootstrap_data.get('elements', [])
            teams = bootstrap_data.get('teams', [])
            events = bootstrap_data.get('events', [])
            element_types = bootstrap_data.get('element_types', [])

            # Find current gameweek
            current_event = None
            for event in events:
                if event.get('is_current', False):
                    current_event = event
                    break

            if not current_event:
                current_event = events[-1] if events else {'id': 6, 'name': 'Gameweek 6'}

            # Process players data
            processed_players = []
            for player in elements:
                team_data = next((t for t in teams if t['id'] == player['team']), {})
                position_data = next((p for p in element_types if p['id'] == player['element_type']), {})

                processed_player = {
                    'id': player['id'],
                    'name': f"{player.get('first_name', '')} {player.get('second_name', '')}".strip(),
                    'position': position_data.get('singular_name_short', 'MID'),
                    'team': player['team'],
                    'team_name': team_data.get('name', 'Unknown'),
                    'team_short': team_data.get('short_name', 'UNK'),
                    'price': player.get('now_cost', 50) / 10.0,
                    'total_points': player.get('total_points', 0),
                    'form': float(player.get('form', '0')),
                    'selected_by_percent': float(player.get('selected_by_percent', '0')),
                    'points_per_game': float(player.get('points_per_game', '0')),
                    'status': player.get('status', 'a'),
                    'chance_of_playing_this_round': player.get('chance_of_playing_this_round'),
                    'chance_of_playing_next_round': player.get('chance_of_playing_next_round'),
                    'news': player.get('news', ''),
                    'minutes': player.get('minutes', 0),
                    'goals_scored': player.get('goals_scored', 0),
                    'assists': player.get('assists', 0),
                    'clean_sheets': player.get('clean_sheets', 0),
                    'goals_conceded': player.get('goals_conceded', 0),
                    'yellow_cards': player.get('yellow_cards', 0),
                    'red_cards': player.get('red_cards', 0),
                    'saves': player.get('saves', 0),
                    'bonus': player.get('bonus', 0),
                    'bps': player.get('bps', 0)
                }
                processed_players.append(processed_player)

            # Save processed data
            processed_data = {
                'players': processed_players,
                'teams': teams,
                'events': events,
                'current_gameweek': current_event['id'],
                'last_updated': datetime.now().isoformat(),
                'element_types': element_types
            }

            # Save to multiple formats for compatibility
            output_file = self.processed_dir / "cloud_fpl_data.json"
            with open(output_file, 'w') as f:
                json.dump(processed_data, f, indent=2)

            # Also save in the format expected by existing code
            players_file = self.processed_dir / "current_players.json"
            with open(players_file, 'w') as f:
                json.dump(processed_players, f, indent=2)

            gameweeks_file = self.processed_dir / "current_gameweeks.json"
            with open(gameweeks_file, 'w') as f:
                json.dump(events, f, indent=2)

            logger.info(f"Successfully processed {len(processed_players)} players")
            return True

        except Exception as e:
            logger.error(f"Failed to process bootstrap data: {e}")
            return False

    def load_cached_data(self) -> Optional[Dict[str, Any]]:
        """Load cached data if available and recent"""
        try:
            cache_file = self.cache_dir / "bootstrap_static.json"

            if not cache_file.exists():
                return None

            # Check if cache is recent (less than 6 hours old)
            file_age = time.time() - cache_file.stat().st_mtime
            max_age = 6 * 3600  # 6 hours

            if file_age > max_age:
                logger.info("Cache expired, will fetch fresh data")
                return None

            with open(cache_file, 'r') as f:
                data = json.load(f)

            logger.info("Using cached FPL data")
            return data

        except Exception as e:
            logger.error(f"Failed to load cached data: {e}")
            return None

    def ensure_fpl_data(self) -> bool:
        """Ensure FPL data is available, fetch if necessary"""
        try:
            # Try to load cached data first
            bootstrap_data = self.load_cached_data()

            # If no cache or cache expired, fetch fresh data
            if not bootstrap_data:
                bootstrap_data = self.fetch_bootstrap_data()

                # If API fails, try to use any existing cache
                if not bootstrap_data:
                    logger.warning("FPL API unavailable, attempting to use existing cache")
                    cache_file = self.cache_dir / "bootstrap_static.json"
                    if cache_file.exists():
                        with open(cache_file, 'r') as f:
                            bootstrap_data = json.load(f)
                        logger.info("Using stale cache data")
                    else:
                        logger.error("No cached data available")
                        return False

            # Process the data
            if bootstrap_data:
                success = self.process_bootstrap_data(bootstrap_data)
                if success:
                    # Also fetch fixtures
                    self.fetch_fixtures_data()
                return success

            return False

        except Exception as e:
            logger.error(f"Failed to ensure FPL data: {e}")
            return False

    def get_data_status(self) -> Dict[str, Any]:
        """Get status of available data"""
        status = {
            'has_bootstrap_cache': (self.cache_dir / "bootstrap_static.json").exists(),
            'has_processed_data': (self.processed_dir / "cloud_fpl_data.json").exists(),
            'has_fixtures': (self.cache_dir / "fixtures.json").exists(),
            'cache_age_hours': 0,
            'last_updated': None
        }

        try:
            cache_file = self.cache_dir / "bootstrap_static.json"
            if cache_file.exists():
                file_age = time.time() - cache_file.stat().st_mtime
                status['cache_age_hours'] = round(file_age / 3600, 1)
                status['last_updated'] = datetime.fromtimestamp(
                    cache_file.stat().st_mtime
                ).isoformat()
        except:
            pass

        return status