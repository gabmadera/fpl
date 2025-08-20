from __future__ import annotations

import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
import logging


class CacheManager:
    """Smart caching system to avoid regenerating predictions and data constantly"""
    
    def __init__(self):
        self.cache_dir = Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Cache durations (in hours)
        self.cache_durations = {
            'predictions': 2,        # 2 hours for predictions
            'team_suggestions': 1,   # 1 hour for team suggestions
            'fixtures': 6,          # 6 hours for fixture data
            'transfer_suggestions': 2,  # 2 hours for transfer suggestions
            'actual_points': 24,    # 24 hours for actual points
            'accuracy_report': 1,   # 1 hour for accuracy reports
            'performance_history': 12  # 12 hours for performance history
        }
    
    def is_cache_valid(self, cache_type: str, custom_duration: Optional[int] = None) -> bool:
        """Check if cache is still valid"""
        try:
            cache_file = self.cache_dir / f"{cache_type}_cache.json"
            if not cache_file.exists():
                return False
            
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            cache_time = datetime.fromisoformat(cache_data.get('timestamp', '2000-01-01'))
            duration = custom_duration or self.cache_durations.get(cache_type, 2)
            
            is_valid = datetime.now() - cache_time < timedelta(hours=duration)
            
            if not is_valid:
                self.logger.info(f"Cache expired for {cache_type} (age: {datetime.now() - cache_time})")
            
            return is_valid
            
        except Exception as e:
            self.logger.error(f"Cache validation failed for {cache_type}: {e}")
            return False
    
    def get_cached_data(self, cache_type: str) -> Optional[Any]:
        """Get cached data if valid"""
        if not self.is_cache_valid(cache_type):
            return None
        
        try:
            cache_file = self.cache_dir / f"{cache_type}_cache.json"
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Handle different data types
            data = cache_data.get('data')
            
            # Convert back to DataFrame if it was one
            if cache_data.get('data_type') == 'dataframe' and data:
                return pd.DataFrame(data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load cached data for {cache_type}: {e}")
            return None
    
    def cache_data(self, cache_type: str, data: Any, metadata: Dict[str, Any] = None) -> bool:
        """Cache data with timestamp and metadata"""
        try:
            cache_file = self.cache_dir / f"{cache_type}_cache.json"
            
            # Prepare data for JSON serialization
            if isinstance(data, pd.DataFrame):
                serialized_data = data.to_dict(orient='records')
                data_type = 'dataframe'
            else:
                serialized_data = data
                data_type = 'json'
            
            cache_content = {
                'timestamp': datetime.now().isoformat(),
                'data_type': data_type,
                'data': serialized_data,
                'cache_type': cache_type,
                'metadata': metadata or {}
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_content, f, indent=2)
            
            self.logger.info(f"Cached data for {cache_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cache data for {cache_type}: {e}")
            return False
    
    def get_cache_status(self, cache_type: str) -> Dict[str, Any]:
        """Get detailed cache status information"""
        try:
            cache_file = self.cache_dir / f"{cache_type}_cache.json"
            
            if not cache_file.exists():
                return {
                    'exists': False,
                    'valid': False,
                    'age': None,
                    'last_updated': None,
                    'expires_in': None
                }
            
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            cache_time = datetime.fromisoformat(cache_data.get('timestamp', '2000-01-01'))
            duration = self.cache_durations.get(cache_type, 2)
            expires_at = cache_time + timedelta(hours=duration)
            
            age = datetime.now() - cache_time
            is_valid = age < timedelta(hours=duration)
            
            time_until_expiry = expires_at - datetime.now()
            
            return {
                'exists': True,
                'valid': is_valid,
                'age': str(age).split('.')[0],  # Remove microseconds
                'last_updated': cache_time.strftime('%Y-%m-%d %H:%M:%S'),
                'expires_in': str(time_until_expiry).split('.')[0] if time_until_expiry.total_seconds() > 0 else 'Expired',
                'metadata': cache_data.get('metadata', {})
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get cache status for {cache_type}: {e}")
            return {'exists': False, 'valid': False, 'error': str(e)}
    
    def clear_cache(self, cache_type: Optional[str] = None) -> bool:
        """Clear specific cache or all caches"""
        try:
            if cache_type:
                cache_file = self.cache_dir / f"{cache_type}_cache.json"
                if cache_file.exists():
                    cache_file.unlink()
                    self.logger.info(f"Cleared cache for {cache_type}")
            else:
                # Clear all caches
                for cache_file in self.cache_dir.glob("*_cache.json"):
                    cache_file.unlink()
                self.logger.info("Cleared all caches")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
            return False
    
    def get_all_cache_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all caches"""
        status = {}
        
        for cache_type in self.cache_durations.keys():
            status[cache_type] = self.get_cache_status(cache_type)
        
        # Also check for any other cache files
        for cache_file in self.cache_dir.glob("*_cache.json"):
            cache_type = cache_file.stem.replace('_cache', '')
            if cache_type not in status:
                status[cache_type] = self.get_cache_status(cache_type)
        
        return status
    
    def cache_with_fallback(self, cache_type: str, data_generator, *args, **kwargs) -> Tuple[Any, bool]:
        """
        Get cached data or generate new data if cache is invalid
        Returns: (data, was_cached)
        """
        # Try to get cached data first
        cached_data = self.get_cached_data(cache_type)
        if cached_data is not None:
            self.logger.info(f"Using cached data for {cache_type}")
            return cached_data, True
        
        # Generate new data
        try:
            self.logger.info(f"Generating fresh data for {cache_type}")
            new_data = data_generator(*args, **kwargs)
            
            # Cache the new data
            self.cache_data(cache_type, new_data)
            
            return new_data, False
            
        except Exception as e:
            self.logger.error(f"Failed to generate data for {cache_type}: {e}")
            # Return empty fallback
            return pd.DataFrame() if 'dataframe' in cache_type else {}, False