from __future__ import annotations

import os
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests


class TwitterNewsClient:
    """Minimal Twitter/X client using bearer token for public tweets.
    Focus: fetch recent tweets from curated handles to adjust availability signals.
    """

    def __init__(self, bearer_token: Optional[str] = None, handles: Optional[List[str]] = None) -> None:
        self.logger = logging.getLogger(__name__)
        self.bearer = bearer_token or os.getenv("TWITTER_BEARER_TOKEN", "")
        self.session = requests.Session()
        if self.bearer:
            self.session.headers.update({
                "Authorization": f"Bearer {self.bearer}",
                "User-Agent": "FPL-AI-News/1.0"
            })
        env_handles = os.getenv("TWITTER_HANDLE_LIST", "")
        self.handles = handles or [h.strip() for h in env_handles.split(",") if h.strip()]
        # Simple caching to avoid rate limits
        self.cache_dir = Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "twitter_news_cache.json"
        self.user_cache_file = self.cache_dir / "twitter_user_ids.json"
        self.min_refresh = int(os.getenv("TWITTER_MIN_REFRESH_SECONDS", "900"))  # default 15 minutes
        self.cooldown_until = 0.0

    def _get_user(self, username: str) -> Optional[str]:
        try:
            # load id cache
            ids: Dict[str, str] = {}
            if self.user_cache_file.exists():
                try:
                    ids = json.loads(self.user_cache_file.read_text())
                except Exception:
                    ids = {}
            if username in ids:
                return ids[username]
            r = self.session.get(
                "https://api.twitter.com/2/users/by/username/" + username,
                params={"user.fields": "id,verified"}, timeout=20
            )
            if r.status_code == 429:
                self.cooldown_until = time.time() + self.min_refresh
                return None
            if not r.ok:
                return None
            data = r.json().get("data")
            uid = data.get("id") if data else None
            if uid:
                ids[username] = uid
                try:
                    self.user_cache_file.write_text(json.dumps(ids))
                except Exception:
                    pass
            return uid
        except Exception:
            return None

    def _get_recent_tweets(self, user_id: str, max_results: int = 10) -> List[Dict[str, Any]]:
        try:
            r = self.session.get(
                f"https://api.twitter.com/2/users/{user_id}/tweets",
                params={
                    "max_results": max(5, min(100, max_results)),
                    "tweet.fields": "created_at,lang,public_metrics",
                    "exclude": "replies"
                }, timeout=20
            )
            if r.status_code == 429:
                self.cooldown_until = time.time() + self.min_refresh
                return []
            if not r.ok:
                return []
            return r.json().get("data", [])
        except Exception:
            return []

    def fetch_news(self, lookback_handles: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Fetch recent tweets for curated handles with caching and backoff."""
        if not self.bearer:
            return []
        handles = lookback_handles or self.handles
        # Backoff if cooling down
        now = time.time()
        if now < self.cooldown_until:
            return self._load_cached_tweets()
        # Use cache if fresh
        cache = self._load_cache()
        if cache and (now - cache.get("fetched_at", 0)) < self.min_refresh:
            return cache.get("tweets", [])
        # Fetch fresh
        out: List[Dict[str, Any]] = []
        for h in handles:
            uid = self._get_user(h)
            if not uid:
                continue
            tweets = self._get_recent_tweets(uid)
            for t in tweets:
                out.append({
                    "handle": h,
                    "text": t.get("text", ""),
                    "created_at": t.get("created_at"),
                    "likes": t.get("public_metrics", {}).get("like_count"),
                    "retweets": t.get("public_metrics", {}).get("retweet_count"),
                })
            time.sleep(0.2)
        if out:
            self._save_cache(out)
            return out
        # If nothing fetched (maybe rate-limited), return cached tweets if any
        cached = self._load_cached_tweets()
        return cached

    def _load_cache(self) -> Dict[str, Any]:
        try:
            if not self.cache_file.exists():
                return {}
            data = json.loads(self.cache_file.read_text())
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _load_cached_tweets(self) -> List[Dict[str, Any]]:
        data = self._load_cache()
        return data.get("tweets", []) if isinstance(data, dict) else []

    def _save_cache(self, tweets: List[Dict[str, Any]]):
        try:
            payload = {"fetched_at": time.time(), "tweets": tweets}
            self.cache_file.write_text(json.dumps(payload))
        except Exception:
            pass

    @staticmethod
    def extract_availability_signals(tweets: List[Dict[str, Any]]) -> Dict[str, float]:
        """Very naive keyword-based signal extractor mapping player last names to availability score deltas.
        Positive -> more likely to start; Negative -> less likely to start.
        """
        signals: Dict[str, float] = {}
        keywords_pos = ["starts", "starting", "fit", "available", "in training", "back"]
        keywords_neg = ["doubt", "doubtful", "injury", "setback", "out", "not in squad", "rested"]
        for t in tweets:
            text = (t.get("text") or "").lower()
            for word in keywords_pos:
                if word in text:
                    for token in text.split():
                        tok = token.strip(".,:;!?()[]{}\"'`").lower()
                        if tok.isalpha() and len(tok) > 2:
                            signals[tok] = signals.get(tok, 0.0) + 0.1
            for word in keywords_neg:
                if word in text:
                    for token in text.split():
                        tok = token.strip(".,:;!?()[]{}\"'`").lower()
                        if tok.isalpha() and len(tok) > 2:
                            signals[tok] = signals.get(tok, 0.0) - 0.1
        # Clamp signal magnitude
        for k, v in list(signals.items()):
            signals[k] = max(-0.3, min(0.3, v))
        return signals

    @staticmethod
    def extract_enhanced_signals(tweets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract richer signals: availability, set-pieces, position, and recommendations.
        Returns a dict with keys: availability, set_pieces, position, recommend.
        """
        availability = TwitterNewsClient.extract_availability_signals(tweets)
        set_pieces: Dict[str, Dict[str, bool]] = {}
        position: Dict[str, str] = {}
        recommend: Dict[str, float] = {}

        # Simple keyword dictionaries
        pens_kw = ["on pens", "takes pens", "penalties", "on penalties", "pen taker"]
        corners_kw = ["on corners", "takes corners", "corner duty", "set pieces", "set-pieces"]
        fks_kw = ["free kick", "free-kick", "free kicks", "on free-kicks"]
        rec_kw_pos = ["good pick", "great pick", "must own", "transfer in", "buy", "recommend", "captain", "vc"]
        rec_kw_neg = ["avoid", "sell", "transfer out", "bad pick"]
        pos_map = {
            "rb": "RB", "right back": "RB", "lb": "LB", "left back": "LB",
            "rwb": "RWB", "lwb": "LWB", "dm": "DM", "cdm": "DM", "cm": "CM",
            "am": "AM", "cam": "AM", "10": "AM", "no.10": "AM", "lw": "LW",
            "rw": "RW", "st": "ST", "cf": "CF", "striker": "ST", "f9": "F9", "false nine": "F9"
        }

        for t in tweets:
            text_raw = t.get("text") or ""
            text = text_raw.lower()
            tokens = [tok.strip(".,:;!?()[]{}\"'`") for tok in text.split()]

            # Find the most likely last-name token (very naive heuristic: any token > 2 letters)
            name_candidates = [tok for tok in tokens if tok.isalpha() and len(tok) > 2]

            # Set pieces
            def ensure_player_entry(last: str):
                if last not in set_pieces:
                    set_pieces[last] = {"pens": False, "corners": False, "fks": False}

            if any(kw in text for kw in pens_kw):
                for last in name_candidates:
                    ensure_player_entry(last)
                    set_pieces[last]["pens"] = True

            if any(kw in text for kw in corners_kw):
                for last in name_candidates:
                    ensure_player_entry(last)
                    set_pieces[last]["corners"] = True

            if any(kw in text for kw in fks_kw):
                for last in name_candidates:
                    ensure_player_entry(last)
                    set_pieces[last]["fks"] = True

            # Position hints
            for key, code in pos_map.items():
                if key in text:
                    for last in name_candidates:
                        position[last] = code

            # Recommendations (very rough sentiment)
            pos_hit = any(kw in text for kw in rec_kw_pos)
            neg_hit = any(kw in text for kw in rec_kw_neg)
            if pos_hit or neg_hit:
                delta = (0.15 if pos_hit else 0.0) - (0.12 if neg_hit else 0.0)
                if abs(delta) > 0:
                    for last in name_candidates:
                        recommend[last] = max(-0.2, min(0.2, recommend.get(last, 0.0) + delta))

        return {
            "availability": availability,
            "set_pieces": set_pieces,
            "position": position,
            "recommend": recommend,
        }


