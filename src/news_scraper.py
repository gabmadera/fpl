from __future__ import annotations

from typing import List, Dict
import time
import requests
from bs4 import BeautifulSoup


class NewsScraper:
    SOURCES = [
        "https://www.premierleague.com/news",
        "https://www.skysports.com/premier-league-news",
        "https://www.bbc.com/sport/football/premier-league",
    ]

    def __init__(self) -> None:
        self.headers = {"User-Agent": "Mozilla/5.0 (compatible; FPL-AI/1.0)"}

    def fetch_latest(self, keywords: List[str] | None = None) -> List[Dict]:
        results: List[Dict] = []
        for url in self.SOURCES:
            try:
                time.sleep(0.5)
                resp = requests.get(url, headers=self.headers, timeout=30)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "lxml")
                for a in soup.find_all("a"):
                    text = (a.get_text() or "").strip()
                    href = a.get("href") or ""
                    if not text or not href:
                        continue
                    lower = text.lower()
                    if keywords and not any(k.lower() in lower for k in keywords):
                        continue
                    results.append({"title": text[:120], "url": href})
            except Exception:
                continue
        return results[:50]

