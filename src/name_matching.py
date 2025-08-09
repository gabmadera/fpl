from __future__ import annotations

from typing import Dict, Optional, List
import unicodedata
import re
import pandas as pd
from rapidfuzz import process, fuzz


class PlayerNameMatcher:
    def __init__(self, fpl_players: pd.DataFrame) -> None:
        self.fpl_players = fpl_players.copy()
        # Normalize and add helper columns
        self.fpl_players["name_norm"] = self.fpl_players["name"].astype(str).map(self._normalize)
        self.fpl_players["last_name"] = self.fpl_players["name_norm"].map(self._extract_last_name)
        self.names = self.fpl_players["name"].astype(str).tolist()
        # Build last-name index for uniqueness checks
        self.last_to_indices: Dict[str, List[int]] = {}
        for idx, ln in enumerate(self.fpl_players["last_name"].tolist()):
            if not ln:
                continue
            self.last_to_indices.setdefault(ln, []).append(idx)

    @staticmethod
    def _normalize(s: str) -> str:
        s = unicodedata.normalize("NFKD", s)
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
        s = s.replace(".", " ")
        s = re.sub(r"[^a-zA-Z\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip().lower()
        return s

    @staticmethod
    def _extract_last_name(s: str) -> str:
        if not s:
            return ""
        parts = s.split()
        return parts[-1] if parts else ""

    def match(self, other_name: str, score_cutoff: int = 85) -> Optional[Dict]:
        if not other_name:
            return None
        other_norm = self._normalize(other_name)
        other_last = self._extract_last_name(other_norm)
        # 1) Exact normalized full-name match
        exact = self.fpl_players[self.fpl_players["name_norm"] == other_norm]
        if not exact.empty:
            row = exact.iloc[0]
            return {"player_id": int(row["player_id"]), "name": str(row["name"]), "score": 100}
        # 2) Unique last-name match (e.g., "M.Salah" -> last_name "salah")
        if other_last and other_last in self.last_to_indices and len(self.last_to_indices[other_last]) == 1:
            idx = self.last_to_indices[other_last][0]
            row = self.fpl_players.iloc[idx]
            return {"player_id": int(row["player_id"]), "name": str(row["name"]), "score": 95}
        # 3) Fuzzy match against FPL display names
        candidate = process.extractOne(other_name, self.names, scorer=fuzz.WRatio, score_cutoff=score_cutoff)
        if not candidate:
            return None
        name, score, idx = candidate
        row = self.fpl_players.iloc[idx]
        return {"player_id": int(row["player_id"]), "name": str(row["name"]), "score": int(score)}

    def bulk_match(self, other_df: pd.DataFrame, other_name_col: str) -> pd.DataFrame:
        """Returns a DataFrame with columns: other_name, player_id, name, score."""
        out_rows = []
        for _, r in other_df.iterrows():
            hit = self.match(str(r.get(other_name_col, "")))
            if hit:
                hit["other_name"] = r.get(other_name_col)
                out_rows.append(hit)
        return pd.DataFrame(out_rows)

