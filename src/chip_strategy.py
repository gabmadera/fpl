from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd


@dataclass
class ChipRecommendation:
    chip_type: str
    chip_set: int
    gameweek: int
    expected_gain: float
    confidence: float
    reasoning: str


class ChipStrategyManager:
    def __init__(self) -> None:
        self.chips_available = {
            1: {"wildcard": True, "free_hit": True, "triple_captain": True, "bench_boost": True},
            2: {"wildcard": True, "free_hit": True, "triple_captain": True, "bench_boost": True},
        }

    def recommend_chips(self, predictions_df: Optional[pd.DataFrame], current_gw: int, fixture_analysis: Dict) -> List[ChipRecommendation]:
        if predictions_df is None or predictions_df.empty:
            return []
        chip_set = 1 if current_gw <= 19 else 2
        recs: List[ChipRecommendation] = []
        # Detect potential doubles/blanks (very basic): if many teams missing in next GW fixtures → blank; if teams have 2 fixtures → double
        try:
            if "fixtures" in fixture_analysis:
                fixtures = fixture_analysis["fixtures"]
                cur = fixtures[fixtures.get("event") == current_gw]
                team_counts = pd.concat([cur["team_h"], cur["team_a"]]).value_counts()
                is_blank = (team_counts < 1).sum() > 2
                has_doubles = (team_counts > 1).sum() > 2
                # Avoid recommending chips in opening GWs
                if current_gw <= 3:
                    has_doubles = False
                    is_blank = False
                if has_doubles and "predicted_points" in predictions_df.columns:
                    bench_points = float(predictions_df.nsmallest(4, "predicted_points")["predicted_points"].sum())
                    if bench_points >= 8.0:
                        recs.append(ChipRecommendation(
                            chip_type="bench_boost", chip_set=chip_set, gameweek=current_gw, expected_gain=bench_points, confidence=0.7,
                            reasoning="Likely double gameweek; bench projection strong"
                        ))
                if is_blank:
                    recs.append(ChipRecommendation(
                        chip_type="free_hit", chip_set=chip_set, gameweek=current_gw, expected_gain=8.0, confidence=0.6,
                        reasoning="Severe blank gameweek detected"
                    ))
        except Exception:
            pass
        # Triple captain: pick best expected points (avoid very early weeks)
        if current_gw >= 3 and "predicted_points" in predictions_df.columns:
            top = predictions_df.nlargest(1, "predicted_points")
            if not top.empty:
                player = top.iloc[0]
                gain = float(player.get("predicted_points", 6.0))
                if gain >= 8.0:  # only if strong projection
                    recs.append(
                        ChipRecommendation(
                            chip_type="triple_captain",
                            chip_set=chip_set,
                            gameweek=current_gw,
                            expected_gain=gain,
                            confidence=0.6,
                            reasoning=f"{player.get('name', 'Top Player')} has a very high projection",
                        )
                    )
        # Bench boost: remove unconditional early suggestion; rely on doubles logic above
        return sorted(recs, key=lambda r: r.expected_gain, reverse=True)

