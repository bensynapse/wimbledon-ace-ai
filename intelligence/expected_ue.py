"""
Expected UE model — contextual unforced error intelligence.

Not "UE high = bad", but "UE higher than expected for this matchup/context".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


# Tour-average baselines per surface (UE per point)
SURFACE_UE_BASELINE = {
    "hard": 0.19,
    "clay": 0.21,
    "grass": 0.18,
    "carpet": 0.19,
}

SURFACE_W_BASELINE = {
    "hard": 0.17,
    "clay": 0.16,
    "grass": 0.18,
    "carpet": 0.17,
}

# Contextual multipliers on expected UE
OPPONENT_ADJUSTMENTS = {
    "heavy_topspin": 0.12,
    "strong_returner": 0.08,
    "big_server": -0.04,
    "defensive_grinder": 0.06,
    "aggressive_net": 0.05,
}

CONTEXT_ADJUSTMENTS = {
    "fatigue_high": 0.10,
    "fatigue_medium": 0.05,
    "heat_high": 0.08,
    "heat_medium": 0.04,
    "wind_high": 0.06,
    "wind_medium": 0.03,
    "five_set_recent": 0.07,
    "pressure_set3plus": 0.05,
    "backhand_under_pressure": 0.06,
}


@dataclass
class PlayerShotProfile:
    """Baseline shot-quality profile for a player."""
    ue_per_point: float
    winners_per_point: float
    forced_errors_drawn: float = 0.0
    source: str = "estimated"  # official | charting | broadcast | estimated

    @property
    def w_ue_ratio(self) -> float:
        return self.winners_per_point / max(self.ue_per_point, 0.01)


@dataclass
class ExpectedUEResult:
    player: str
    surface: str
    baseline_ue_pp: float
    expected_ue_pp: float
    actual_ue_pp: Optional[float]
    baseline_w_pp: float
    expected_w_pp: float
    actual_w_pp: Optional[float]
    ue_delta_vs_expected: Optional[float]
    w_ue_ratio: Optional[float]
    ue_risk: str  # low | medium | high
    signals: list


class ExpectedUEModel:
    """Calculate expected UE/W given surface, opponent and physical context."""

    def analyze(
        self,
        player: str,
        surface: str,
        profile: Optional[PlayerShotProfile] = None,
        actual_ue_pp: Optional[float] = None,
        actual_w_pp: Optional[float] = None,
        opponent_tags: Optional[list] = None,
        context_flags: Optional[list] = None,
    ) -> ExpectedUEResult:
        surface = (surface or "hard").lower()
        baseline_ue = SURFACE_UE_BASELINE.get(surface, 0.19)
        baseline_w = SURFACE_W_BASELINE.get(surface, 0.17)

        if profile:
            baseline_ue = profile.ue_per_point
            baseline_w = profile.winners_per_point

        expected_ue = baseline_ue
        expected_w = baseline_w
        signals = []

        for tag in opponent_tags or []:
            adj = OPPONENT_ADJUSTMENTS.get(tag, 0.0)
            if adj:
                expected_ue += baseline_ue * adj
                signals.append(f"vs {tag}: UE expectation +{adj:.0%}")

        for flag in context_flags or []:
            adj = CONTEXT_ADJUSTMENTS.get(flag, 0.0)
            if adj:
                expected_ue += baseline_ue * adj
                signals.append(f"{flag}: UE expectation +{adj:.0%}")

        ue_delta = None
        ue_risk = "low"
        w_ue_ratio = None

        if actual_ue_pp is not None:
            ue_delta = actual_ue_pp - expected_ue
            pct_change = ue_delta / max(expected_ue, 0.01)
            w_ue_ratio = (actual_w_pp or expected_w) / max(actual_ue_pp, 0.01)

            if pct_change > 0.20 and (actual_w_pp or 0) <= expected_w:
                ue_risk = "high"
                signals.append(
                    f"UE {pct_change:+.0%} above expected without winner uplift"
                )
            elif pct_change > 0.10:
                ue_risk = "medium"
                signals.append(f"UE {pct_change:+.0%} above surface baseline")
            elif pct_change < -0.10:
                ue_risk = "low"
                signals.append("UE below expected — stable ball-striking")

        return ExpectedUEResult(
            player=player,
            surface=surface,
            baseline_ue_pp=round(baseline_ue, 3),
            expected_ue_pp=round(expected_ue, 3),
            actual_ue_pp=actual_ue_pp,
            baseline_w_pp=round(baseline_w, 3),
            expected_w_pp=round(expected_w, 3),
            actual_w_pp=actual_w_pp,
            ue_delta_vs_expected=round(ue_delta, 3) if ue_delta is not None else None,
            w_ue_ratio=round(w_ue_ratio, 2) if w_ue_ratio is not None else None,
            ue_risk=ue_risk,
            signals=signals,
        )