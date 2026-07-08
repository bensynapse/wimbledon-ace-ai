"""Fatigue and heat collapse model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class FatigueInput:
    minutes_last_7_days: float = 0.0
    sets_last_7_days: float = 0.0
    matches_last_7_days: float = 0.0
    rest_days: float = 2.0
    five_set_recent: bool = False
    medical_timeout_recent: bool = False
    retirement_recent: bool = False
    age: float = 27.0
    temperature_c: Optional[float] = None
    humidity_pct: Optional[float] = None
    wind_kmh: Optional[float] = None


@dataclass
class FatigueResult:
    player: str
    fatigue_score: float  # 0-1
    heat_score: float  # 0-1
    collapse_risk: str  # low | medium | high
    signals: List[str]


class FatigueModel:
    """Estimate physical collapse risk for long matches / hot conditions."""

    def analyze(self, player: str, data: FatigueInput) -> FatigueResult:
        score = 0.0
        signals = []

        if data.minutes_last_7_days > 600:
            score += 0.25
            signals.append(f"{data.minutes_last_7_days:.0f} min on court last 7 days")
        elif data.minutes_last_7_days > 400:
            score += 0.15

        if data.sets_last_7_days >= 12:
            score += 0.15
            signals.append(f"{data.sets_last_7_days:.0f} sets played last 7 days")

        if data.rest_days < 1:
            score += 0.20
            signals.append("Less than 1 rest day")
        elif data.rest_days < 2:
            score += 0.10

        if data.five_set_recent:
            score += 0.15
            signals.append("Five-setter in previous round")

        if data.medical_timeout_recent:
            score += 0.12
            signals.append("Medical timeout recently")

        if data.retirement_recent:
            score += 0.10

        if data.age >= 32:
            score += 0.08
            signals.append(f"Age {data.age:.0f} — slower recovery baseline")

        heat = 0.0
        if data.temperature_c is not None:
            if data.temperature_c >= 33:
                heat = 0.85
                signals.append(f"Extreme heat {data.temperature_c:.0f}°C")
            elif data.temperature_c >= 28:
                heat = 0.55
                signals.append(f"High temperature {data.temperature_c:.0f}°C")
            elif data.temperature_c >= 24:
                heat = 0.25

        if data.humidity_pct and data.humidity_pct >= 70 and heat > 0:
            heat = min(1.0, heat + 0.15)
            signals.append(f"High humidity {data.humidity_pct:.0f}%")

        if data.wind_kmh and data.wind_kmh >= 25:
            heat = min(1.0, heat + 0.10)
            signals.append(f"Wind {data.wind_kmh:.0f} km/h")

        combined = min(1.0, score + heat * 0.35)
        if combined >= 0.55:
            risk = "high"
        elif combined >= 0.30:
            risk = "medium"
        else:
            risk = "low"

        return FatigueResult(
            player=player,
            fatigue_score=round(score, 2),
            heat_score=round(heat, 2),
            collapse_risk=risk,
            signals=signals,
        )