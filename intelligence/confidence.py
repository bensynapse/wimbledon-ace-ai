"""Data confidence scoring — never treat all inputs equally."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MISSING = "missing"


@dataclass
class DataConfidence:
    """Per-source confidence for a match analysis."""
    elo: ConfidenceLevel = ConfidenceLevel.HIGH
    odds: ConfidenceLevel = ConfidenceLevel.MISSING
    ue_stats: ConfidenceLevel = ConfidenceLevel.MISSING
    fatigue: ConfidenceLevel = ConfidenceLevel.MEDIUM
    weather: ConfidenceLevel = ConfidenceLevel.MISSING
    injury: ConfidenceLevel = ConfidenceLevel.MISSING
    serve_return: ConfidenceLevel = ConfidenceLevel.MISSING

    def to_dict(self) -> Dict[str, str]:
        return {k: v.value for k, v in self.__dict__.items()}

    def overall_score(self) -> float:
        weights = {
            ConfidenceLevel.HIGH: 1.0,
            ConfidenceLevel.MEDIUM: 0.65,
            ConfidenceLevel.LOW: 0.35,
            ConfidenceLevel.MISSING: 0.0,
        }
        fields = list(self.__dict__.values())
        if not fields:
            return 0.0
        return sum(weights[f] for f in fields) / len(fields)


def score_ue_confidence(source: Optional[str] = None) -> ConfidenceLevel:
    mapping = {
        "official": ConfidenceLevel.HIGH,
        "charting": ConfidenceLevel.MEDIUM,
        "broadcast": ConfidenceLevel.LOW,
    }
    if not source:
        return ConfidenceLevel.MISSING
    return mapping.get(source.lower(), ConfidenceLevel.LOW)