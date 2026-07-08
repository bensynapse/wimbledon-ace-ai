"""
Rank/points probability model for backtesting — no odds leakage.

Mirrors the notebook feature engineering (Rank_Diff, Pts_Diff) without using Odds_Diff
in the model, so we can detect market mispricing vs historical bookmaker lines.
"""

from __future__ import annotations

import math
from typing import Tuple


def implied_probability(odd1: float, odd2: float) -> Tuple[float, float]:
    """Vig-stripped implied win probability from decimal odds."""
    if odd1 <= 1 or odd2 <= 1:
        return 0.5, 0.5
    raw1, raw2 = 1.0 / odd1, 1.0 / odd2
    total = raw1 + raw2
    return raw1 / total, raw2 / total


def rank_points_probability(
    rank1: float,
    rank2: float,
    pts1: float,
    pts2: float,
    rank_weight: float = 0.035,
    pts_weight: float = 0.00035,
) -> float:
    """
    Logistic win probability for player 1 from rank and ranking points.

    Positive rank_diff (rank2 > rank1) means player 1 is better ranked.
    """
    rank_diff = rank2 - rank1
    pts_diff = pts1 - pts2
    logit = rank_weight * rank_diff + pts_weight * pts_diff
    logit = max(-6.0, min(6.0, logit))
    return 1.0 / (1.0 + math.exp(-logit))


def market_edge(model_prob: float, market_prob: float) -> float:
    """Edge = model probability minus market implied probability."""
    return model_prob - market_prob