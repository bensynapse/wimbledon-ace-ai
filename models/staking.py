"""Stake sizing rules from master spec — conservative by design."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


EDGE_THRESHOLDS = {
    "no_bet": 0.03,
    "watchlist": 0.05,
    "small_value": 0.08,
    "value": 0.12,
}

STAKE_BY_CONFIDENCE = {
    "low": (0.0025, 0.0025),
    "medium": (0.005, 0.005),
    "high": (0.0075, 0.0075),
    "extreme": (0.01, 0.01),
}


@dataclass
class StakeRecommendation:
    action: str
    stake_low_pct: float
    stake_high_pct: float
    minimum_odds: float
    edge_label: str


class StakingRules:
    """Map edge + confidence → action and stake range."""

    def __init__(self, min_odds_buffer: float = 0.08):
        self.min_odds_buffer = min_odds_buffer

    def recommend(
        self,
        edge: float,
        fair_odds: float,
        market_odds: float,
        confidence: str,
        data_ok: bool = True,
    ) -> StakeRecommendation:
        min_odds = fair_odds * (1 + self.min_odds_buffer)
        edge_label = self._edge_label(edge)

        if not data_ok or edge < EDGE_THRESHOLDS["no_bet"]:
            return StakeRecommendation("NO BET", 0.0, 0.0, min_odds, edge_label)

        if market_odds > 1 and market_odds < min_odds:
            return StakeRecommendation("LIVE WAIT", 0.0, 0.0, min_odds, edge_label)

        if edge < EDGE_THRESHOLDS["watchlist"]:
            return StakeRecommendation("WATCHLIST", 0.0, 0.0, min_odds, edge_label)

        low, high = STAKE_BY_CONFIDENCE.get(confidence, STAKE_BY_CONFIDENCE["low"])
        if edge >= EDGE_THRESHOLDS["value"] and confidence == "high":
            low, high = STAKE_BY_CONFIDENCE["extreme"]

        if edge < EDGE_THRESHOLDS["small_value"]:
            action = "SMALL VALUE"
        elif edge < EDGE_THRESHOLDS["value"]:
            action = "VALUE"
        else:
            action = "STRONG VALUE" if confidence in ("high", "extreme") else "HIGH RISK VALUE"

        return StakeRecommendation(action, low, high, min_odds, edge_label)

    @staticmethod
    def _edge_label(edge: float) -> str:
        if edge < EDGE_THRESHOLDS["no_bet"]:
            return "no edge"
        if edge < EDGE_THRESHOLDS["watchlist"]:
            return "watchlist"
        if edge < EDGE_THRESHOLDS["small_value"]:
            return "small value"
        if edge < EDGE_THRESHOLDS["value"]:
            return "value"
        return "strong value"

    def format_stake_range(self, rec: StakeRecommendation) -> str:
        if rec.stake_low_pct <= 0:
            return "0%"
        return f"{rec.stake_low_pct * 100:.2f}%-{rec.stake_high_pct * 100:.2f}%"