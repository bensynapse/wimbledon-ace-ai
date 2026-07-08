"""
WimbledonAce AI — Feature Engineering
Elo ratings, surface form, H2H & Grand Slam match vectors.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import MATCH_HISTORY_WINDOW, OVERALL_ELO_WEIGHT, SURFACE_ELO_WEIGHT

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "elo_diff",
    "surface_elo_diff",
    "form_win_rate_diff",
    "surface_form_diff",
    "h2h_win_rate_diff",
    "days_since_last_match_diff",
    "rank_diff",
    "combined_strength_diff",
]

K_FACTOR = 32
DEFAULT_ELO = 1500.0


class TennisFeatureEngine:
    """Build chronological features and maintain player state."""

    def __init__(self, history_window: int = MATCH_HISTORY_WINDOW):
        self.window = history_window
        self.elo: Dict[str, float] = defaultdict(lambda: DEFAULT_ELO)
        self.surface_elo: Dict[Tuple[str, str], float] = defaultdict(lambda: DEFAULT_ELO)
        self.matches: Dict[str, List[Dict]] = defaultdict(list)
        self.h2h: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        self.last_match_date: Dict[str, datetime] = {}
        self.rankings: Dict[str, int] = {}

    def set_rankings(self, standings: List[Dict]) -> None:
        for row in standings:
            player = row.get("player", "").strip()
            try:
                self.rankings[player] = int(row.get("place", 999))
            except (TypeError, ValueError):
                continue

    def build_training_matrix(
        self,
        df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        df = df.sort_values("date").reset_index(drop=True)
        features: List[List[float]] = []
        labels: List[int] = []

        for _, row in df.iterrows():
            feat, ctx = self._match_features(
                player1=row["player1"],
                player2=row["player2"],
                surface=row.get("surface", "hard"),
                match_date=row["date"],
            )
            features.append([feat[name] for name in FEATURE_NAMES])
            labels.append(1 if row["winner_is_player1"] else 0)

            self._update_state(
                player1=row["player1"],
                player2=row["player2"],
                surface=row.get("surface", "hard"),
                winner_is_player1=row["winner_is_player1"],
                match_date=row["date"],
            )

        X = np.array(features, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)
        return X, y, FEATURE_NAMES

    def build_match_features(
        self,
        player1: str,
        player2: str,
        surface: str = "hard",
        match_date: Optional[str] = None,
    ) -> Tuple[List[float], Dict[str, float]]:
        feat, ctx = self._match_features(player1, player2, surface, match_date)
        return [feat[name] for name in FEATURE_NAMES], ctx

    def ingest_history(self, df: pd.DataFrame) -> None:
        """Replay historical matches to warm up player state."""
        for _, row in df.sort_values("date").iterrows():
            self._update_state(
                player1=row["player1"],
                player2=row["player2"],
                surface=row.get("surface", "hard"),
                winner_is_player1=row["winner_is_player1"],
                match_date=row["date"],
            )

    def _match_features(
        self,
        player1: str,
        player2: str,
        surface: str,
        match_date: Optional[str],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        surface = (surface or "hard").lower()
        p1_elo = self.elo[player1]
        p2_elo = self.elo[player2]
        p1_surface = self.surface_elo[(player1, surface)]
        p2_surface = self.surface_elo[(player2, surface)]

        p1_form = self._recent_win_rate(player1)
        p2_form = self._recent_win_rate(player2)
        p1_surface_form = self._recent_win_rate(player1, surface=surface)
        p2_surface_form = self._recent_win_rate(player2, surface=surface)
        h2h_diff = self._h2h_diff(player1, player2)
        rest_diff = self._rest_diff(player1, player2, match_date)
        rank_diff = self._rank_diff(player1, player2)

        p1_combined = (
            OVERALL_ELO_WEIGHT * p1_elo + SURFACE_ELO_WEIGHT * p1_surface
        )
        p2_combined = (
            OVERALL_ELO_WEIGHT * p2_elo + SURFACE_ELO_WEIGHT * p2_surface
        )

        feat = {
            "elo_diff": p1_elo - p2_elo,
            "surface_elo_diff": p1_surface - p2_surface,
            "form_win_rate_diff": p1_form - p2_form,
            "surface_form_diff": p1_surface_form - p2_surface_form,
            "h2h_win_rate_diff": h2h_diff,
            "days_since_last_match_diff": rest_diff,
            "rank_diff": rank_diff,
            "combined_strength_diff": p1_combined - p2_combined,
        }
        ctx = {
            "elo_diff": feat["elo_diff"],
            "surface_elo_diff": feat["surface_elo_diff"],
            "h2h_win_rate_diff": h2h_diff,
            "days_since_last_match_diff": rest_diff,
        }
        return feat, ctx

    def _update_state(
        self,
        player1: str,
        player2: str,
        surface: str,
        winner_is_player1: bool,
        match_date: str,
    ) -> None:
        surface = (surface or "hard").lower()
        p1_elo = self.elo[player1]
        p2_elo = self.elo[player2]
        p1_surface = self.surface_elo[(player1, surface)]
        p2_surface = self.surface_elo[(player2, surface)]

        expected_p1 = _expected_score(p1_elo, p2_elo)
        expected_p1_surface = _expected_score(p1_surface, p2_surface)
        actual = 1.0 if winner_is_player1 else 0.0

        self.elo[player1] = p1_elo + K_FACTOR * (actual - expected_p1)
        self.elo[player2] = p2_elo + K_FACTOR * ((1 - actual) - (1 - expected_p1))
        self.surface_elo[(player1, surface)] = (
            p1_surface + K_FACTOR * (actual - expected_p1_surface)
        )
        self.surface_elo[(player2, surface)] = (
            p2_surface + K_FACTOR * ((1 - actual) - (1 - expected_p1_surface))
        )

        self.matches[player1].append({"won": winner_is_player1, "surface": surface, "date": match_date})
        self.matches[player2].append({"won": not winner_is_player1, "surface": surface, "date": match_date})

        first, second = sorted([player1, player2])
        winner = player1 if winner_is_player1 else player2
        self.h2h[(first, second)].append(1 if winner == first else 0)

        parsed = _parse_date(match_date)
        if parsed:
            self.last_match_date[player1] = parsed
            self.last_match_date[player2] = parsed

    def _recent_win_rate(self, player: str, surface: Optional[str] = None) -> float:
        history = self.matches[player]
        if surface:
            history = [m for m in history if m["surface"] == surface]
        recent = history[-self.window:]
        if not recent:
            return 0.5
        return sum(1 for m in recent if m["won"]) / len(recent)

    def _h2h_diff(self, player1: str, player2: str) -> float:
        first, second = sorted([player1, player2])
        history = self.h2h[(first, second)]
        if not history:
            return 0.0
        first_win_rate = sum(history) / len(history)
        p1_rate = first_win_rate if player1 == first else 1 - first_win_rate
        return p1_rate - 0.5

    def _rest_diff(
        self,
        player1: str,
        player2: str,
        match_date: Optional[str],
    ) -> float:
        current = _parse_date(match_date)
        if not current:
            return 0.0

        def days_since(player: str) -> float:
            last = self.last_match_date.get(player)
            if not last:
                return 14.0
            return max((current - last).days, 0)

        return days_since(player1) - days_since(player2)

    def _rank_diff(self, player1: str, player2: str) -> float:
        r1 = self.rankings.get(player1, 100)
        r2 = self.rankings.get(player2, 100)
        return r2 - r1


def _expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def _parse_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(str(value)[:10], fmt)
        except ValueError:
            continue
    return None