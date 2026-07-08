"""
WimbledonAce Elo tracker — eigen ratinglijst met echte Elo-logica.

Formule (standaard Elo):
  expected = 1 / (1 + 10^((opponent_elo - player_elo) / 400))
  delta    = K * (actual - expected)   # actual=1 bij win, 0 bij verlies

K-factor wordt geschaald op toernooi-niveau, ronde en best-of.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from config import PATHS
from feature_engineering import _expected_score

logger = logging.getLogger(__name__)

DEFAULT_ELO = 1500.0
BASE_K = 32.0

TOURNAMENT_K = {
    "grand slam": 1.0,
    "masters": 0.85,
    "atp 500": 0.75,
    "atp 250": 0.70,
    "challenger": 0.55,
    "default": 0.70,
}

ROUND_K = {
    "f": 1.0,
    "final": 1.0,
    "sf": 0.9,
    "semi": 0.9,
    "qf": 0.85,
    "quarter": 0.85,
    "r16": 0.8,
    "r32": 0.75,
    "r64": 0.75,
    "r128": 0.7,
    "default": 0.8,
}

SURFACES = ("hard", "clay", "grass", "carpet")


@dataclass
class EloExchange:
    """Puntenwissel als één match gespeeld wordt."""
    player_a: str
    player_b: str
    surface: str
    elo_a: float
    elo_b: float
    surface_elo_a: float
    surface_elo_b: float
    win_prob_a: float
    win_prob_b: float
    k_factor: float
    if_a_wins_delta_a: float
    if_a_wins_delta_b: float
    if_b_wins_delta_a: float
    if_b_wins_delta_b: float

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PlayerEloRow:
    player: str
    overall_elo: float
    hard_elo: float
    clay_elo: float
    grass_elo: float
    carpet_elo: float
    matches: int
    wins: int
    last_match: str
    elo_rank: int = 0

    def surface_elo(self, surface: str) -> float:
        return getattr(self, f"{surface}_elo", self.overall_elo)


class WimbledonAceEloTracker:
    """Bouw en onderhoud eigen Elo-ratings uit matchgeschiedenis."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or f"{PATHS['data_dir']}wimbledon_ace_elo/")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.overall: Dict[str, float] = {}
        self.surface: Dict[Tuple[str, str], float] = {}
        self.matches: Dict[str, int] = {}
        self.wins: Dict[str, int] = {}
        self.last_match: Dict[str, str] = {}

    def rebuild_from_history(self, df: pd.DataFrame, tour: str = "atp") -> pd.DataFrame:
        """Replay alle wedstrijden chronologisch en herbereken Elo."""
        self.overall.clear()
        self.surface.clear()
        self.matches.clear()
        self.wins.clear()
        self.last_match.clear()

        ordered = df.sort_values("date").reset_index(drop=True)
        for _, row in ordered.iterrows():
            self._apply_result(
                player1=row["player1"],
                player2=row["player2"],
                winner_is_p1=bool(row["winner_is_player1"]),
                surface=str(row.get("surface", "hard")).lower(),
                tournament=str(row.get("tournament", "")),
                round_name=str(row.get("round", "")),
                best_of=int(row.get("best_of", 3) or 3),
                date=str(row.get("date", ""))[:10],
            )

        leaderboard = self.leaderboard(top_n=len(self.overall))
        out = self.cache_dir / f"{tour}_ratings.parquet"
        leaderboard.to_parquet(out, index=False)
        meta = {
            "tour": tour,
            "updated_at": datetime.now().isoformat(),
            "players": len(leaderboard),
            "matches_replayed": len(ordered),
        }
        (self.cache_dir / f"{tour}_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        logger.info("WimbledonAce Elo rebuilt: %d players, %d matches", len(leaderboard), len(ordered))
        return leaderboard

    def load(self, tour: str = "atp") -> pd.DataFrame:
        path = self.cache_dir / f"{tour}_ratings.parquet"
        if not path.exists():
            raise FileNotFoundError(f"No Elo ratings at {path}. Run: python3 cli.py elo rebuild")
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            player = row["player"]
            self.overall[player] = float(row["overall_elo"])
            self.matches[player] = int(row.get("matches", 0))
            self.wins[player] = int(row.get("wins", 0))
            self.last_match[player] = str(row.get("last_match", ""))
            for surf in SURFACES:
                col = f"{surf}_elo"
                if col in row and pd.notna(row[col]):
                    self.surface[(player, surf)] = float(row[col])
        return df

    def leaderboard(self, surface: Optional[str] = None, top_n: int = 100) -> pd.DataFrame:
        rows: List[PlayerEloRow] = []
        players = set(self.overall.keys())
        for player in players:
            rows.append(PlayerEloRow(
                player=player,
                overall_elo=round(self.overall.get(player, DEFAULT_ELO), 1),
                hard_elo=round(self.surface.get((player, "hard"), DEFAULT_ELO), 1),
                clay_elo=round(self.surface.get((player, "clay"), DEFAULT_ELO), 1),
                grass_elo=round(self.surface.get((player, "grass"), DEFAULT_ELO), 1),
                carpet_elo=round(self.surface.get((player, "carpet"), DEFAULT_ELO), 1),
                matches=int(self.matches.get(player, 0)),
                wins=int(self.wins.get(player, 0)),
                last_match=self.last_match.get(player, ""),
            ))

        rows.sort(
            key=lambda r: r.surface_elo(surface) if surface else r.overall_elo,
            reverse=True,
        )
        for i, row in enumerate(rows[:top_n], start=1):
            row.elo_rank = i
        data = [asdict(r) for r in rows[:top_n]]
        return pd.DataFrame(data)

    def compute_exchange(
        self,
        player_a: str,
        player_b: str,
        surface: str = "hard",
        tournament: str = "",
        round_name: str = "",
        best_of: int = 3,
        k_override: Optional[float] = None,
    ) -> EloExchange:
        """Bereken hoeveel Elo-punten beide spelers winnen/verliezen bij winst of verlies."""
        surface = (surface or "hard").lower()
        elo_a = self._rating(player_a)
        elo_b = self._rating(player_b)
        surf_a = self._surface_rating(player_a, surface)
        surf_b = self._surface_rating(player_b, surface)

        k = k_override or self._match_k(tournament, round_name, best_of)
        exp_a = _expected_score(surf_a, surf_b)
        exp_b = 1.0 - exp_a

        d_a_win = k * (1.0 - exp_a)
        d_b_loss = k * (0.0 - (1.0 - exp_b))
        d_a_loss = k * (0.0 - exp_a)
        d_b_win = k * (1.0 - exp_b)

        return EloExchange(
            player_a=player_a,
            player_b=player_b,
            surface=surface,
            elo_a=round(elo_a, 1),
            elo_b=round(elo_b, 1),
            surface_elo_a=round(surf_a, 1),
            surface_elo_b=round(surf_b, 1),
            win_prob_a=round(exp_a, 4),
            win_prob_b=round(exp_b, 4),
            k_factor=round(k, 2),
            if_a_wins_delta_a=round(d_a_win, 2),
            if_a_wins_delta_b=round(d_b_loss, 2),
            if_b_wins_delta_a=round(d_a_loss, 2),
            if_b_wins_delta_b=round(d_b_win, 2),
        )

    def export_csv(self, tour: str = "atp", surface: Optional[str] = None) -> Path:
        df = self.load(tour)
        if surface:
            df = df.sort_values(f"{surface}_elo", ascending=False)
        else:
            df = df.sort_values("overall_elo", ascending=False)
        df["elo_rank"] = range(1, len(df) + 1)
        out = self.cache_dir / f"{tour}_elo_list.csv"
        df.to_csv(out, index=False)
        return out

    def _apply_result(
        self,
        player1: str,
        player2: str,
        winner_is_p1: bool,
        surface: str,
        tournament: str,
        round_name: str,
        best_of: int,
        date: str,
    ) -> None:
        k = self._match_k(tournament, round_name, best_of)
        for player in (player1, player2):
            self._ensure_player(player)

        p1_o, p2_o = self.overall[player1], self.overall[player2]
        p1_s, p2_s = self.surface[(player1, surface)], self.surface[(player2, surface)]
        exp_p1 = _expected_score(p1_s, p2_s)
        actual = 1.0 if winner_is_p1 else 0.0

        self.overall[player1] = p1_o + k * (actual - _expected_score(p1_o, p2_o))
        self.overall[player2] = p2_o + k * ((1 - actual) - _expected_score(p2_o, p1_o))
        self.surface[(player1, surface)] = p1_s + k * (actual - exp_p1)
        self.surface[(player2, surface)] = p2_s + k * ((1 - actual) - (1 - exp_p1))

        for player, won in ((player1, winner_is_p1), (player2, not winner_is_p1)):
            self.matches[player] += 1
            if won:
                self.wins[player] += 1
            self.last_match[player] = date

    def _match_k(self, tournament: str, round_name: str, best_of: int) -> float:
        t_key = tournament.lower()
        t_mult = TOURNAMENT_K["default"]
        for key, mult in TOURNAMENT_K.items():
            if key in t_key:
                t_mult = mult
                break

        r_key = round_name.lower()
        r_mult = ROUND_K["default"]
        for key, mult in ROUND_K.items():
            if key in r_key:
                r_mult = mult
                break

        bo_mult = 1.0 if best_of >= 5 else 0.9
        return BASE_K * t_mult * r_mult * bo_mult

    def _ensure_player(self, player: str) -> None:
        if player not in self.overall:
            self.overall[player] = DEFAULT_ELO
            self.matches[player] = 0
            self.wins[player] = 0
            for surf in SURFACES:
                self.surface[(player, surf)] = DEFAULT_ELO

    def _rating(self, player: str) -> float:
        self._ensure_player(player)
        return self.overall[player]

    def _surface_rating(self, player: str, surface: str) -> float:
        self._ensure_player(player)
        return self.surface.get((player, surface), DEFAULT_ELO)