"""
Historical odds backtest — model probability vs bookmaker lines.

Uses dissfya/atp-tennis-2000-2023daily-pull (Odd_1, Odd_2) to simulate
value-bet detection without lookahead: rank/points model only, no Odds_Diff.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from config import BANKROLL, MIN_VALUE_THRESHOLD, PATHS
from data_sources.kaggle_odds import KaggleOddsSource
from database.db import TennisDatabase
from models.rank_probability import implied_probability, market_edge, rank_points_probability
from models.staking import StakingRules

logger = logging.getLogger(__name__)


@dataclass
class BacktestBet:
    date: str
    match: str
    pick: str
    odds: float
    model_prob: float
    market_prob: float
    edge: float
    stake_pct: float
    won: bool
    profit: float
    surface: str
    tournament: str


@dataclass
class BacktestResult:
    start_year: int
    end_year: int
    surface: Optional[str]
    min_edge: float
    matches_scanned: int
    bets_placed: int
    wins: int
    losses: int
    hit_rate: float
    total_staked: float
    total_profit: float
    roi_pct: float
    avg_edge: float
    model_accuracy: float
    market_favorite_accuracy: float
    bets: List[BacktestBet] = field(default_factory=list)

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["bets"] = [asdict(b) for b in self.bets[:50]]
        return data


class OddsBacktester:
    """Walk historical ATP matches with bookmaker odds and simulate value bets."""

    def __init__(
        self,
        min_edge: float = MIN_VALUE_THRESHOLD,
        bankroll: float = BANKROLL,
        output_dir: Optional[str] = None,
    ):
        self.min_edge = min_edge
        self.bankroll = bankroll
        self.output_dir = Path(output_dir or PATHS["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.odds_source = KaggleOddsSource()
        self.staking = StakingRules()
        self.db = TennisDatabase()

    def run(
        self,
        start_year: int = 2018,
        end_year: Optional[int] = None,
        surface: Optional[str] = None,
        max_bets: Optional[int] = None,
        log_to_db: bool = True,
    ) -> BacktestResult:
        end_year = end_year or datetime.now().year
        df = self.odds_source.load(start_year=start_year, end_year=end_year, surface=surface)

        bets: List[BacktestBet] = []
        model_correct = 0
        market_correct = 0
        total_staked = 0.0
        total_profit = 0.0

        for _, row in df.iterrows():
            prob_p1 = rank_points_probability(
                row["rank_player1"], row["rank_player2"],
                row["pts_player1"], row["pts_player2"],
            )
            prob_p2 = 1.0 - prob_p1
            mkt_p1, mkt_p2 = implied_probability(row["odd_player1"], row["odd_player2"])

            model_pick_p1 = prob_p1 >= 0.5
            market_pick_p1 = mkt_p1 >= mkt_p2
            actual_p1 = bool(row["winner_is_player1"])

            if model_pick_p1 == actual_p1:
                model_correct += 1
            if market_pick_p1 == actual_p1:
                market_correct += 1

            edge_p1 = market_edge(prob_p1, mkt_p1)
            edge_p2 = market_edge(prob_p2, mkt_p2)

            pick_side = None
            if edge_p1 >= self.min_edge and edge_p1 >= edge_p2:
                pick_side = "p1"
            elif edge_p2 >= self.min_edge:
                pick_side = "p2"

            if pick_side is None:
                continue

            if pick_side == "p1":
                pick_name = row["player1"]
                odds = row["odd_player1"]
                model_prob = prob_p1
                market_prob = mkt_p1
                edge = edge_p1
                won = actual_p1
            else:
                pick_name = row["player2"]
                odds = row["odd_player2"]
                model_prob = prob_p2
                market_prob = mkt_p2
                edge = edge_p2
                won = not actual_p1

            stake_rec = self.staking.recommend(
                edge=edge,
                fair_odds=1.0 / model_prob if model_prob > 0 else 99,
                market_odds=odds,
                confidence="medium",
                data_ok=True,
            )
            if stake_rec.action in ("NO BET", "LIVE WAIT", "WATCHLIST"):
                continue

            stake_pct = stake_rec.stake_high_pct or stake_rec.stake_low_pct
            if stake_pct <= 0:
                stake_pct = 0.005

            stake_amount = self.bankroll * stake_pct
            profit = stake_amount * (odds - 1) if won else -stake_amount
            total_staked += stake_amount
            total_profit += profit

            bet = BacktestBet(
                date=str(row["date"])[:10],
                match=f"{row['player1']} vs {row['player2']}",
                pick=pick_name,
                odds=round(odds, 2),
                model_prob=round(model_prob, 4),
                market_prob=round(market_prob, 4),
                edge=round(edge, 4),
                stake_pct=round(stake_pct, 4),
                won=won,
                profit=round(profit, 2),
                surface=row["surface"],
                tournament=str(row.get("tournament", "")),
            )
            bets.append(bet)

            if log_to_db:
                self.db.log_bet(
                    date=bet.date,
                    match=bet.match,
                    pick=bet.pick,
                    odds_taken=bet.odds,
                    stake_percent=bet.stake_pct * 100,
                    reason=f"backtest edge {bet.edge:+.1%}",
                    model_version="rank_pts_v1",
                    result="win" if won else "loss",
                    profit_loss=bet.profit,
                    clv=0.0,
                )

            if max_bets and len(bets) >= max_bets:
                break

        n = len(df)
        wins = sum(1 for b in bets if b.won)
        losses = len(bets) - wins
        roi = (total_profit / total_staked * 100) if total_staked > 0 else 0.0
        avg_edge = sum(b.edge for b in bets) / len(bets) if bets else 0.0

        result = BacktestResult(
            start_year=start_year,
            end_year=end_year,
            surface=surface,
            min_edge=self.min_edge,
            matches_scanned=n,
            bets_placed=len(bets),
            wins=wins,
            losses=losses,
            hit_rate=wins / len(bets) if bets else 0.0,
            total_staked=round(total_staked, 2),
            total_profit=round(total_profit, 2),
            roi_pct=round(roi, 2),
            avg_edge=round(avg_edge, 4),
            model_accuracy=round(model_correct / n, 4) if n else 0.0,
            market_favorite_accuracy=round(market_correct / n, 4) if n else 0.0,
            bets=bets,
        )

        self._save_report(result)
        return result

    def _save_report(self, result: BacktestResult) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"backtest_{result.start_year}_{result.end_year}_{stamp}.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(result.to_dict(), handle, indent=2)
        logger.info("Backtest report saved to %s", path)
        return path


def format_backtest_report(result: BacktestResult) -> str:
    lines = [
        "",
        "═══ WimbledonAce AI — Historical Odds Backtest ═══",
        f"Period:     {result.start_year}–{result.end_year}"
        + (f" ({result.surface})" if result.surface else ""),
        f"Min edge:   {result.min_edge:.1%}",
        f"Matches:    {result.matches_scanned:,}",
        "",
        "── Model vs Market ──",
        f"Rank/pts model accuracy:  {result.model_accuracy:.1%}",
        f"Market favorite accuracy: {result.market_favorite_accuracy:.1%}",
        "",
        "── Value Bets Simulated ──",
        f"Bets placed:  {result.bets_placed:,}",
        f"Hit rate:     {result.hit_rate:.1%} ({result.wins}W / {result.losses}L)",
        f"Avg edge:     {result.avg_edge:+.1%}",
        f"Total staked: €{result.total_staked:,.2f}",
        f"Profit/Loss:  €{result.total_profit:+,.2f}",
        f"ROI:          {result.roi_pct:+.2f}%",
        "",
    ]
    if result.bets:
        lines.append("── Sample bets (first 5) ──")
        for bet in result.bets[:5]:
            icon = "✓" if bet.won else "✗"
            lines.append(
                f"  {icon} {bet.date} | {bet.match} → {bet.pick} @ {bet.odds:.2f} "
                f"edge {bet.edge:+.1%} | €{bet.profit:+.2f}"
            )
    lines.append("")
    return "\n".join(lines)