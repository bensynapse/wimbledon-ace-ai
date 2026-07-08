"""
Underdog probability adjustments — fixes ranking/Elo overweight.

Triggers (from post-mortems e.g. Fery @ Wimbledon 2026):
- Grand Slam momentum (wins vs higher-ranked opponents this event)
- Home wildcard at Wimbledon
- Recent H2H upset
- Surface-form gap smaller than Elo gap suggests
- Deep GS run from low ranking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


MAX_FAVORITE_PROB = 0.68
MAX_SINGLE_BOOST = 0.12


@dataclass
class UnderdogAdjustment:
    base_prob_p1: float
    adjusted_prob_p1: float
    boost_p1: float
    boost_p2: float
    signals: List[str] = field(default_factory=list)

    @property
    def applied(self) -> bool:
        return abs(self.boost_p1) > 0.005 or abs(self.boost_p2) > 0.005


def apply_underdog_adjustments(
    prob_p1: float,
    player1: str,
    player2: str,
    ctx: Dict,
) -> UnderdogAdjustment:
    """Shift probability toward underdog when contextual upset signals fire."""
    boost_p1 = _underdog_boost(player1, player2, ctx, is_player1=True)
    boost_p2 = _underdog_boost(player2, player1, ctx, is_player1=False)

    adjusted = prob_p1 + boost_p1 - boost_p2
    adjusted = max(0.05, min(0.95, adjusted))

    favorite_is_p1 = adjusted >= 0.5
    if favorite_is_p1 and adjusted > MAX_FAVORITE_PROB:
        adjusted = MAX_FAVORITE_PROB
    elif not favorite_is_p1 and adjusted < (1.0 - MAX_FAVORITE_PROB):
        adjusted = 1.0 - MAX_FAVORITE_PROB

    signals = list(ctx.get("underdog_signals", []))
    return UnderdogAdjustment(
        base_prob_p1=round(prob_p1, 4),
        adjusted_prob_p1=round(adjusted, 4),
        boost_p1=round(boost_p1, 4),
        boost_p2=round(boost_p2, 4),
        signals=signals,
    )


def _underdog_boost(
    player: str,
    opponent: str,
    ctx: Dict,
    is_player1: bool,
) -> float:
    prefix = "player1" if is_player1 else "player2"
    opp_prefix = "player2" if is_player1 else "player1"

    player_elo = ctx.get(f"{prefix}_elo", 1500)
    opp_elo = ctx.get(f"{opp_prefix}_elo", 1500)
    if player_elo >= opp_elo:
        return 0.0

    elo_gap = opp_elo - player_elo
    boost = 0.0
    signals: List[str] = ctx.setdefault("underdog_signals", [])

    def add(amount: float, msg: str) -> None:
        nonlocal boost
        capped = min(amount, MAX_SINGLE_BOOST - boost)
        if capped <= 0:
            return
        boost += capped
        label = f"{player}: {msg} (+{capped:.0%})"
        if label not in signals:
            signals.append(label)

    tournament = str(ctx.get("tournament", "")).lower()
    surface = str(ctx.get("surface", "")).lower()

    # Grand Slam momentum — beat higher-ranked players this event
    tm_wins = int(ctx.get(f"{prefix}_tournament_upset_wins", 0) or 0)
    if tm_wins >= 1:
        add(min(0.04 * tm_wins, 0.10), f"{tm_wins} upset win(s) this tournament")

    # Deep run from low rank (Ivanisevic pattern)
    rank = ctx.get(f"{prefix}_rank")
    round_name = str(ctx.get("round", "")).lower()
    in_late_round = any(r in round_name for r in ("qf", "quarter", "sf", "semi", "f"))
    if rank and int(rank) > 100 and in_late_round:
        add(0.06, f"deep GS run from rank #{int(rank)}")

    # Home player / wildcard at Wimbledon
    if ctx.get(f"{prefix}_home_slam"):
        label = "home wildcard at Wimbledon" if ctx.get(f"{prefix}_is_wildcard") else "home player at Wimbledon"
        add(0.08, label)

    # Recent H2H win as underdog
    h2h = ctx.get("h2h_recent_winner")
    if h2h and _name_match(h2h, player):
        add(0.07, "won last head-to-head")

    # Surface form nearly matches favorite despite Elo gap
    p_surf_form = ctx.get(f"{prefix}_surface_form", 0.5) or 0.5
    o_surf_form = ctx.get(f"{opp_prefix}_surface_form", 0.5) or 0.5
    if elo_gap > 80 and (p_surf_form - o_surf_form) >= -0.10:
        add(0.04, "grass/surface form keeps pace with favorite")

    # Aggressive net profile (charting W/UE ratio)
    w_ue = ctx.get(f"{prefix}_w_ue_ratio")
    if w_ue and float(w_ue) >= 1.15 and surface == "grass":
        add(0.03, "aggressive grass profile (high W/UE)")

    # Opponent fatigue / injury flag from context
    if ctx.get(f"{opp_prefix}_injury_flag") or "injury" in str(ctx.get(f"{opp_prefix}_context_flags", [])):
        add(0.05, "opponent fitness concern")

    # Scale boost by elo gap — bigger gaps need more evidence, cap total
    if elo_gap < 50:
        boost *= 0.5
    elif elo_gap > 200:
        boost = min(boost * 1.15, 0.18)

    return min(boost, 0.18)


def _name_match(a: str, b: str) -> bool:
    return a.lower().split()[-1] in b.lower() or b.lower().split()[-1] in a.lower()