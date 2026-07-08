"""
WimbledonAce Elo leaderboard — lijst + puntensimulatie per matchup.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

from data_sources.tennis_abstract_elo import TennisAbstractEloSource
from models.elo_tracker import EloExchange, WimbledonAceEloTracker


def rebuild_elo_ratings(tour: str = "atp", start_year: int = 2018) -> pd.DataFrame:
    from data_collector import HistoricalDataCollector

    hc = HistoricalDataCollector()
    df = hc.download_tour(tour, start_year=start_year, refresh=False)
    tracker = WimbledonAceEloTracker()
    return tracker.rebuild_from_history(df, tour=tour)


def format_exchange_report(
    exchange: EloExchange,
    tournament: str = "",
    round_name: str = "",
) -> str:
    a, b = exchange.player_a, exchange.player_b
    surf = exchange.surface
    header = f"{a} vs {b}"
    if tournament:
        header += f" | {tournament}"
    if round_name:
        header += f" | {round_name}"
    header += f" | {surf}"

    lines = [
        "",
        f"═══ WimbledonAce Elo — Puntensimulatie ═══",
        header,
        f"K-factor deze match: {exchange.k_factor}",
        "",
        f"Huidige ratings ({surf}):",
        f"  {a}: overall {exchange.elo_a} | {surf} {exchange.surface_elo_a}",
        f"  {b}: overall {exchange.elo_b} | {surf} {exchange.surface_elo_b}",
        "",
        f"Win-kans ({surf} Elo):",
        f"  {a}: {exchange.win_prob_a:.1%}",
        f"  {b}: {exchange.win_prob_b:.1%}",
        "",
        "Als de match gespeeld wordt:",
        f"  ✅ {a} wint → {a} {exchange.if_a_wins_delta_a:+.1f} | {b} {exchange.if_a_wins_delta_b:+.1f}",
        f"     nieuw {surf}: {exchange.surface_elo_a + exchange.if_a_wins_delta_a:.1f} vs "
        f"{exchange.surface_elo_b + exchange.if_a_wins_delta_b:.1f}",
        f"  ✅ {b} wint → {b} {exchange.if_b_wins_delta_b:+.1f} | {a} {exchange.if_b_wins_delta_a:+.1f}",
        f"     nieuw {surf}: {exchange.surface_elo_b + exchange.if_b_wins_delta_b:.1f} vs "
        f"{exchange.surface_elo_a + exchange.if_b_wins_delta_a:.1f}",
        "",
        "Elo-logica: delta = K × (resultaat − verwachting)",
        "  resultaat = 1 bij win, 0 bij verlies",
        "  verwachting = 1 / (1 + 10^((tegenstander − jij) / 400))",
        "",
    ]
    return "\n".join(lines)


def format_leaderboard_table(
    df: pd.DataFrame,
    surface: Optional[str] = None,
    top_n: int = 25,
    compare_ta: bool = False,
) -> str:
    sort_col = f"{surface}_elo" if surface else "overall_elo"
    if sort_col not in df.columns:
        sort_col = "overall_elo"
    view = df.sort_values(sort_col, ascending=False).head(top_n).copy()
    view["elo_rank"] = range(1, len(view) + 1)

    ta = TennisAbstractEloSource() if compare_ta else None
    lines = [
        "",
        f"═══ WimbledonAce Elo Leaderboard (top {len(view)}) ═══",
        f"Sorted by: {sort_col}",
        "",
        f"{'#':>3}  {'Player':<22} {'Overall':>8} {'Grass':>8} {'Hard':>8} {'Clay':>8}  {'M':>4} {'W%':>5}",
    ]
    if compare_ta:
        lines[-1] += f"  {'TA Elo':>8}"

    for _, row in view.iterrows():
        wr = row["wins"] / row["matches"] if row["matches"] else 0
        line = (
            f"{int(row['elo_rank']):>3}  {str(row['player'])[:22]:<22} "
            f"{row['overall_elo']:>8.0f} {row['grass_elo']:>8.0f} "
            f"{row['hard_elo']:>8.0f} {row['clay_elo']:>8.0f}  "
            f"{int(row['matches']):>4} {wr:>4.0%}"
        )
        if compare_ta and ta:
            lookup = ta.lookup(str(row["player"]))
            ta_elo = f"{lookup['elo']:.0f}" if lookup else "n/a"
            line += f"  {ta_elo:>8}"
        lines.append(line)
    lines.append("")
    return "\n".join(lines)


def export_matchup_sheet(
    matchups: List[dict],
    tour: str = "atp",
    output_path: Optional[Path] = None,
) -> Path:
    """Export CSV met winst/verlies Elo-delta per geplande matchup."""
    tracker = WimbledonAceEloTracker()
    tracker.load(tour)
    rows = []
    for m in matchups:
        ex = tracker.compute_exchange(
            m["player1"], m["player2"],
            surface=m.get("surface", "hard"),
            tournament=m.get("tournament", ""),
            round_name=m.get("round", ""),
            best_of=m.get("best_of", 3),
        )
        rows.append({
            "player1": ex.player_a,
            "player2": ex.player_b,
            "surface": ex.surface,
            "tournament": m.get("tournament", ""),
            "round": m.get("round", ""),
            "p1_grass_elo": ex.surface_elo_a if ex.surface == "grass" else "",
            "p2_grass_elo": ex.surface_elo_b if ex.surface == "grass" else "",
            "p1_win_prob": ex.win_prob_a,
            "p2_win_prob": ex.win_prob_b,
            "k_factor": ex.k_factor,
            "p1_if_wins": ex.if_a_wins_delta_a,
            "p1_if_loses": ex.if_b_wins_delta_a,
            "p2_if_wins": ex.if_b_wins_delta_b,
            "p2_if_loses": ex.if_a_wins_delta_b,
        })
    out = output_path or Path(f"data/wimbledon_ace_elo/{tour}_matchup_deltas.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return out