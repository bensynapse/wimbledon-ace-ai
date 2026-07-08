"""Shared Sackmann/TML CSV → internal match schema."""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

SURFACE_MAP = {
    "hard": "hard",
    "clay": "clay",
    "grass": "grass",
    "carpet": "carpet",
}


def normalize_match_csv(df: pd.DataFrame, tour: str = "atp") -> pd.DataFrame:
    if df.empty:
        return df
    rows: List[Dict] = []
    for _, r in df.iterrows():
        row = _row_to_match(r, tour=tour)
        if row:
            rows.append(row)
    return pd.DataFrame(rows)


def _row_to_match(r, tour: str = "atp") -> Optional[Dict]:
    winner = str(r.get("winner_name", "")).strip()
    loser = str(r.get("loser_name", "")).strip()
    if not winner or not loser or winner == "nan" or loser == "nan":
        return None

    if winner.lower() < loser.lower():
        player1, player2 = winner, loser
        winner_is_p1 = True
    else:
        player1, player2 = loser, winner
        winner_is_p1 = False

    surface = SURFACE_MAP.get(str(r.get("surface", "Hard")).lower(), "hard")
    formatted_date = _format_tourney_date(r.get("tourney_date"))
    if not formatted_date:
        return None

    sets_w, sets_l = _parse_score_sets(str(r.get("score", "")))
    total_sets = max(sets_w + sets_l, 3)
    minutes = _safe_int(r.get("minutes"), default=total_sets * 52)

    return {
        "match_id": f"{r.get('tourney_id', '')}_{_safe_int(r.get('match_num'), 0)}_{formatted_date}",
        "date": formatted_date,
        "player1": player1,
        "player2": player2,
        "player1_key": str(r.get("winner_id" if winner_is_p1 else "loser_id", "")),
        "player2_key": str(r.get("loser_id" if winner_is_p1 else "winner_id", "")),
        "winner_is_player1": winner_is_p1,
        "tournament": str(r.get("tourney_name", "")),
        "round": str(r.get("round", "")),
        "surface": surface,
        "tour": tour.lower(),
        "sets_player1": sets_w if winner_is_p1 else sets_l,
        "sets_player2": sets_l if winner_is_p1 else sets_w,
        "total_sets": total_sets,
        "minutes_played": minutes,
        "winner_name": winner,
        "loser_name": loser,
        "winner_rank": _safe_float(r.get("winner_rank")),
        "loser_rank": _safe_float(r.get("loser_rank")),
        "winner_rank_points": _safe_float(r.get("winner_rank_points")),
        "loser_rank_points": _safe_float(r.get("loser_rank_points")),
        "w_ace": _safe_float(r.get("w_ace")),
        "w_df": _safe_float(r.get("w_df")),
        "l_ace": _safe_float(r.get("l_ace")),
        "l_df": _safe_float(r.get("l_df")),
        "w_svpt": _safe_float(r.get("w_svpt")),
        "l_svpt": _safe_float(r.get("l_svpt")),
    }


def _safe_int(value, default: int = 0) -> int:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_float(value, default: float = 0.0) -> float:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_tourney_date(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    try:
        date_str = str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)[:10]
    if len(date_str) == 8:
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    return ""


def normalize_dissfya_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    dissfya/atp-tennis-2000-2023daily-pull → internal schema with historical odds.

    Columns: Tournament, Date, Player_1, Player_2, Winner, Rank_1/2, Pts_1/2,
    Odd_1/2, Surface, Round, Score, Best of, Series, Court.
    """
    if df.empty:
        return df

    col_map = {c.lower(): c for c in df.columns}
    rows: List[Dict] = []
    for _, r in df.iterrows():
        row = _dissfya_row_to_match(r, col_map)
        if row:
            rows.append(row)
    return pd.DataFrame(rows)


def _dissfya_row_to_match(r, col_map: Dict[str, str]) -> Optional[Dict]:
    def get(*names, default=None):
        for name in names:
            key = col_map.get(name.lower())
            if key is not None:
                val = r.get(key, default)
                if val is not None and not (isinstance(val, float) and pd.isna(val)):
                    return val
        return default

    p1 = str(get("player_1", "Player_1", default="")).strip()
    p2 = str(get("player_2", "Player_2", default="")).strip()
    winner = str(get("winner", "Winner", default="")).strip()
    if not p1 or not p2 or not winner or p1 == "nan" or p2 == "nan":
        return None

    rank1 = _safe_float(get("rank_1", "Rank_1"), default=-1)
    rank2 = _safe_float(get("rank_2", "Rank_2"), default=-1)
    pts1 = _safe_float(get("pts_1", "Pts_1"), default=-1)
    pts2 = _safe_float(get("pts_2", "Pts_2"), default=-1)
    odd1 = _safe_float(get("odd_1", "Odd_1"), default=-1)
    odd2 = _safe_float(get("odd_2", "Odd_2"), default=-1)

    if rank1 < 1 or rank2 < 1 or odd1 < 1 or odd2 < 1:
        return None

    date_raw = get("date", "Date")
    formatted_date = _format_dissfya_date(date_raw)
    if not formatted_date:
        return None

    surface = SURFACE_MAP.get(str(get("surface", "Surface", default="Hard")).lower(), "hard")
    score = str(get("score", "Score", default=""))
    sets_w, sets_l = _parse_score_sets(score)
    winner_is_p1 = winner == p1
    best_of = _safe_int(get("best of", "Best of", "Best_of"), default=3)

    return {
        "match_id": f"{formatted_date}_{p1}_{p2}".replace(" ", "_").lower(),
        "date": formatted_date,
        "player1": p1,
        "player2": p2,
        "winner_is_player1": winner_is_p1,
        "tournament": str(get("tournament", "Tournament", default="")),
        "series": str(get("series", "Series", default="")),
        "round": str(get("round", "Round", default="")),
        "surface": surface,
        "court": str(get("court", "Court", default="")),
        "best_of": best_of,
        "tour": "atp",
        "sets_player1": sets_w if winner_is_p1 else sets_l,
        "sets_player2": sets_l if winner_is_p1 else sets_w,
        "total_sets": max(sets_w + sets_l, best_of),
        "score": score,
        "winner_name": winner,
        "loser_name": p2 if winner_is_p1 else p1,
        "rank_player1": rank1,
        "rank_player2": rank2,
        "pts_player1": pts1,
        "pts_player2": pts2,
        "rank_diff": rank2 - rank1,
        "pts_diff": pts1 - pts2,
        "odds_diff": odd2 - odd1,
        "odd_player1": odd1,
        "odd_player2": odd2,
        "winner_rank": rank1 if winner_is_p1 else rank2,
        "loser_rank": rank2 if winner_is_p1 else rank1,
        "winner_rank_points": pts1 if winner_is_p1 else pts2,
        "loser_rank_points": pts2 if winner_is_p1 else pts1,
    }


def _format_dissfya_date(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()[:10]
    if len(text) == 10 and text[4] == "-":
        return text
    try:
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return ""
        return parsed.strftime("%Y-%m-%d")
    except Exception:
        return ""


def _parse_score_sets(score: str) -> tuple:
    if not score or score.lower() in ("w/o", "ret", "def"):
        return 2, 0
    sets_w, sets_l = 0, 0
    for part in score.split():
        if "-" not in part:
            continue
        left = part.split("-")[0].split("(")[0]
        right = part.split("-")[1].split("(")[0]
        try:
            g1, g2 = int(left), int(right)
            if g1 > g2:
                sets_w += 1
            elif g2 > g1:
                sets_l += 1
        except ValueError:
            continue
    return sets_w or 2, sets_l or 1