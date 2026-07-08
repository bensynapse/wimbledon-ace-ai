"""
Auto-build match context from cached history + live rankings.

Feeds probability model + intelligence layer without manual JSON.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

from config import MATCH_HISTORY_WINDOW
from data_sources.charting_source import ChartingDataSource, apply_charting_to_context
from data_sources.news_quotes import NewsQuotesSource
from data_sources.tennis_abstract_elo import TennisAbstractEloSource
from feature_engineering import DEFAULT_ELO, TennisFeatureEngine

logger = logging.getLogger(__name__)

MINUTES_PER_SET = 52
FIVE_SET_MATCH_SETS = 5


class MatchContextBuilder:
    def __init__(self, history_df: Optional[pd.DataFrame] = None):
        self.df = history_df
        self.engine = TennisFeatureEngine(history_window=MATCH_HISTORY_WINDOW)
        self.charting = ChartingDataSource()
        self.tennis_abstract_elo = TennisAbstractEloSource()
        self.news_quotes = NewsQuotesSource()
        self._charting_warmed = False

    def warm(
        self,
        df: pd.DataFrame,
        standings: Optional[List[Dict]] = None,
        tour: str = "atp",
    ) -> None:
        self.df = df
        self.engine = TennisFeatureEngine(history_window=MATCH_HISTORY_WINDOW)
        self.engine.ingest_history(df)
        if standings:
            self.engine.set_rankings(standings)
        try:
            self.charting.warm(tour=tour)
            self._charting_warmed = True
        except Exception as exc:
            logger.warning("Charting data not loaded: %s", exc)

    def build(
        self,
        player1: str,
        player2: str,
        surface: str = "hard",
        match_date: Optional[str] = None,
        tournament: str = "",
        extra: Optional[Dict] = None,
    ) -> Dict:
        extra = dict(extra or {})
        match_date = match_date or datetime.now().strftime("%Y-%m-%d")
        surface = (surface or "hard").lower()

        _, feat_ctx = self.engine.build_match_features(
            player1, player2, surface=surface, match_date=match_date
        )
        ctx = {**feat_ctx, **extra}

        p1_elo = self.engine.elo.get(player1, DEFAULT_ELO)
        p2_elo = self.engine.elo.get(player2, DEFAULT_ELO)
        p1_surf = self.engine.surface_elo.get((player1, surface), DEFAULT_ELO)
        p2_surf = self.engine.surface_elo.get((player2, surface), DEFAULT_ELO)

        ctx["player1_elo"] = p1_elo
        ctx["player2_elo"] = p2_elo
        ctx["player1_surface_elo"] = p1_surf
        ctx["player2_surface_elo"] = p2_surf
        ctx["player1_form"] = self.engine._recent_win_rate(player1)
        ctx["player2_form"] = self.engine._recent_win_rate(player2)
        ctx["player1_surface_form"] = self.engine._recent_win_rate(player1, surface=surface)
        ctx["player2_surface_form"] = self.engine._recent_win_rate(player2, surface=surface)

        if self.df is not None and not self.df.empty:
            self._append_schedule_context(ctx, player1, player2, match_date, tournament)
            self._append_serve_context(ctx, player1, player2)

        if self._charting_warmed:
            apply_charting_to_context(ctx, self.charting, player1, player2, surface)

        if self.df is not None and not self.df.empty:
            self._append_h2h_context(ctx, player1, player2)
            self._append_tournament_momentum(ctx, player1, player2, tournament, match_date)
            self._append_player_identity(ctx, player1, player2, tournament, surface)

        self._apply_tennis_abstract_elo(ctx, player1, player2, surface)
        self.news_quotes.apply_to_context(ctx, player1, player2, tournament=tournament)

        ctx["tournament"] = tournament or ctx.get("tournament", "")
        ctx.update({k: v for k, v in extra.items() if v is not None})
        return ctx

    def _append_serve_context(self, ctx: Dict, player1: str, player2: str) -> None:
        """UE/serve hints from TML stats — skipped if charting already set UE."""
        for key, player in (("player1", player1), ("player2", player2)):
            if (
                ctx.get(f"{key}_ue_source") == "charting"
                and ctx.get(f"{key}_ue_confidence") in ("medium", "high")
            ):
                continue
            recent = self._player_recent_matches(player, datetime.now())
            if recent.empty or "w_svpt" not in recent.columns:
                continue
            pts, dfs, aces = 0.0, 0.0, 0.0
            for _, row in recent.head(5).iterrows():
                is_winner = row.get("winner_name") == player
                svpt = row.get("w_svpt" if is_winner else "l_svpt", 0) or 0
                df = row.get("w_df" if is_winner else "l_df", 0) or 0
                ace = row.get("w_ace" if is_winner else "l_ace", 0) or 0
                pts += float(svpt)
                dfs += float(df)
                aces += float(ace)
            if pts > 0:
                ctx[f"{key}_ue_pp"] = round(dfs / pts, 3)
                ctx[f"{key}_w_pp"] = round(aces / pts, 3)
                ctx[f"{key}_ue_source"] = "official"
                ctx[f"{key}_serve_return"] = round(0.5 + (aces - dfs) / pts * 0.5, 3)

    def _append_schedule_context(
        self,
        ctx: Dict,
        player1: str,
        player2: str,
        match_date: str,
        tournament: str,
    ) -> None:
        current = _parse_date(match_date) or datetime.now()
        for key, player in (("player1", player1), ("player2", player2)):
            recent = self._player_recent_matches(player, current)
            ctx.update(self._fatigue_fields(key, recent, current))
            ctx[f"{key}_path_note"] = self._path_note(recent, player)

        p1_path = ctx.get("player1_path_note", "")
        p2_path = ctx.get("player2_path_note", "")
        if p1_path and p2_path and p1_path != p2_path:
            easier = player1 if "lighter" in p1_path else player2 if "lighter" in p2_path else ""
            if easier:
                other = player2 if easier == player1 else player1
                ctx["path_diff"] = f"{easier} had lighter draw; {other} faced heavier opposition"

    def _player_recent_matches(self, player: str, current: datetime) -> pd.DataFrame:
        mask = (self.df["player1"] == player) | (self.df["player2"] == player)
        sub = self.df.loc[mask].copy()
        if sub.empty:
            return sub
        sub["parsed_date"] = pd.to_datetime(sub["date"], errors="coerce")
        sub = sub[sub["parsed_date"] <= current]
        return sub.sort_values("parsed_date", ascending=False)

    def _fatigue_fields(self, prefix: str, recent: pd.DataFrame, current: datetime) -> Dict:
        if recent.empty:
            return {
                f"{prefix}_minutes_7d": 0,
                f"{prefix}_sets_7d": 0,
                f"{prefix}_matches_7d": 0,
                f"{prefix}_rest_days": 3,
                f"{prefix}_five_set_recent": False,
            }

        week_ago = current - timedelta(days=7)
        last_7 = recent[recent["parsed_date"] >= week_ago]
        sets_7d = int(last_7["total_sets"].sum()) if "total_sets" in last_7 else len(last_7) * 3
        minutes_7d = int(last_7["minutes_played"].sum()) if "minutes_played" in last_7 else sets_7d * MINUTES_PER_SET

        last = recent.iloc[0]
        last_date = last["parsed_date"]
        rest_days = max((current - last_date).days, 0) if pd.notna(last_date) else 2
        last_sets = int(last.get("total_sets", 3))
        five_set = last_sets >= FIVE_SET_MATCH_SETS

        flags = []
        if minutes_7d > 400:
            flags.append("fatigue_medium")
        if minutes_7d > 550:
            flags.append("fatigue_high")
        if five_set:
            flags.append("five_set_recent")

        return {
            f"{prefix}_minutes_7d": minutes_7d,
            f"{prefix}_sets_7d": sets_7d,
            f"{prefix}_matches_7d": len(last_7),
            f"{prefix}_rest_days": rest_days,
            f"{prefix}_five_set_recent": five_set,
            f"{prefix}_fatigue_score": min(1.0, minutes_7d / 700),
            f"{prefix}_context_flags": flags,
        }

    def _apply_tennis_abstract_elo(
        self,
        ctx: Dict,
        player1: str,
        player2: str,
        surface: str,
    ) -> None:
        try:
            self.tennis_abstract_elo.apply_to_context(ctx, player1, player2, surface=surface)
        except Exception as exc:
            logger.debug("Tennis Abstract Elo not applied: %s", exc)

    def _append_h2h_context(self, ctx: Dict, player1: str, player2: str) -> None:
        meetings = self.df[
            ((self.df["player1"] == player1) & (self.df["player2"] == player2))
            | ((self.df["player1"] == player2) & (self.df["player2"] == player1))
        ].sort_values("date", ascending=False)
        if meetings.empty:
            return
        last = meetings.iloc[0]
        winner = last.get("winner_name", "")
        ctx["h2h_recent_winner"] = str(winner)
        ctx["h2h_meetings"] = len(meetings)
        p1_wins = sum(1 for _, r in meetings.iterrows() if r.get("winner_name") == player1)
        ctx["h2h_p1_win_rate"] = round(p1_wins / len(meetings), 3)

    def _append_tournament_momentum(
        self,
        ctx: Dict,
        player1: str,
        player2: str,
        tournament: str,
        match_date: str,
    ) -> None:
        if not tournament:
            return
        current = _parse_date(match_date) or datetime.now()
        t_key = tournament.lower()
        for prefix, player in (("player1", player1), ("player2", player2)):
            mask = (
                (self.df["player1"] == player) | (self.df["player2"] == player)
            ) & self.df["tournament"].astype(str).str.lower().str.contains(
                "wimbledon" if "wimbledon" in t_key else t_key.split()[0],
                case=False,
                na=False,
            )
            sub = self.df.loc[mask].copy()
            if sub.empty:
                continue
            sub["parsed_date"] = pd.to_datetime(sub["date"], errors="coerce")
            sub = sub[sub["parsed_date"] <= current].sort_values("parsed_date", ascending=False)
            upset_wins = 0
            for _, row in sub.iterrows():
                if row.get("winner_name") != player:
                    continue
                opp = row["player2"] if row["player1"] == player else row["player1"]
                opp_elo = self.engine.elo.get(opp, DEFAULT_ELO)
                player_elo = self.engine.elo.get(player, DEFAULT_ELO)
                if opp_elo >= player_elo + 40:
                    upset_wins += 1
            ctx[f"{prefix}_tournament_upset_wins"] = upset_wins
            ctx[f"{prefix}_tournament_matches"] = len(sub)

    def _append_player_identity(
        self,
        ctx: Dict,
        player1: str,
        player2: str,
        tournament: str,
        surface: str,
    ) -> None:
        home_players = {
            "arthur fery": "gbr",
            "jacob fearnley": "gbr",
            "cameron norrie": "gbr",
            "jack draper": "gbr",
            "jack pinnington jones": "gbr",
        }
        t_lower = tournament.lower()
        is_wimbledon = "wimbledon" in t_lower

        for prefix, player in (("player1", player1), ("player2", player2)):
            recent = self._player_recent_matches(player, datetime.now())
            rank = None
            if not recent.empty:
                row = recent.iloc[0]
                if row.get("player1") == player:
                    rank = row.get("winner_rank") if row.get("winner_name") == player else row.get("loser_rank")
                else:
                    rank = row.get("winner_rank") if row.get("winner_name") == player else row.get("loser_rank")
                for col in ("winner_rank", "loser_rank", "rank_player1", "rank_player2"):
                    if rank is None or (isinstance(rank, float) and pd.isna(rank)):
                        if col in row.index:
                            val = row.get(col)
                            if val and not (isinstance(val, float) and pd.isna(val)):
                                rank = val
            if rank is not None and not (isinstance(rank, float) and pd.isna(rank)):
                ctx[f"{prefix}_rank"] = int(rank)

            nation = home_players.get(player.lower(), "")
            ctx[f"{prefix}_home_slam"] = is_wimbledon and nation == "gbr"
            rank_val = ctx.get(f"{prefix}_rank")
            if rank_val and int(rank_val) > 150 and is_wimbledon:
                ctx[f"{prefix}_is_wildcard"] = True
            if ctx.get(f"{prefix}_tournament_upset_wins", 0) >= 2 and is_wimbledon:
                ctx[f"{prefix}_is_wildcard"] = True
            round_name = str(ctx.get("round", "")).lower()
            if (
                is_wimbledon
                and rank_val
                and int(rank_val) > 100
                and any(r in round_name for r in ("qf", "quarter", "sf", "semi", "f"))
            ):
                ctx[f"{prefix}_is_wildcard"] = True

    def _path_note(self, recent: pd.DataFrame, player: str) -> str:
        if recent.empty:
            return ""
        strengths = []
        for _, row in recent.head(5).iterrows():
            opp = row["player2"] if row["player1"] == player else row["player1"]
            strengths.append(self.engine.elo.get(opp, DEFAULT_ELO))
        avg_opp = sum(strengths) / len(strengths)
        if avg_opp >= 1550:
            return "heavier recent opposition"
        if avg_opp <= 1480:
            return "lighter recent draw"
        return "average opposition quality"


def _parse_date(value: str) -> Optional[datetime]:
    for fmt in ("%Y-%m-%d", "%d.%m.%Y"):
        try:
            return datetime.strptime(str(value)[:10], fmt)
        except ValueError:
            continue
    return None