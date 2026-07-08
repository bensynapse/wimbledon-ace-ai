"""
Jeff Sackmann Match Charting Project — contextual UE / winner intelligence.

Repo: https://github.com/JeffSackmann/tennis_MatchChartingProject

Provides charted winners, unforced errors (FH/BH split), rally profiles.
This is the core edge for contextual error intelligence.
"""

from __future__ import annotations

import logging
import re
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

from config import GITHUB_CHARTING_REPO, PATHS

logger = logging.getLogger(__name__)

RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_CHARTING_REPO}/master"

SURFACE_MAP = {
    "hard": "hard",
    "clay": "clay",
    "grass": "grass",
    "carpet": "carpet",
}

# Tour-average baselines for sparse-data blending (matches intelligence/expected_ue.py)
SURFACE_UE_BASELINE = {"hard": 0.19, "clay": 0.21, "grass": 0.18, "carpet": 0.19}
SURFACE_W_BASELINE = {"hard": 0.17, "clay": 0.16, "grass": 0.18, "carpet": 0.17}
SURFACE_W_UE_BASELINE = {"hard": 0.89, "clay": 0.76, "grass": 1.0, "carpet": 0.89}

MIN_SURFACE_MATCHES = 3
SPARSE_BLEND_K = 3  # pseudo-match count toward tour baseline when data is thin


class ChartingDataSource:
    """Player error/winner profiles from Sackmann charting data."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or f"{PATHS['data_dir']}charting/")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._overview: Optional[pd.DataFrame] = None
        self._matches: Optional[pd.DataFrame] = None
        self._loaded = False

    @property
    def enabled(self) -> bool:
        return True

    def warm(self, tour: str = "atp", refresh: bool = False) -> None:
        prefix = "m" if tour.lower() == "atp" else "w"
        overview = self._load_csv(
            f"charting-{prefix}-stats-Overview.csv",
            refresh=refresh,
        )
        matches = self._load_csv(f"charting-{prefix}-matches.csv", refresh=refresh)
        shot_types = self._load_csv(
            f"charting-{prefix}-stats-ShotTypes.csv",
            refresh=refresh,
        )

        if overview.empty:
            logger.warning("Charting Overview data empty for %s", tour)
            return

        overview = overview[overview["set"].astype(str).str.lower() == "total"].copy()
        if not matches.empty:
            meta = matches[["match_id", "Surface", "Tournament", "Date"]].copy()
            meta.columns = ["match_id", "surface", "tournament", "date"]
            meta["surface"] = meta["surface"].astype(str).str.lower().map(
                lambda s: SURFACE_MAP.get(s, "hard")
            )
            meta["date"] = meta["date"].astype(str)
            overview = overview.merge(meta, on="match_id", how="left")

        overview["player_norm"] = overview["player"].map(_normalize_name)
        overview["total_pts"] = (
            overview["serve_pts"].fillna(0)
            + overview["return_pts"].fillna(0)
        )
        overview["ue_pp"] = overview["unforced"] / overview["total_pts"].replace(0, 1)
        overview["w_pp"] = overview["winners"] / overview["total_pts"].replace(0, 1)
        overview["w_ue_ratio"] = overview["winners"] / overview["unforced"].replace(0, 1)
        overview["fh_ue_rate"] = overview["unforced_fh"] / overview["unforced"].replace(0, 1)
        overview["bh_ue_rate"] = overview["unforced_bh"] / overview["unforced"].replace(0, 1)

        if not shot_types.empty:
            st = shot_types[shot_types["row"].astype(str).str.lower() == "total"].copy()
            st["player_norm"] = st["player"].map(_normalize_name)
            overview = overview.merge(
                st[["match_id", "player_norm", "unforced", "winners", "induced_forced"]],
                on=["match_id", "player_norm"],
                how="left",
                suffixes=("", "_st"),
            )

        manual = self._load_manual_overrides()
        if not manual.empty:
            overview = pd.concat([overview, manual], ignore_index=True)
            overview = overview.sort_values("date", ascending=False) if "date" in overview else overview
            logger.info("Merged %d manual charting override rows", len(manual))

        self._overview = overview
        self._matches = matches
        self._loaded = True
        logger.info(
            "Charting data loaded: %d match-player rows (%s)",
            len(self._overview),
            tour.upper(),
        )

    def get_player_profile(
        self,
        player: str,
        surface: Optional[str] = None,
        last_n: int = 8,
    ) -> Optional[Dict]:
        if not self._loaded or self._overview is None or self._overview.empty:
            return None

        norm = _normalize_name(player)
        pool = self._overview[self._overview["player_norm"] == norm]
        if pool.empty:
            pool = self._fuzzy_match(player)
            if pool is None:
                return None

        surf_key = (surface or "").lower()
        surf_pool = pool[pool["surface"] == surf_key] if surf_key else pool
        use_pool = surf_pool if surf_key else pool
        recent = use_pool.head(last_n)
        if recent.empty and surf_key:
            return self._baseline_profile(player, surf_key)

        total_pts = recent["total_pts"].sum()
        if total_pts <= 0:
            return None

        ue = recent["unforced"].sum()
        winners = recent["winners"].sum()
        fh_ue = recent["unforced_fh"].sum()
        bh_ue = recent["unforced_bh"].sum()
        n = len(recent)
        manual_n = int(recent.get("is_manual", pd.Series([0])).sum()) if "is_manual" in recent else 0

        raw_ue_pp = ue / total_pts
        raw_w_pp = winners / total_pts
        raw_w_ue = winners / max(ue, 1)
        raw_fh = fh_ue / max(ue, 1)
        raw_bh = bh_ue / max(ue, 1)

        blended = surf_key and n < MIN_SURFACE_MATCHES
        if blended:
            ue_pp, w_pp, w_ue, fh_rate, bh_rate = _blend_sparse(
                n, raw_ue_pp, raw_w_pp, raw_w_ue, raw_fh, raw_bh, surf_key,
            )
            confidence = "low"
            quality = "sparse_blend"
            note = (
                f"Sparse grass data ({n} charted) — blended with tour baseline "
                f"(weight {SPARSE_BLEND_K} pseudo-matches)"
            )
        else:
            ue_pp, w_pp, w_ue, fh_rate, bh_rate = raw_ue_pp, raw_w_pp, raw_w_ue, raw_fh, raw_bh
            confidence = "high" if n >= 5 else "medium" if n >= MIN_SURFACE_MATCHES else "low"
            if manual_n == n:
                quality = "manual"
            elif manual_n > 0:
                quality = "mixed"
            else:
                quality = "charted"
            note = f"Charted {n} matches on {surf_key or 'all surfaces'}"

        return {
            "player": player,
            "charted_matches": n,
            "manual_matches": manual_n,
            "ue_per_point": round(ue_pp, 3),
            "winners_per_point": round(w_pp, 3),
            "w_ue_ratio": round(w_ue, 2),
            "forehand_ue_rate": round(fh_rate, 2),
            "backhand_ue_rate": round(bh_rate, 2),
            "forced_errors_drawn": round(
                recent.get("induced_forced", pd.Series([0])).sum() / max(total_pts, 1), 3
            ),
            "surface_filter": surf_key or "all",
            "source": "charting",
            "confidence": confidence,
            "data_quality": quality,
            "blend_note": note,
            "raw_ue_per_point": round(raw_ue_pp, 3),
            "raw_winners_per_point": round(raw_w_pp, 3),
        }

    def _fuzzy_match(self, player: str) -> Optional[pd.DataFrame]:
        if self._overview is None:
            return None
        target = _normalize_name(player)
        last = target.split()[-1] if target else ""
        if len(last) < 4:
            return None
        hits = self._overview[
            self._overview["player_norm"].str.contains(last, na=False)
        ]
        if hits["player_norm"].nunique() > 1:
            best = hits["player_norm"].value_counts().index[0]
            hits = hits[hits["player_norm"] == best]
        return hits if not hits.empty else None

    def _load_manual_overrides(self) -> pd.DataFrame:
        """Local match stats not yet in Sackmann repo (e.g. Wimbledon 2026)."""
        path = self.cache_dir / "manual_overrides.csv"
        if not path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as exc:
            logger.warning("Manual charting overrides unreadable: %s", exc)
            return pd.DataFrame()
        if df.empty:
            return pd.DataFrame()

        required = {
            "match_id", "player", "surface", "date",
            "serve_pts", "return_pts", "winners", "unforced",
        }
        if not required.issubset(df.columns):
            logger.warning("manual_overrides.csv missing columns: %s", required - set(df.columns))
            return pd.DataFrame()

        df = df.copy()
        df["set"] = "Total"
        df["surface"] = df["surface"].astype(str).str.lower()
        df["date"] = df["date"].astype(str)
        df["player_norm"] = df["player"].map(_normalize_name)
        df["total_pts"] = df["serve_pts"].fillna(0) + df["return_pts"].fillna(0)
        for col, default in (
            ("winners_fh", "winners"), ("winners_bh", 0),
            ("unforced_fh", "unforced"), ("unforced_bh", "unforced"),
        ):
            if col not in df.columns:
                if isinstance(default, str):
                    df[col] = (df[default] * 0.5).round().astype(int)
                else:
                    df[col] = default
        df["is_manual"] = 1
        if "source_note" not in df.columns:
            df["source_note"] = "manual"
        return df

    def _load_csv(self, filename: str, refresh: bool = False) -> pd.DataFrame:
        cache = self.cache_dir / filename
        try:
            if cache.exists() and not refresh:
                return pd.read_csv(cache, low_memory=False)
            url = f"{RAW_BASE}/{filename}"
            logger.info("Downloading charting data: %s", filename)
            response = requests.get(url, timeout=90)
            response.raise_for_status()
            cache.write_text(response.text, encoding="utf-8")
            return pd.read_csv(StringIO(response.text), low_memory=False)
        except Exception as exc:
            logger.warning("Charting fetch failed %s: %s", filename, exc)
            if cache.exists():
                return pd.read_csv(cache, low_memory=False)
            return pd.DataFrame()


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _blend_sparse(
    n: int,
    raw_ue: float,
    raw_w: float,
    raw_w_ue: float,
    raw_fh: float,
    raw_bh: float,
    surface: str,
) -> tuple:
    """Empirical-Bayes shrink toward tour baseline when n < MIN_SURFACE_MATCHES."""
    k = SPARSE_BLEND_K
    base_ue = SURFACE_UE_BASELINE.get(surface, 0.19)
    base_w = SURFACE_W_BASELINE.get(surface, 0.17)
    base_w_ue = SURFACE_W_UE_BASELINE.get(surface, 0.9)
    ue_pp = (n * raw_ue + k * base_ue) / (n + k)
    w_pp = (n * raw_w + k * base_w) / (n + k)
    w_ue = (n * raw_w_ue + k * base_w_ue) / (n + k)
    fh_rate = (n * raw_fh + k * 0.5) / (n + k)
    bh_rate = (n * raw_bh + k * 0.5) / (n + k)
    return ue_pp, w_pp, w_ue, fh_rate, bh_rate


def _baseline_profile(player: str, surface: str) -> Dict:
    return {
        "player": player,
        "charted_matches": 0,
        "manual_matches": 0,
        "ue_per_point": SURFACE_UE_BASELINE.get(surface, 0.19),
        "winners_per_point": SURFACE_W_BASELINE.get(surface, 0.18),
        "w_ue_ratio": SURFACE_W_UE_BASELINE.get(surface, 1.0),
        "forehand_ue_rate": 0.5,
        "backhand_ue_rate": 0.5,
        "forced_errors_drawn": 0.09,
        "surface_filter": surface,
        "source": "estimated",
        "confidence": "low",
        "data_quality": "baseline",
        "blend_note": f"No charted {surface} matches — tour-average baseline only",
    }


def apply_charting_to_context(ctx: Dict, charting: ChartingDataSource, p1: str, p2: str, surface: str) -> None:
    """Merge charting UE/W profiles into match context for intelligence layer."""
    for key, player in (("player1", p1), ("player2", p2)):
        profile = charting.get_player_profile(player, surface=surface)
        if not profile:
            continue
        ctx[f"{key}_ue_pp"] = profile["ue_per_point"]
        ctx[f"{key}_w_pp"] = profile["winners_per_point"]
        ctx[f"{key}_ue_source"] = "charting"
        ctx[f"{key}_forehand_ue_rate"] = profile["forehand_ue_rate"]
        ctx[f"{key}_backhand_ue_rate"] = profile["backhand_ue_rate"]
        ctx[f"{key}_charted_matches"] = profile["charted_matches"]
        ctx[f"{key}_w_ue_ratio"] = profile["w_ue_ratio"]
        ctx[f"{key}_ue_confidence"] = profile.get("confidence", "low")
        ctx[f"{key}_ue_data_quality"] = profile.get("data_quality", "charted")
        if profile.get("blend_note"):
            ctx[f"{key}_ue_blend_note"] = profile["blend_note"]
        if profile.get("raw_ue_per_point") is not None:
            ctx[f"{key}_raw_ue_pp"] = profile["raw_ue_per_point"]
        if profile["backhand_ue_rate"] > 0.55:
            ctx.setdefault(f"{key}_context_flags", []).append("backhand_under_pressure")
        quality = profile.get("data_quality", "charted")
        prefix = "Charted" if quality == "charted" else quality.replace("_", " ").title()
        ctx[f"{key}_error_profile_note"] = (
            f"{prefix} {profile['charted_matches']} matches on {surface}: "
            f"UE {profile['ue_per_point']:.2f}/pt, W {profile['winners_per_point']:.2f}/pt, "
            f"FH/BH UE split {profile['forehand_ue_rate']:.0%}/{profile['backhand_ue_rate']:.0%}"
            + (f" — {profile['blend_note']}" if profile.get("blend_note") and quality == "sparse_blend" else "")
        )