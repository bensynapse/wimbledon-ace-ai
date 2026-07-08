"""
Free ATP tennis data from GitHub — no API key required.

Primary source: Tennismylife/TML-Database
  https://github.com/Tennismylife/TML-Database

Covers: match results (1968–2026), rankings, serve stats, ongoing tourneys.
Classic Jeff Sackmann tennis_atp repo is offline; TML is the active replacement.
"""

from __future__ import annotations

import logging
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

from config import GITHUB_TENNIS_REPO, PATHS
from data_sources.tennis_normalize import normalize_match_csv

logger = logging.getLogger(__name__)

RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_TENNIS_REPO}/master"

SURFACE_MAP = {
    "hard": "hard",
    "clay": "clay",
    "grass": "grass",
    "carpet": "carpet",
}


class GitHubTennisSource:
    """Download and normalize ATP data from GitHub CSV files."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or f"{PATHS['data_dir']}github_atp/")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return True

    def download_tour(
        self,
        tour: str,
        start_year: int = 2018,
        end_year: Optional[int] = None,
        refresh: bool = False,
    ) -> pd.DataFrame:
        if tour.lower() != "atp":
            raise ValueError(
                f"GitHub source covers ATP only. For WTA use TENNIS_API_KEY or another source."
            )

        cache = self.cache_dir / f"atp_matches_{start_year}_{end_year or datetime.now().year}.parquet"
        if cache.exists() and not refresh:
            logger.info("Loading GitHub ATP cache from %s", cache)
            return pd.read_parquet(cache)

        end_year = end_year or datetime.now().year
        frames: List[pd.DataFrame] = []

        for year in range(start_year, end_year + 1):
            df = self._fetch_year(year)
            if not df.empty:
                frames.append(df)

        ongoing = self._fetch_ongoing()
        if not ongoing.empty:
            frames.append(ongoing)

        if not frames:
            raise RuntimeError("No ATP data downloaded from GitHub")

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["match_id"], keep="last"
        ).sort_values("date")
        combined.to_parquet(cache, index=False)
        logger.info("Saved %d ATP matches from GitHub to %s", len(combined), cache)
        return combined

    def get_standings(self, tour: str = "atp", top_n: int = 200) -> List[Dict]:
        """Build current rankings from most recent match per player."""
        if tour.lower() != "atp":
            return []

        try:
            df = self.download_tour("atp", start_year=datetime.now().year - 1)
        except Exception as exc:
            logger.warning("Rankings fallback failed: %s", exc)
            return []

        rankings: Dict[str, Dict] = {}
        for _, row in df.sort_values("date").iterrows():
            for role in ("winner", "loser"):
                name = row.get(f"{role}_name", "")
                rank = row.get(f"{role}_rank")
                if not name or pd.isna(rank):
                    continue
                rankings[name] = {
                    "player": name,
                    "place": _safe_int(rank, 999),
                    "points": _safe_int(row.get(f"{role}_rank_points"), 0),
                }

        sorted_players = sorted(rankings.values(), key=lambda x: x["place"])
        return sorted_players[:top_n]

    def _fetch_year(self, year: int) -> pd.DataFrame:
        url = f"{RAW_BASE}/{year}.csv"
        return self._fetch_csv(url, label=str(year))

    def _fetch_ongoing(self) -> pd.DataFrame:
        url = f"{RAW_BASE}/ongoing_tourneys.csv"
        return self._fetch_csv(url, label="ongoing")

    def _fetch_csv(self, url: str, label: str) -> pd.DataFrame:
        cache_file = self.cache_dir / f"raw_{label}.csv"
        try:
            if cache_file.exists():
                raw = cache_file.read_text(encoding="utf-8")
            else:
                logger.info("Downloading GitHub ATP data: %s", url)
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                raw = response.text
                cache_file.write_text(raw, encoding="utf-8")

            tml = pd.read_csv(StringIO(raw), low_memory=False)
            return normalize_match_csv(tml, tour="atp")
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", url, exc)
            return pd.DataFrame()

