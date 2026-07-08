"""
Kaggle Huge Tennis Database — guillemservera/tennis

ATP + WTA match history (Sackmann format).
https://www.kaggle.com/datasets/guillemservera/tennis

Download:
    kaggle datasets download -d guillemservera/tennis -p data/kaggle_tennis/ --unzip

Requires ~/.kaggle/kaggle.json (Kaggle API token).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from config import KAGGLE_TENNIS_DATASET, PATHS
from data_sources.tennis_normalize import normalize_match_csv

logger = logging.getLogger(__name__)


class KaggleTennisSource:
    """Load ATP/WTA data from Kaggle guillemservera/tennis."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or f"{PATHS['data_dir']}kaggle_tennis/")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self._find_csv_files("atp") or self._find_csv_files("wta") or self._has_kaggle_cli()

    def download(self, unzip: bool = True) -> bool:
        """Download dataset via Kaggle CLI."""
        kaggle_bin = shutil.which("kaggle")
        if not kaggle_bin:
            logger.error(
                "Kaggle CLI not found. Run: pip install kaggle\n"
                "Then place API token at ~/.kaggle/kaggle.json"
            )
            return False

        cmd = [
            kaggle_bin, "datasets", "download",
            "-d", KAGGLE_TENNIS_DATASET,
            "-p", str(self.cache_dir),
        ]
        if unzip:
            cmd.append("--unzip")

        try:
            logger.info("Downloading Kaggle dataset %s ...", KAGGLE_TENNIS_DATASET)
            subprocess.run(cmd, check=True, timeout=600)
            logger.info("Kaggle dataset saved to %s", self.cache_dir)
            return True
        except subprocess.CalledProcessError as exc:
            logger.error("Kaggle download failed: %s", exc)
            return False
        except FileNotFoundError:
            logger.error("Kaggle CLI not available")
            return False

    def download_tour(
        self,
        tour: str,
        start_year: int = 2018,
        end_year: Optional[int] = None,
        refresh: bool = False,
    ) -> pd.DataFrame:
        tour = tour.lower()
        if tour not in ("atp", "wta"):
            raise ValueError(f"Kaggle source supports atp/wta, got {tour}")

        end_year = end_year or datetime.now().year
        cache = self.cache_dir / f"{tour}_matches_{start_year}_{end_year}.parquet"

        if cache.exists() and not refresh:
            return pd.read_parquet(cache)

        if not self._find_csv_files(tour):
            logger.info("No local Kaggle CSVs — attempting download...")
            self.download(unzip=True)

        frames: List[pd.DataFrame] = []
        for year in range(start_year, end_year + 1):
            for path in self._year_files(tour, year):
                try:
                    raw = pd.read_csv(path, low_memory=False)
                    norm = normalize_match_csv(raw, tour=tour)
                    if not norm.empty:
                        frames.append(norm)
                        logger.info("Loaded %d matches from %s", len(norm), path.name)
                except Exception as exc:
                    logger.warning("Skip %s: %s", path, exc)

        if not frames:
            raise RuntimeError(
                f"No {tour.upper()} data in {self.cache_dir}. "
                f"Run: kaggle datasets download -d {KAGGLE_TENNIS_DATASET} -p {self.cache_dir} --unzip"
            )

        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(subset=["match_id"], keep="last").sort_values("date")
        df.to_parquet(cache, index=False)
        return df

    def get_standings(self, tour: str = "atp", top_n: int = 200) -> List[Dict]:
        try:
            df = self.download_tour(tour, start_year=datetime.now().year - 1)
        except Exception:
            return []

        rankings: Dict[str, Dict] = {}
        for _, row in df.sort_values("date").iterrows():
            for role, rank_col, pts_col in (
                ("winner", "winner_rank", "winner_rank_points"),
                ("loser", "loser_rank", "loser_rank_points"),
            ):
                name = row.get(f"{role}_name", "")
                rank = row.get(rank_col)
                if not name or pd.isna(rank):
                    continue
                rankings[name] = {
                    "player": name,
                    "place": int(rank),
                    "points": int(row.get(pts_col, 0) or 0),
                }
        return sorted(rankings.values(), key=lambda x: x["place"])[:top_n]

    def _find_csv_files(self, tour: str) -> List[Path]:
        prefix = f"{tour}_matches_"
        files = []
        for path in sorted(self.cache_dir.rglob("*.csv")):
            name = path.name.lower()
            if name.startswith(prefix) and "qual" not in name and "futures" not in name:
                files.append(path)
        return files

    def _year_files(self, tour: str, year: int) -> List[Path]:
        patterns = [
            f"{tour}_matches_{year}.csv",
            f"{year}.csv",
        ]
        found = []
        for pattern in patterns:
            for path in self.cache_dir.rglob(pattern):
                if path not in found:
                    found.append(path)
        return found

    def _has_kaggle_cli(self) -> bool:
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        return kaggle_json.exists() and shutil.which("kaggle") is not None