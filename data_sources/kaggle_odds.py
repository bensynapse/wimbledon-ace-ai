"""
Kaggle ATP Tennis with historical bookmaker odds — dissfya/atp-tennis-2000-2023daily-pull

65k+ ATP matches (2000–2025) with Odd_1 / Odd_2 for backtesting value detection.
https://www.kaggle.com/datasets/dissfya/atp-tennis-2000-2023daily-pull

Download:
    kaggle datasets download -d dissfya/atp-tennis-2000-2023daily-pull -p data/kaggle_odds/ --unzip

Requires ~/.kaggle/kaggle.json (Kaggle API token).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd

from config import KAGGLE_ATP_ODDS_DATASET, PATHS
from data_sources.tennis_normalize import normalize_dissfya_csv

logger = logging.getLogger(__name__)

CSV_NAMES = ("atp_tennis.csv", "ATP_Tennis.csv")


class KaggleOddsSource:
    """Load ATP match history with bookmaker odds from Kaggle dissfya dataset."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or f"{PATHS['data_dir']}kaggle_odds/")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self._find_csv() is not None or self._has_kaggle_cli()

    def download(self, unzip: bool = True) -> bool:
        kaggle_bin = shutil.which("kaggle")
        if not kaggle_bin:
            logger.error(
                "Kaggle CLI not found. Run: pip install kaggle\n"
                "Then place API token at ~/.kaggle/kaggle.json"
            )
            return False

        cmd = [
            kaggle_bin, "datasets", "download",
            "-d", KAGGLE_ATP_ODDS_DATASET,
            "-p", str(self.cache_dir),
        ]
        if unzip:
            cmd.append("--unzip")

        try:
            logger.info("Downloading Kaggle odds dataset %s ...", KAGGLE_ATP_ODDS_DATASET)
            subprocess.run(cmd, check=True, timeout=600)
            logger.info("Kaggle odds dataset saved to %s", self.cache_dir)
            return True
        except subprocess.CalledProcessError as exc:
            logger.error("Kaggle download failed: %s", exc)
            return False
        except FileNotFoundError:
            logger.error("Kaggle CLI not available")
            return False

    def load(
        self,
        start_year: int = 2000,
        end_year: Optional[int] = None,
        surface: Optional[str] = None,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Load normalized matches with historical odds, cached as parquet."""
        from datetime import datetime

        end_year = end_year or datetime.now().year
        cache_key = f"atp_odds_{start_year}_{end_year}"
        if surface:
            cache_key += f"_{surface}"
        cache = self.cache_dir / f"{cache_key}.parquet"

        if cache.exists() and not refresh:
            df = pd.read_parquet(cache)
            return df

        csv_path = self._find_csv()
        if csv_path is None:
            logger.info("No local odds CSV — attempting Kaggle download...")
            if not self.download(unzip=True):
                raise RuntimeError(
                    f"No ATP odds data in {self.cache_dir}. "
                    f"Run: bash scripts/download_kaggle_odds.sh"
                )
            csv_path = self._find_csv()

        if csv_path is None:
            raise RuntimeError(f"Download succeeded but no CSV found in {self.cache_dir}")

        raw = pd.read_csv(csv_path, low_memory=False)
        df = normalize_dissfya_csv(raw)
        if df.empty:
            raise RuntimeError(f"No valid rows after normalizing {csv_path}")

        df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year
        df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]
        if surface:
            df = df[df["surface"] == surface.lower()]

        df = df.drop_duplicates(subset=["match_id"], keep="last").sort_values("date")
        df.to_parquet(cache, index=False)
        logger.info("Loaded %d ATP matches with odds (%d–%d)", len(df), start_year, end_year)
        return df.reset_index(drop=True)

    def summary(self) -> dict:
        try:
            df = self.load()
        except Exception as exc:
            return {"available": False, "error": str(exc)}
        return {
            "available": True,
            "rows": len(df),
            "date_min": str(df["date"].min()),
            "date_max": str(df["date"].max()),
            "surfaces": df["surface"].value_counts().to_dict(),
            "with_odds": int((df["odd_player1"] > 1).sum()),
        }

    def _find_csv(self) -> Optional[Path]:
        for name in CSV_NAMES:
            direct = self.cache_dir / name
            if direct.exists():
                return direct
        for path in sorted(self.cache_dir.rglob("*.csv")):
            if path.name.lower() in {n.lower() for n in CSV_NAMES}:
                return path
            if "atp" in path.name.lower() and "tennis" in path.name.lower():
                return path
        return None

    def _has_kaggle_cli(self) -> bool:
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        return kaggle_json.exists() and shutil.which("kaggle") is not None