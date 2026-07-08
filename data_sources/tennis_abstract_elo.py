"""
Tennis Abstract Elo ratings — weekly updated professional Elo source.

Source: https://www.tennisabstract.com/reports/atp_elo_ratings.html
Updated weekly(ish) by Jeff Sackmann / Heavy Topspin.

Provides overall Elo plus surface-specific hElo / cElo / gElo.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

from config import PATHS

logger = logging.getLogger(__name__)

ATP_ELO_URL = "https://www.tennisabstract.com/reports/atp_elo_ratings.html"
WTA_ELO_URL = "https://www.tennisabstract.com/reports/wta_elo_ratings.html"
CACHE_MAX_AGE_DAYS = 7

SURFACE_ELO_COL = {
    "hard": "hElo",
    "clay": "cElo",
    "grass": "gElo",
    "carpet": "hElo",
}

USER_AGENT = "WimbledonAceAI/1.0 (+tennis-intelligence-bot)"


class TennisAbstractEloSource:
    """Fetch and cache Tennis Abstract Elo leaderboards."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or f"{PATHS['data_dir']}tennis_abstract/")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self._cache_path("atp").exists()

    def download(self, tour: str = "atp", refresh: bool = False) -> pd.DataFrame:
        tour = tour.lower()
        cache = self._cache_path(tour)
        meta = self._meta_path(tour)

        if cache.exists() and not refresh and not self._is_stale(meta):
            return pd.read_parquet(cache)

        url = ATP_ELO_URL if tour == "atp" else WTA_ELO_URL
        logger.info("Fetching Tennis Abstract %s Elo from %s", tour.upper(), url)
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=45)
        response.raise_for_status()

        tables = pd.read_html(StringIO(response.text))
        df = self._pick_table(tables)
        df = self._normalize_dataframe(df, tour=tour)
        df.to_parquet(cache, index=False)
        meta.write_text(datetime.now().isoformat(), encoding="utf-8")
        logger.info("Saved %d %s Elo rows to %s", len(df), tour.upper(), cache)
        return df

    def load(self, tour: str = "atp", refresh: bool = False) -> pd.DataFrame:
        cache = self._cache_path(tour)
        if cache.exists() and not refresh:
            return pd.read_parquet(cache)
        try:
            return self.download(tour=tour, refresh=True)
        except Exception as exc:
            if cache.exists():
                logger.warning("Tennis Abstract fetch failed, using cache: %s", exc)
                return pd.read_parquet(cache)
            raise

    def lookup(self, player: str, tour: str = "atp") -> Optional[Dict]:
        df = self.load(tour=tour)
        return self._lookup_in_df(df, player)

    def apply_to_context(
        self,
        ctx: Dict,
        player1: str,
        player2: str,
        surface: str = "hard",
        tour: str = "atp",
    ) -> Dict:
        """Overlay Tennis Abstract Elo onto match context (primary over internal Elo)."""
        surface = (surface or "hard").lower()
        surf_col = SURFACE_ELO_COL.get(surface, "hElo")

        for prefix, player in (("player1", player1), ("player2", player2)):
            row = self.lookup(player, tour=tour)
            if not row:
                continue

            overall = row.get("elo")
            surface_elo = row.get(surf_col.lower()) or row.get(surf_col)
            if overall:
                ctx[f"{prefix}_elo"] = float(overall)
                ctx[f"{prefix}_elo_source"] = "tennis_abstract"
            if surface_elo and not pd.isna(surface_elo):
                ctx[f"{prefix}_surface_elo"] = float(surface_elo)
                ctx[f"{prefix}_surface_elo_source"] = "tennis_abstract"
            if row.get("atp_rank"):
                ctx[f"{prefix}_rank"] = int(row["atp_rank"])
            ctx[f"{prefix}_ta_elo"] = overall
            ctx[f"{prefix}_ta_surface_elo"] = surface_elo

        if ctx.get("player1_elo") and ctx.get("player2_elo"):
            ctx["elo_diff"] = ctx["player1_elo"] - ctx["player2_elo"]
        if ctx.get("player1_surface_elo") and ctx.get("player2_surface_elo"):
            ctx["surface_elo_diff"] = ctx["player1_surface_elo"] - ctx["player2_surface_elo"]
        ctx["elo_provider"] = "tennis_abstract"
        return ctx

    def summary(self, tour: str = "atp") -> Dict:
        try:
            df = self.load(tour=tour)
        except Exception as exc:
            return {"available": False, "error": str(exc)}
        meta = self._meta_path(tour)
        updated = meta.read_text(encoding="utf-8").strip() if meta.exists() else ""
        return {
            "available": True,
            "tour": tour,
            "players": len(df),
            "updated_at": updated,
            "stale": self._is_stale(meta),
            "source": ATP_ELO_URL if tour == "atp" else WTA_ELO_URL,
        }

    def _pick_table(self, tables: List[pd.DataFrame]) -> pd.DataFrame:
        for table in tables:
            cols = {str(c).replace("\xa0", " ").strip().lower() for c in table.columns}
            if "player" in cols and "elo" in cols:
                return table.copy()
        raise ValueError("No Elo table found on Tennis Abstract page")

    def _normalize_dataframe(self, df: pd.DataFrame, tour: str) -> pd.DataFrame:
        df = df.copy()
        df.columns = [str(c).replace("\xa0", " ").strip() for c in df.columns]
        rename = {
            "ATP Rank": "atp_rank",
            "Elo Rank": "elo_rank",
            "Peak Elo": "peak_elo",
            "Peak Month": "peak_month",
            "Log diff": "log_diff",
        }
        df = df.rename(columns=rename)

        df["player"] = df["Player"].astype(str).str.replace("\xa0", " ", regex=False).str.strip()
        df["player_key"] = df["player"].map(_normalize_name)
        df["tour"] = tour

        for col in ("Elo", "hElo", "cElo", "gElo", "peak_elo", "atp_rank", "elo_rank"):
            if col in df.columns:
                df[col.lower()] = pd.to_numeric(df[col], errors="coerce")

        keep = [
            "player", "player_key", "tour", "age",
            "elo", "elo_rank", "helo", "celo", "gelo",
            "atp_rank", "peak_elo", "peak_month", "log_diff",
        ]
        keep = [c for c in keep if c in df.columns]
        return df[keep].drop_duplicates(subset=["player_key"], keep="first")

    def _lookup_in_df(self, df: pd.DataFrame, player: str) -> Optional[Dict]:
        key = _normalize_name(player)
        exact = df[df["player_key"] == key]
        if not exact.empty:
            return exact.iloc[0].to_dict()

        parts = key.split()
        if parts:
            last = parts[-1]
            matches = df[df["player_key"].str.endswith(f" {last}") | df["player_key"].str.endswith(last)]
            if len(matches) == 1:
                return matches.iloc[0].to_dict()
            if len(matches) > 1:
                for _, row in matches.iterrows():
                    if all(p in row["player_key"] for p in parts):
                        return row.to_dict()

        fuzzy = df[df["player"].str.contains(player.split()[-1], case=False, na=False)]
        if len(fuzzy) == 1:
            return fuzzy.iloc[0].to_dict()
        return None

    def _cache_path(self, tour: str) -> Path:
        return self.cache_dir / f"{tour}_elo.parquet"

    def _meta_path(self, tour: str) -> Path:
        return self.cache_dir / f"{tour}_elo_updated.txt"

    def _is_stale(self, meta_path: Path) -> bool:
        if not meta_path.exists():
            return True
        try:
            updated = datetime.fromisoformat(meta_path.read_text(encoding="utf-8").strip())
            return datetime.now() - updated > timedelta(days=CACHE_MAX_AGE_DAYS)
        except ValueError:
            return True


def _normalize_name(name: str) -> str:
    text = re.sub(r"\s+", " ", str(name).strip().lower())
    text = text.replace(".", "")
    return text