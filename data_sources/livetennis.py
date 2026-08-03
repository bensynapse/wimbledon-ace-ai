"""
Live Tennis API — optional source for upcoming ATP/WTA fixtures.

  https://livetennisapi.com  ·  docs https://docs.livetennisapi.com

Opt-in. Without LIVETENNIS_API_KEY the source reports `enabled = False` and
every call returns an empty list, so the collector falls through to whatever
it used before. Nothing here replaces an existing source.

Why it exists: fixtures are the one input the pipeline cannot run without, and
today they come from The Odds API or api-tennis.com. If either is out of quota
on a match day there is no upcoming card to price. This is one more place to
ask.

Scope is deliberately narrow — upcoming fixtures only:

  * Rankings are not exposed as a leaderboard by this API (its /rankings route
    resolves specific player ids), so `get_standings` is not implemented and
    the existing GitHub/Kaggle standings paths are untouched.
  * Odds are not mapped. This API's /markets prices are prediction-market
    probabilities in [0,1], not decimal bookmaker odds; feeding them into a
    value model built against bookmaker prices would be comparing two
    different things, so the odds path is left entirely to The Odds API and
    api-tennis.com.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import requests

from config import API_KEYS, DATA_SOURCES

logger = logging.getLogger(__name__)

LIVETENNIS_API_BASE = DATA_SOURCES.get(
    "livetennis_api", "https://api.livetennisapi.com/api/public/v1"
)

# `tour` filter vocabulary accepted by the API.
TOUR_KEYS = {
    "atp": "atp",
    "wta": "wta",
}

# Provider caps `limit` at 200; page until meta.has_more clears.
PAGE_LIMIT = 200
MAX_PAGES = 10


class LiveTennisSource:
    """Upcoming fixtures from the Live Tennis API. Inert without a key."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = (api_key or API_KEYS.get("livetennis_api") or "").strip()

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _get(self, path: str, params: Dict) -> Dict:
        # Key travels in a header, not the query string — a URL ends up in
        # logs, history and referrers.
        response = requests.get(
            f"{LIVETENNIS_API_BASE}{path}",
            params=params,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def _iter_matches(self, status: str, tour: Optional[str]) -> List[Dict]:
        """Page through GET /matches for one lifecycle status."""
        matches: List[Dict] = []

        for page in range(MAX_PAGES):
            params = {
                "status": status,
                "limit": PAGE_LIMIT,
                "offset": page * PAGE_LIMIT,
            }
            if tour:
                params["tour"] = tour

            payload = self._get("/matches", params)
            data = payload.get("data") or []
            matches.extend(data)

            if not data or not (payload.get("meta") or {}).get("has_more"):
                break

        return matches

    def get_upcoming_fixtures(
        self,
        tour: str,
        days_ahead: int = 2,
    ) -> List[Dict]:
        """Upcoming singles fixtures within the next `days_ahead` days.

        Returns the same dict shape as the other upcoming-fixture paths in
        data_collector. `surface` may be empty when the API does not know it —
        the caller infers it from the tournament name in that case, rather than
        this source guessing.
        """
        if not self.enabled:
            return []

        tour_key = TOUR_KEYS.get(tour.lower())
        if tour_key is None:
            logger.warning("Live Tennis API: unsupported tour %r", tour)
            return []

        try:
            matches = self._iter_matches("upcoming", tour_key)
        except Exception as exc:
            logger.warning("Live Tennis API fixtures failed: %s", exc)
            return []

        cutoff = date.today() + timedelta(days=days_ahead)
        fixtures: List[Dict] = []

        for match in matches:
            if match.get("is_doubles"):
                continue

            scheduled = match.get("scheduled_time") or ""
            match_date = _parse_date(scheduled)
            if match_date is None or match_date > cutoff:
                continue

            players = match.get("players") or {}
            p1 = players.get("p1") or {}
            p2 = players.get("p2") or {}

            name1 = (p1.get("name") or "").strip()
            name2 = (p2.get("name") or "").strip()
            if not name1 or not name2:
                continue

            fixtures.append({
                "match_id": str(match.get("id", "")),
                "date": match_date.isoformat(),
                "time": scheduled[11:16] if len(scheduled) >= 16 else "",
                "player1": name1,
                "player2": name2,
                "player1_key": str(p1.get("id", "")),
                "player2_key": str(p2.get("id", "")),
                "tournament": match.get("tournament") or "",
                "round": match.get("round") or "",
                # 'hard' | 'clay' | 'grass' upstream, or None when unknown.
                "surface": match.get("surface") or "",
                "status": match.get("status") or "",
            })

        logger.info(
            "Live Tennis API: %d upcoming %s fixtures", len(fixtures), tour.upper()
        )
        return fixtures


def _parse_date(timestamp: str) -> Optional[date]:
    """Date part of a UTC ISO 8601 timestamp, or None if unparseable."""
    if not timestamp:
        return None
    try:
        return datetime.strptime(timestamp[:10], "%Y-%m-%d").date()
    except ValueError:
        return None
