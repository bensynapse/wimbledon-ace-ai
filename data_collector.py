"""
WimbledonAce AI — Data Collector
Live & historical ATP/WTA data via api-tennis.com and The Odds API.
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

from config import (
    API_KEYS,
    DATA_SOURCE,
    DATA_SOURCES,
    PATHS,
    TOURS,
)
from data_sources.github_tennis import GitHubTennisSource
from data_sources.kaggle_tennis import KaggleTennisSource

logger = logging.getLogger(__name__)

TENNIS_API_BASE = "https://api.api-tennis.com/tennis/"
EVENT_TYPE_KEYS = {
    "atp": "265",  # Atp Singles
    "wta": "266",  # Wta Singles
}


class TennisAPIClient:
    """Client for api-tennis.com — primary match & results data source."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = (api_key or API_KEYS.get("tennis_api") or "").strip()

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _get(self, params: Dict) -> Dict:
        if not self.enabled:
            raise ValueError(
                "TENNIS_API_KEY missing — set in config.py or env. "
                "Sign up at https://api-tennis.com/"
            )
        payload = {"APIkey": self.api_key, **params}
        response = requests.get(TENNIS_API_BASE, params=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        if not data.get("success"):
            raise RuntimeError(f"Tennis API error: {data}")
        return data

    def get_fixtures(
        self,
        date_start: str,
        date_stop: str,
        tour: Optional[str] = None,
    ) -> List[Dict]:
        params = {
            "method": "get_fixtures",
            "date_start": date_start,
            "date_stop": date_stop,
        }
        if tour:
            params["event_type_key"] = EVENT_TYPE_KEYS[tour.lower()]
        result = self._get(params)
        return result.get("result", [])

    def get_odds(
        self,
        date_start: str,
        date_stop: str,
        tour: Optional[str] = None,
    ) -> Dict:
        params = {
            "method": "get_odds",
            "date_start": date_start,
            "date_stop": date_stop,
        }
        if tour:
            params["event_type_key"] = EVENT_TYPE_KEYS[tour.lower()]
        result = self._get(params)
        return result.get("result", {})

    def get_standings(self, tour: str) -> List[Dict]:
        result = self._get({
            "method": "get_standings",
            "event_type": tour.upper(),
        })
        return result.get("result", [])


class OddsAPIClient:
    """Client for the-odds-api.com tennis markets."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = (api_key or API_KEYS.get("odds_api") or "").strip()
        self.base_url = DATA_SOURCES["odds_api"]

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def get_match_odds(self, tour: str, regions: str = "eu") -> List[Dict]:
        if not self.enabled:
            return []

        sport_key = TOURS[tour.lower()]["odds_api_key"]
        url = f"{self.base_url}/sports/{sport_key}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": "h2h",
            "oddsFormat": "decimal",
        }
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()


class HistoricalDataCollector:
    """Download and cache historical match results."""

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir or PATHS["data_dir"]
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        self._client: Optional[TennisAPIClient] = None
        self.github = GitHubTennisSource()
        self.kaggle = KaggleTennisSource()

    @property
    def client(self) -> TennisAPIClient:
        if self._client is None:
            self._client = TennisAPIClient()
        return self._client

    def _cache_path(self, tour: str) -> str:
        return f"{self.data_dir}{tour.lower()}_matches.parquet"

    def download_tour(
        self,
        tour: str,
        start_year: int = 2018,
        end_year: Optional[int] = None,
        refresh: bool = False,
    ) -> pd.DataFrame:
        cache = Path(self._cache_path(tour))
        if cache.exists() and not refresh:
            logger.info("Loading cached data from %s", cache)
            return pd.read_parquet(cache)

        if tour.lower() == "wta":
            try:
                df = self.kaggle.download_tour(tour, start_year, end_year, refresh)
                df.to_parquet(cache, index=False)
                return df
            except Exception as exc:
                logger.warning("Kaggle WTA source failed: %s", exc)

        if DATA_SOURCE == "kaggle":
            try:
                df = self.kaggle.download_tour(tour, start_year, end_year, refresh)
                df.to_parquet(cache, index=False)
                return df
            except Exception as exc:
                logger.warning("Kaggle source failed: %s", exc)

        if DATA_SOURCE == "github" or not self.client.enabled:
            try:
                df = self.github.download_tour(tour, start_year, end_year, refresh)
                df.to_parquet(cache, index=False)
                return df
            except Exception as exc:
                logger.warning("GitHub ATP source failed: %s", exc)
                if cache.exists():
                    return pd.read_parquet(cache)
                if not self.client.enabled:
                    raise ValueError(
                        "GitHub download failed and no TENNIS_API_KEY set. "
                        "For WTA: kaggle datasets download -d guillemservera/tennis"
                    ) from exc

        if not self.client.enabled:
            if cache.exists():
                logger.warning("No TENNIS_API_KEY — using cached data only")
                return pd.read_parquet(cache)
            raise ValueError(
                "No cached data and TENNIS_API_KEY missing. "
                "Use GitHub source (default) or set TENNIS_API_KEY"
            )

        end_year = end_year or datetime.now().year
        frames: List[pd.DataFrame] = []

        for year in range(start_year, end_year + 1):
            logger.info("Fetching %s %s fixtures...", tour.upper(), year)
            year_frames = []
            for month in range(1, 13):
                start = date(year, month, 1)
                if month == 12:
                    stop = date(year, 12, 31)
                else:
                    stop = date(year, month + 1, 1) - timedelta(days=1)

                try:
                    fixtures = self.client.get_fixtures(
                        start.isoformat(),
                        stop.isoformat(),
                        tour=tour,
                    )
                    df = self._fixtures_to_df(fixtures, tour)
                    if not df.empty:
                        year_frames.append(df)
                except Exception as exc:
                    logger.warning("Failed %s-%02d: %s", year, month, exc)
                time.sleep(0.3)

            if year_frames:
                frames.append(pd.concat(year_frames, ignore_index=True))

        if not frames:
            raise RuntimeError(f"No historical data downloaded for {tour}")

        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(subset=["match_id"]).sort_values("date")
        df.to_parquet(cache, index=False)
        logger.info("Saved %d matches to %s", len(df), cache)
        return df

    def _fixtures_to_df(self, fixtures: List[Dict], tour: str) -> pd.DataFrame:
        rows = []
        singles_label = "Atp Singles" if tour.lower() == "atp" else "Wta Singles"

        for fix in fixtures:
            if fix.get("event_type_type") != singles_label:
                continue
            if fix.get("event_status") != "Finished":
                continue
            if fix.get("event_winner") not in ("First Player", "Second Player"):
                continue

            sets_p1, sets_p2, total_sets = _parse_set_score(fix)
            rows.append({
                "match_id": str(fix.get("event_key", "")),
                "date": fix.get("event_date"),
                "player1": fix.get("event_first_player", "").strip(),
                "player2": fix.get("event_second_player", "").strip(),
                "player1_key": str(fix.get("first_player_key", "")),
                "player2_key": str(fix.get("second_player_key", "")),
                "winner_is_player1": fix.get("event_winner") == "First Player",
                "tournament": fix.get("tournament_name", ""),
                "round": fix.get("tournament_round", ""),
                "surface": _infer_surface(fix),
                "tour": tour.lower(),
                "sets_player1": sets_p1,
                "sets_player2": sets_p2,
                "total_sets": total_sets,
                "minutes_played": total_sets * 52,
            })

        return pd.DataFrame(rows)


class LiveFixtureCollector:
    """Fetch upcoming/live fixtures and odds for predictions."""

    def __init__(self):
        self._tennis: Optional[TennisAPIClient] = None
        self.odds_api = OddsAPIClient()
        self.github = GitHubTennisSource()

    @property
    def tennis(self) -> TennisAPIClient:
        if self._tennis is None:
            self._tennis = TennisAPIClient()
        return self._tennis

    def get_upcoming_fixtures(
        self,
        tour: str,
        days_ahead: int = 2,
    ) -> List[Dict]:
        if self.odds_api.enabled:
            fixtures = self._fixtures_from_odds_api(tour)
            if fixtures:
                logger.info("Loaded %d fixtures from The Odds API", len(fixtures))
                return fixtures

        if not self.tennis.enabled:
            logger.warning(
                "No ODDS_API_KEY or TENNIS_API_KEY — using demo fixtures. "
                "Set ODDS_API_KEY for live upcoming matches."
            )
            return get_demo_fixtures(tour)

        today = date.today()
        stop = today + timedelta(days=days_ahead)
        fixtures = self.tennis.get_fixtures(
            today.isoformat(),
            stop.isoformat(),
            tour=tour,
        )

        singles_label = "Atp Singles" if tour.lower() == "atp" else "Wta Singles"
        upcoming = []
        for fix in fixtures:
            if fix.get("event_type_type") != singles_label:
                continue
            if fix.get("event_status") == "Finished":
                continue
            upcoming.append({
                "match_id": str(fix.get("event_key", "")),
                "date": fix.get("event_date", ""),
                "time": fix.get("event_time", ""),
                "player1": fix.get("event_first_player", "").strip(),
                "player2": fix.get("event_second_player", "").strip(),
                "player1_key": str(fix.get("first_player_key", "")),
                "player2_key": str(fix.get("second_player_key", "")),
                "tournament": fix.get("tournament_name", ""),
                "round": fix.get("tournament_round", ""),
                "surface": _infer_surface(fix),
                "status": fix.get("event_status", ""),
            })
        return upcoming

    def attach_odds(self, fixtures: List[Dict], tour: str) -> List[Dict]:
        tennis_odds = {}
        today = date.today().isoformat()
        try:
            tennis_odds = self.tennis.get_odds(today, today, tour=tour)
        except Exception as exc:
            logger.warning("Tennis API odds failed: %s", exc)

        odds_api_events = []
        if self.odds_api.enabled:
            try:
                odds_api_events = self.odds_api.get_match_odds(tour)
            except Exception as exc:
                logger.warning("Odds API failed: %s", exc)

        enriched = []
        for fix in fixtures:
            odds_p1, odds_p2 = _extract_tennis_api_odds(
                tennis_odds.get(fix["match_id"], {}),
            )
            if not odds_p1 or not odds_p2:
                api_p1, api_p2 = _match_odds_api_event(
                    fix["player1"],
                    fix["player2"],
                    odds_api_events,
                )
                odds_p1 = odds_p1 or api_p1
                odds_p2 = odds_p2 or api_p2

            enriched.append({
                **fix,
                "odds_player1": odds_p1,
                "odds_player2": odds_p2,
            })
        return enriched

    def get_standings(self, tour: str) -> List[Dict]:
        if tour.lower() == "atp":
            standings = self.github.get_standings(tour)
            if standings:
                return standings
        if tour.lower() == "wta":
            standings = self.kaggle.get_standings(tour)
            if standings:
                return standings
        if self.tennis.enabled:
            return self.tennis.get_standings(tour)
        return []

    def _fixtures_from_odds_api(self, tour: str) -> List[Dict]:
        events = self.odds_api.get_match_odds(tour)
        fixtures = []
        for event in events:
            commence = event.get("commence_time", "")
            match_date = commence[:10] if commence else date.today().isoformat()
            home = event.get("home_team", "").strip()
            away = event.get("away_team", "").strip()
            if not home or not away:
                continue

            odds_p1, odds_p2 = 0.0, 0.0
            for bookmaker in event.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    if market.get("key") != "h2h":
                        continue
                    for outcome in market.get("outcomes", []):
                        try:
                            price = float(outcome["price"])
                        except (KeyError, TypeError, ValueError):
                            continue
                        if outcome.get("name") == home:
                            odds_p1 = max(odds_p1, price)
                        elif outcome.get("name") == away:
                            odds_p2 = max(odds_p2, price)

            fixtures.append({
                "match_id": event.get("id", ""),
                "date": match_date,
                "time": commence[11:16] if len(commence) > 16 else "",
                "player1": home,
                "player2": away,
                "tournament": _tournament_from_odds_event(event),
                "round": "",
                "surface": _surface_from_sport_key(tour, event),
                "status": "Scheduled",
                "odds_player1": odds_p1,
                "odds_player2": odds_p2,
            })
        return fixtures


def _tournament_from_odds_event(event: Dict) -> str:
    sport_title = event.get("sport_title", "")
    if "wimbledon" in sport_title.lower():
        return "Wimbledon"
    return sport_title or "ATP Tour"


def _surface_from_sport_key(tour: str, event: Dict) -> str:
    title = (event.get("sport_title", "") + event.get("home_team", "")).lower()
    if "wimbledon" in title or "grass" in title:
        return "grass"
    if "roland" in title or "french" in title or "clay" in title:
        return "clay"
    return "hard"


def _infer_surface(fixture: Dict) -> str:
    name = " ".join([
        fixture.get("tournament_name", ""),
        fixture.get("tournament_round", ""),
    ]).lower()
    if any(word in name for word in ("wimbledon", "queens", "halle", "grass")):
        return "grass"
    if any(word in name for word in ("roland garros", "french", "clay", "monte carlo", "rome", "madrid")):
        return "clay"
    if "carpet" in name:
        return "carpet"
    return "hard"


def _extract_tennis_api_odds(match_odds: Dict) -> Tuple[float, float]:
    home_away = match_odds.get("Home/Away", {})
    home = home_away.get("Home", {})
    away = home_away.get("Away", {})
    if not home or not away:
        return 0.0, 0.0

    def best_price(book_prices: Dict) -> float:
        values = []
        for raw in book_prices.values():
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                continue
        return max(values) if values else 0.0

    return best_price(home), best_price(away)


def _normalize_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _match_odds_api_event(
    player1: str,
    player2: str,
    events: List[Dict],
) -> Tuple[float, float]:
    p1_norm = _normalize_name(player1)
    p2_norm = _normalize_name(player2)

    for event in events:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        home_norm = _normalize_name(home)
        away_norm = _normalize_name(away)

        direct = (
            (p1_norm in home_norm or home_norm in p1_norm)
            and (p2_norm in away_norm or away_norm in p2_norm)
        )
        reverse = (
            (p1_norm in away_norm or away_norm in p1_norm)
            and (p2_norm in home_norm or home_norm in p2_norm)
        )
        if not (direct or reverse):
            continue

        best_home = 0.0
        best_away = 0.0
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    try:
                        price = float(outcome["price"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if outcome.get("name") == home:
                        best_home = max(best_home, price)
                    elif outcome.get("name") == away:
                        best_away = max(best_away, price)

        if direct:
            return best_home, best_away
        return best_away, best_home

    return 0.0, 0.0


def _parse_set_score(fixture: Dict) -> Tuple[int, int, int]:
    """Parse match set score from api-tennis fixture."""
    result = fixture.get("event_final_result", "") or ""
    parts = result.replace(" ", "").split("-")
    if len(parts) == 2:
        try:
            s1, s2 = int(parts[0]), int(parts[1])
            return s1, s2, s1 + s2
        except ValueError:
            pass

    scores = fixture.get("scores") or []
    if scores:
        n = len(scores)
        return n, n, n * 2 - 1

    return 2, 1, 3


def get_demo_fixtures(tour: str = "atp") -> List[Dict]:
    """Offline demo fixtures when API keys are not configured."""
    return [
        {
            "match_id": "demo_001",
            "date": date.today().isoformat(),
            "time": "14:00",
            "player1": "Alex de Minaur",
            "player2": "Flavio Cobolli",
            "tournament": "Wimbledon",
            "round": "Round of 16",
            "surface": "grass",
            "status": "Scheduled",
            "odds_player1": 1.25,
            "odds_player2": 4.10,
            "surface_elo_diff": 120,
            "player1_form": 0.72,
            "player2_form": 0.55,
            "player1_elo": 1980,
            "player2_elo": 1860,
            "player1_surface_elo": 1920,
            "player2_surface_elo": 1800,
            "player1_minutes_7d": 320,
            "player2_minutes_7d": 580,
            "player2_five_set_recent": True,
            "player2_context_flags": ["fatigue_medium"],
            "path_diff": "ADM had clean path; Cobolli faced heavier opposition",
            "temperature_c": 24,
        },
        {
            "match_id": "demo_002",
            "date": date.today().isoformat(),
            "time": "16:30",
            "player1": "Taylor Fritz",
            "player2": "Lorenzo Musetti",
            "tournament": "Wimbledon",
            "round": "Quarterfinal",
            "surface": "grass",
            "status": "Scheduled",
            "odds_player1": 1.61,
            "odds_player2": 2.45,
            "surface_elo_diff": 80,
            "player1_form": 0.68,
            "player2_form": 0.62,
            "player1_elo": 1950,
            "player2_elo": 1870,
            "temperature_c": 22,
        },
    ]