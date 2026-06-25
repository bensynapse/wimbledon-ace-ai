"""
Enhanced Data Collector Module
Gathers: injuries, suspensions, cards, referee stats, player stats,
formations, lineups, xG, head-to-head, and news sentiment.

Integrates with:
  - api-football.com (comprehensive football data)
  - football-data.org (historical match data)  
  - the-odds-api.com (live odds)
  - newsapi.org (news sentiment)
  - football-data.co.uk (ProphitBet-style historical CSVs)
"""

import requests
import pandas as pd
import numpy as np
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

# WC specific config (graceful if not present)
try:
    from config import WC_LEAGUE_ID, WC_SEASON, VENUES_2026
except Exception:
    WC_LEAGUE_ID = 1
    WC_SEASON = 2026
    VENUES_2026 = {}

# soccerdata primary (optional)
try:
    from scrapers import SoccerDataWrapper
    HAS_SCRAPERS_SD = True
except Exception:
    HAS_SCRAPERS_SD = False
    SoccerDataWrapper = None  # type: ignore

# StatsBomb for detailed WC event data (optional, from analytics-handbook)
try:
    from scrapers import StatsBombWCImporter
    HAS_STATSBOMB = True
except Exception:
    HAS_STATSBOMB = False
    StatsBombWCImporter = None  # type: ignore

# socceraction for VAEP / xT action valuation (optional)
try:
    from scrapers import SoccerActionProcessor
    HAS_SOCCERACTION = True
except Exception:
    HAS_SOCCERACTION = False
    SoccerActionProcessor = None  # type: ignore

# roboflow/sports for CV tracking & physical metrics from video (optional)
try:
    from scrapers import RoboflowSportsAnalyzer
    HAS_ROBOFLOW_SPORTS = True
except Exception:
    HAS_ROBOFLOW_SPORTS = False
    RoboflowSportsAnalyzer = None  # type: ignore

logger = logging.getLogger(__name__)


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class PlayerInfo:
    """Individual player data for a specific match."""
    name: str
    position: str
    team: str
    is_available: bool = True
    injury_type: Optional[str] = None
    yellow_cards_season: int = 0
    red_cards_season: int = 0
    is_suspended: bool = False
    goals_season: int = 0
    assists_season: int = 0
    shots_per_90: float = 0.0
    shots_on_target_per_90: float = 0.0
    key_passes_per_90: float = 0.0
    xg_per_90: float = 0.0
    xa_per_90: float = 0.0
    minutes_played: int = 0
    is_key_player: bool = False  # Top contributor to team


@dataclass
class TeamSquadStatus:
    """Full squad status for a team before a match."""
    team_name: str
    injured_players: List[PlayerInfo] = field(default_factory=list)
    suspended_players: List[PlayerInfo] = field(default_factory=list)
    doubtful_players: List[PlayerInfo] = field(default_factory=list)
    available_key_players: List[PlayerInfo] = field(default_factory=list)
    missing_key_players: List[PlayerInfo] = field(default_factory=list)
    expected_formation: Optional[str] = None
    squad_strength_score: float = 1.0  # 1.0 = full strength


@dataclass
class RefereeProfile:
    """Referee statistics relevant for match prediction."""
    name: str
    matches_officiated: int = 0
    avg_yellows_per_match: float = 0.0
    avg_reds_per_match: float = 0.0
    avg_fouls_per_match: float = 0.0
    avg_penalties_per_match: float = 0.0
    home_win_pct: float = 0.0
    away_win_pct: float = 0.0
    draw_pct: float = 0.0
    avg_goals_per_match: float = 0.0
    card_strictness_score: float = 0.5  # 0=lenient, 1=strict


@dataclass
class MatchContext:
    """Complete pre-match context for prediction."""
    match_id: str
    home_team: str
    away_team: str
    date: str
    league: str
    # Squad info
    home_squad: Optional[TeamSquadStatus] = None
    away_squad: Optional[TeamSquadStatus] = None
    # Referee
    referee: Optional[RefereeProfile] = None
    # H2H
    h2h_home_wins: int = 0
    h2h_away_wins: int = 0
    h2h_draws: int = 0
    h2h_avg_goals: float = 0.0
    h2h_matches: int = 0
    # Form features (from ProphitBet-style rolling stats)
    home_form_features: Dict[str, float] = field(default_factory=dict)
    away_form_features: Dict[str, float] = field(default_factory=dict)
    # Odds
    odds_home: float = 0.0
    odds_draw: float = 0.0
    odds_away: float = 0.0
    odds_over25: float = 0.0
    odds_under25: float = 0.0
    # News/momentum
    home_news_sentiment: float = 0.0  # -1 to 1
    away_news_sentiment: float = 0.0
    # Advanced metrics
    home_xg_for: float = 0.0
    home_xg_against: float = 0.0
    away_xg_for: float = 0.0
    away_xg_against: float = 0.0

    # WC / Tournament specific (new for WK bot)
    venue: Optional[str] = None
    stage: str = "group"  # group, r32, r16, qf, sf, final
    weather_summary: Optional[Dict] = None
    expected_crowd: Optional[int] = None
    crowd_factor: float = 1.0
    travel_days: int = 0
    climate_adapt_score: float = 1.0  # <1 = disadvantaged
    is_national_team: bool = True


# ============================================================
# API-FOOTBALL COLLECTOR
# ============================================================

class APIFootballCollector:
    """
    Collects data from api-football.com (RapidAPI).
    Covers: injuries, lineups, player stats, referee, fixtures, H2H, xG.
    """

    BASE_URL = "https://v3.football.api-sports.io"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "x-apisports-key": api_key
        }
        self._rate_limit_remaining = 100
        self._cache: Dict[str, Any] = {}

    def _request(self, endpoint: str, params: dict) -> Optional[dict]:
        """Make API request with rate limiting and caching."""
        cache_key = f"{endpoint}_{json.dumps(params, sort_keys=True)}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            url = f"{self.BASE_URL}/{endpoint}"
            resp = requests.get(url, headers=self.headers, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            if data.get("errors"):
                logger.warning(f"API error for {endpoint}: {data['errors']}")
                return None

            self._cache[cache_key] = data
            remaining = resp.headers.get("x-ratelimit-requests-remaining", "?")
            logger.debug(f"API call {endpoint} - remaining: {remaining}")

            return data
        except Exception as e:
            logger.error(f"API request failed for {endpoint}: {e}")
            return None

    # ------ INJURIES & SUSPENSIONS ------

    def get_injuries(self, league_id: int, season: int,
                     fixture_id: Optional[int] = None) -> List[PlayerInfo]:
        """Get current injuries for a league/fixture."""
        params = {"league": league_id, "season": season}
        if fixture_id:
            params["fixture"] = fixture_id

        data = self._request("injuries", params)
        if not data or not data.get("response"):
            return []

        injuries = []
        for entry in data["response"]:
            player = entry.get("player", {})
            team = entry.get("team", {})
            injuries.append(PlayerInfo(
                name=player.get("name", "Unknown"),
                position=player.get("type", "Unknown"),
                team=team.get("name", "Unknown"),
                is_available=False,
                injury_type=entry.get("player", {}).get("reason", "Injured")
            ))
        return injuries

    # ------ PLAYER STATISTICS ------

    def get_player_stats(self, team_id: int, season: int,
                         league_id: int) -> List[PlayerInfo]:
        """Get detailed player statistics for a team."""
        params = {"team": team_id, "season": season, "league": league_id}
        data = self._request("players", params)
        if not data or not data.get("response"):
            return []

        players = []
        for entry in data["response"]:
            p = entry.get("player", {})
            stats_list = entry.get("statistics", [{}])
            stats = stats_list[0] if stats_list else {}

            games = stats.get("games", {})
            goals_data = stats.get("goals", {})
            cards = stats.get("cards", {})
            shots = stats.get("shots", {})
            passes = stats.get("passes", {})

            minutes = games.get("minutes") or 0
            per_90_factor = 90.0 / max(minutes, 1) if minutes > 0 else 0

            players.append(PlayerInfo(
                name=p.get("name", "Unknown"),
                position=games.get("position", "Unknown"),
                team=stats.get("team", {}).get("name", "Unknown"),
                is_available=True,
                yellow_cards_season=(cards.get("yellow") or 0),
                red_cards_season=(cards.get("red") or 0),
                goals_season=(goals_data.get("total") or 0),
                assists_season=(goals_data.get("assists") or 0),
                shots_per_90=(shots.get("total") or 0) * per_90_factor,
                shots_on_target_per_90=(shots.get("on") or 0) * per_90_factor,
                key_passes_per_90=(passes.get("key") or 0) * per_90_factor,
                minutes_played=minutes
            ))
        return players

    # ------ REFEREE DATA ------

    def get_fixture_referee(self, fixture_id: int) -> Optional[RefereeProfile]:
        """Get referee assigned to a fixture plus their historical stats."""
        data = self._request("fixtures", {"id": fixture_id})
        if not data or not data.get("response"):
            return None

        fixture = data["response"][0]
        ref_name = fixture.get("fixture", {}).get("referee")
        if not ref_name:
            return None

        # Clean referee name (sometimes has country suffix)
        ref_name = ref_name.split(",")[0].strip()
        return RefereeProfile(name=ref_name)

    def enrich_referee_stats(self, referee_name: str, league_id: int,
                             season: int) -> Optional[RefereeProfile]:
        """
        Build referee profile from historical fixtures they officiated.
        API-Football doesn't have a dedicated referee endpoint, so we
        search fixtures and filter.
        """
        # This requires iterating through fixtures - expensive but valuable
        # In practice, cache this heavily or use a pre-built database
        return RefereeProfile(name=referee_name)

    # ------ HEAD TO HEAD ------

    def get_h2h(self, home_team_id: int, away_team_id: int,
                last_n: int = 10) -> Dict[str, Any]:
        """Get head-to-head record between two teams."""
        data = self._request("fixtures/headtohead", {
            "h2h": f"{home_team_id}-{away_team_id}",
            "last": last_n
        })
        if not data or not data.get("response"):
            return {"home_wins": 0, "away_wins": 0, "draws": 0,
                    "avg_goals": 0, "matches": 0}

        home_wins = away_wins = draws = total_goals = 0
        for match in data["response"]:
            home_goals = match["goals"]["home"] or 0
            away_goals = match["goals"]["away"] or 0
            total_goals += home_goals + away_goals

            home_id = match["teams"]["home"]["id"]
            if home_goals > away_goals:
                if home_id == home_team_id:
                    home_wins += 1
                else:
                    away_wins += 1
            elif away_goals > home_goals:
                if home_id == home_team_id:
                    away_wins += 1
                else:
                    home_wins += 1
            else:
                draws += 1

        n_matches = len(data["response"])
        return {
            "home_wins": home_wins,
            "away_wins": away_wins,
            "draws": draws,
            "avg_goals": total_goals / max(n_matches, 1),
            "matches": n_matches
        }

    # ------ LINEUPS & FORMATIONS ------

    def get_predicted_lineups(self, fixture_id: int) -> Dict[str, Any]:
        """Get predicted lineups and formations for a fixture."""
        data = self._request("predictions", {"fixture": fixture_id})
        if not data or not data.get("response"):
            return {}

        prediction = data["response"][0]
        lineups = prediction.get("lineups", {})

        result = {}
        for lineup in lineups if isinstance(lineups, list) else []:
            team_name = lineup.get("team", {}).get("name", "Unknown")
            result[team_name] = {
                "formation": lineup.get("formation"),
                "starting_xi": [
                    p.get("player", {}).get("name")
                    for p in lineup.get("startXI", [])
                ],
                "substitutes": [
                    p.get("player", {}).get("name")
                    for p in lineup.get("substitutes", [])
                ]
            }
        return result

    # ------ xG DATA ------

    def get_team_xg(self, team_id: int, league_id: int,
                    season: int, last_n: int = 10) -> Dict[str, float]:
        """Calculate rolling xG stats for a team from recent fixtures."""
        data = self._request("fixtures", {
            "team": team_id, "league": league_id,
            "season": season, "last": last_n,
            "status": "FT"
        })
        if not data or not data.get("response"):
            return {"xg_for": 0.0, "xg_against": 0.0}

        xg_for_total = 0.0
        xg_against_total = 0.0
        count = 0

        for match in data["response"]:
            home_id = match["teams"]["home"]["id"]
            # xG might not always be available
            stats = match.get("statistics", [])
            # Parse from fixture statistics if available
            for stat_block in stats:
                team_stat_id = stat_block.get("team", {}).get("id")
                for stat in stat_block.get("statistics", []):
                    if stat.get("type") == "expected_goals":
                        xg_val = float(stat.get("value") or 0)
                        if team_stat_id == team_id:
                            xg_for_total += xg_val
                        else:
                            xg_against_total += xg_val
            count += 1

        n = max(count, 1)
        return {
            "xg_for": round(xg_for_total / n, 2),
            "xg_against": round(xg_against_total / n, 2)
        }

    # ------ WORLD CUP 2026 SPECIFIC (league=1, season=2026) ------

    def get_wc_teams(self) -> List[Dict]:
        """Get all 48 national teams for WC."""
        data = self._request("teams", {"league": WC_LEAGUE_ID, "season": WC_SEASON})
        if not data or not data.get("response"):
            return []
        return [{"id": t["team"]["id"], "name": t["team"]["name"], "code": t["team"].get("code")}
                for t in data["response"] if t.get("team")]

    def get_wc_squads(self, team_id: Optional[int] = None) -> List[PlayerInfo]:
        """Get current WC squads via players endpoint. Returns PlayerInfo list (basic)."""
        params = {"league": WC_LEAGUE_ID, "season": WC_SEASON}
        if team_id:
            params["team"] = team_id
        data = self._request("players", params)
        if not data or not data.get("response"):
            return []
        players = []
        for entry in data["response"]:
            p = entry.get("player", {})
            stats_list = entry.get("statistics", [{}])
            stats = stats_list[0] if stats_list else {}
            games = stats.get("games", {})
            goals = stats.get("goals", {})
            cards = stats.get("cards", {})
            players.append(PlayerInfo(
                name=p.get("name", "Unknown"),
                position=games.get("position", "Unknown"),
                team=stats.get("team", {}).get("name", "Unknown"),
                is_available=True,
                goals_season=goals.get("total") or 0,
                assists_season=goals.get("assists") or 0,
                yellow_cards_season=cards.get("yellow") or 0,
                red_cards_season=cards.get("red") or 0,
            ))
        return players

    def get_wc_fixtures(self, round_name: Optional[str] = None) -> List[Dict]:
        params = {"league": WC_LEAGUE_ID, "season": WC_SEASON}
        if round_name:
            params["round"] = round_name
        data = self._request("fixtures", params)
        return data.get("response", []) if data else []

    def enrich_wc_referee(self, ref_name: str) -> Optional[RefereeProfile]:
        """Stub + basic profile. In prod: aggregate historical from int'l fixtures."""
        if not ref_name:
            return None
        clean = ref_name.split(",")[0].strip()
        return RefereeProfile(
            name=clean,
            avg_yellows_per_match=4.2,
            card_strictness_score=0.55,
        )


# ============================================================
# ODDS COLLECTOR
# ============================================================

class OddsCollector:
    """Collects live odds from The Odds API."""

    BASE_URL = "https://api.the-odds-api.com/v4"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_match_odds(self, sport: str = "soccer_epl",
                       regions: str = "eu",
                       markets: str = "h2h,totals") -> List[Dict]:
        """Get current odds for upcoming matches."""
        try:
            resp = requests.get(
                f"{self.BASE_URL}/sports/{sport}/odds",
                params={
                    "apiKey": self.api_key,
                    "regions": regions,
                    "markets": markets,
                    "oddsFormat": "decimal"
                },
                timeout=15
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Odds API error: {e}")
            return []

    def find_best_odds(self, odds_data: List[Dict],
                       home_team: str,
                       away_team: str) -> Dict[str, float]:
        """Find best available odds across bookmakers for a specific match."""
        best = {"home": 0, "draw": 0, "away": 0, "over25": 0, "under25": 0}

        for event in odds_data:
            h = event.get("home_team", "").lower()
            a = event.get("away_team", "").lower()

            if home_team.lower() not in h and away_team.lower() not in a:
                continue

            for bookmaker in event.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    if market["key"] == "h2h":
                        for outcome in market.get("outcomes", []):
                            name = outcome["name"].lower()
                            price = outcome["price"]
                            if "home" in name or home_team.lower() in name:
                                best["home"] = max(best["home"], price)
                            elif "away" in name or away_team.lower() in name:
                                best["away"] = max(best["away"], price)
                            elif "draw" in name:
                                best["draw"] = max(best["draw"], price)
                    elif market["key"] == "totals":
                        for outcome in market.get("outcomes", []):
                            if outcome.get("point") == 2.5:
                                if outcome["name"] == "Over":
                                    best["over25"] = max(
                                        best["over25"], outcome["price"])
                                else:
                                    best["under25"] = max(
                                        best["under25"], outcome["price"])
        return best


# ============================================================
# NEWS SENTIMENT COLLECTOR
# ============================================================

class NewsSentimentCollector:
    """Collects and analyzes news sentiment for teams."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"

    def get_team_sentiment(self, team_name: str,
                           days_back: int = 3) -> float:
        """
        Get news sentiment for a team.
        Returns: float between -1 (very negative) and 1 (very positive).

        Uses simple keyword-based sentiment since we don't want heavy
        NLP dependencies. For production, integrate with a proper
        sentiment model.
        """
        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        try:
            resp = requests.get(
                f"{self.base_url}/everything",
                params={
                    "q": f'"{team_name}" football',
                    "from": from_date,
                    "language": "en",
                    "sortBy": "relevancy",
                    "pageSize": 20,
                    "apiKey": self.api_key
                },
                timeout=15
            )
            resp.raise_for_status()
            articles = resp.json().get("articles", [])
        except Exception as e:
            logger.warning(f"News API error for {team_name}: {e}")
            return 0.0

        if not articles:
            return 0.0

        # Simple keyword-based sentiment analysis
        positive_words = {
            "win", "victory", "excellent", "dominant", "strong", "unbeaten",
            "comeback", "brilliant", "superb", "confident", "surge", "boost",
            "signing", "return", "fit", "recovered", "impressive", "form"
        }
        negative_words = {
            "loss", "defeat", "injury", "injured", "out", "suspended",
            "crisis", "poor", "struggling", "doubt", "concern", "sack",
            "fired", "relegation", "ban", "missing", "absence", "setback",
            "red card", "fracture", "torn", "hamstring", "knee"
        }

        total_score = 0
        for article in articles:
            text = (
                (article.get("title") or "") + " " +
                (article.get("description") or "")
            ).lower()

            pos = sum(1 for w in positive_words if w in text)
            neg = sum(1 for w in negative_words if w in text)
            total = pos + neg
            if total > 0:
                total_score += (pos - neg) / total

        return round(total_score / max(len(articles), 1), 3)


# ============================================================
# HISTORICAL DATA COLLECTOR (ProphitBet-style)
# ============================================================

class HistoricalDataCollector:
    """
    Downloads historical match data from football-data.co.uk
    (same source as ProphitBet) and enriches with additional columns.
    """

    COLUMN_MAP = {
        "HomeTeam": "Home", "AwayTeam": "Away",
        "FTHG": "HG", "FTAG": "AG", "FTR": "Result",
        "AvgH": "Odds_H", "AvgD": "Odds_D", "AvgA": "Odds_A",
        "HST": "HST", "AST": "AST", "HC": "HC", "AC": "AC",
        "HS": "HS", "AS": "AS",     # Total shots
        "HF": "HF", "AF": "AF",     # Fouls committed
        "HY": "HY", "AY": "AY",     # Yellow cards
        "HR": "HR", "AR": "AR",     # Red cards
        "Referee": "Referee"          # Referee name
    }

    # Columns ProphitBet uses
    PROPHITBET_BASIC = [
        "Date", "Season", "Home", "Away", "HG", "AG", "Result",
        "HST", "AST", "HC", "AC"
    ]

    # EXTRA columns ProphitBet MISSES - we add these
    ENHANCED_COLUMNS = [
        "HS", "AS",       # Total shots (not just on target)
        "HF", "AF",       # Fouls
        "HY", "AY",       # Yellow cards
        "HR", "AR",       # Red cards
        "Referee",         # Referee name
        "Odds_H", "Odds_D", "Odds_A"  # Odds
    ]

    def download_league(self, league_code: str, country_code: str,
                        start_year: int = 2018) -> Optional[pd.DataFrame]:
        """
        Download all seasons for a league.
        E.g., league_code='E0' (Premier League), country_code='england'
        """
        from datetime import date
        dfs = []
        for year in range(start_year, date.today().year + 1):
            season_code = f"{str(year)[-2:]}{str(year + 1)[-2:]}"
            url = (f"https://www.football-data.co.uk/"
                   f"mmz4281/{season_code}/{league_code}.csv")
            try:
                df = pd.read_csv(url, on_bad_lines='skip')
                df["Season"] = year
                dfs.append(df)
                logger.info(f"Downloaded {league_code} season {year}")
            except Exception as e:
                logger.debug(f"No data for {league_code} {year}: {e}")
                continue

        if not dfs:
            return None

        combined = pd.concat(dfs, ignore_index=True)

        # Rename and select columns
        available = {c: self.COLUMN_MAP[c] for c in self.COLUMN_MAP
                     if c in combined.columns}
        combined = combined.rename(columns=available)

        # Keep all available columns
        keep_cols = ["Date", "Season"] + [
            v for v in available.values()
            if v not in ("Date", "Season")
        ]
        keep_cols = [c for c in keep_cols if c in combined.columns]
        combined = combined[keep_cols]

        # Parse dates
        combined["Date"] = pd.to_datetime(combined["Date"],
                                          dayfirst=True,
                                          errors="coerce")
        combined = combined.dropna(subset=["Date"])
        combined = combined.sort_values("Date").reset_index(drop=True)

        return combined


# ============================================================
# MASTER DATA AGGREGATOR
# ============================================================

class MatchDataAggregator:
    """
    Combines all data sources into a unified MatchContext object
    ready for feature engineering.
    """

    def __init__(
        self,
        api_football_key: str = "",
        odds_api_key: str = "",
        news_api_key: str = ""
    ):
        self.api_football = (
            APIFootballCollector(api_football_key) if api_football_key else None
        )
        self.odds = OddsCollector(odds_api_key) if odds_api_key else None
        self.news = NewsSentimentCollector(news_api_key) if news_api_key else None
        self.historical = HistoricalDataCollector()
        # soccerdata as primary advanced stats source (for WC + club)
        self.soccerdata = SoccerDataWrapper() if HAS_SCRAPERS_SD and SoccerDataWrapper else None
        # StatsBomb event data for granular WC analysis (player actions, shots, etc.)
        self.statsbomb = StatsBombWCImporter() if HAS_STATSBOMB and StatsBombWCImporter else None
        # socceraction VAEP/xT for advanced action-value player/team features
        self.socceraction = SoccerActionProcessor() if HAS_SOCCERACTION and SoccerActionProcessor else None
        # roboflow/sports CV for tracking-derived workload, speed, formations (from video)
        self.roboflow_sports = RoboflowSportsAnalyzer() if HAS_ROBOFLOW_SPORTS and RoboflowSportsAnalyzer else None
        # All gap data loaders (injuries, set-pieces, sentiment, CLV, WC patterns)
        self.injury_news = getattr(globals().get('scrapers', None), 'injury_news', None)
        self.set_piece = getattr(globals().get('scrapers', None), 'set_piece', None)
        self.sentiment = getattr(globals().get('scrapers', None), 'sentiment', None)
        self.clv_tracker = getattr(globals().get('scrapers', None), 'clv_tracker', None)
        self.wc_patterns = getattr(globals().get('scrapers', None), 'wc_patterns', None)
        # Coach tactical + detailed historical WC (jfjelstul) + recent this-year data
        self.coach = getattr(globals().get('scrapers', None), 'coach', None)
        self.jfjelstul = getattr(globals().get('scrapers', None), 'jfjelstul', None)
        if self.jfjelstul is None:
            # direct instantiate if available
            try:
                from scrapers import JfjelstulWCImporter
                self.jfjelstul = JfjelstulWCImporter()
            except Exception:
                self.jfjelstul = None
        if self.coach is None:
            try:
                from scrapers import CoachTacticalLoader
                self.coach = CoachTacticalLoader()
            except Exception:
                self.coach = None

    def build_match_context(
        self,
        fixture_id: int,
        home_team: str,
        away_team: str,
        home_team_id: int,
        away_team_id: int,
        league_id: int,
        season: int,
        date_str: str,
        league_name: str = ""
    ) -> MatchContext:
        """
        Build complete match context by querying all available data sources.
        """
        ctx = MatchContext(
            match_id=str(fixture_id),
            home_team=home_team,
            away_team=away_team,
            date=date_str,
            league=league_name
        )

        # --- API-Football data ---
        if self.api_football:
            # Injuries
            injuries = self.api_football.get_injuries(league_id, season,
                                                       fixture_id)
            home_injuries = [p for p in injuries
                            if p.team.lower() in home_team.lower()]
            away_injuries = [p for p in injuries
                            if p.team.lower() in away_team.lower()]

            # Player stats
            home_players = self.api_football.get_player_stats(
                home_team_id, season, league_id)
            away_players = self.api_football.get_player_stats(
                away_team_id, season, league_id)

            # Mark key players (top 5 by goals+assists)
            for players in [home_players, away_players]:
                players.sort(
                    key=lambda p: p.goals_season + p.assists_season,
                    reverse=True
                )
                for p in players[:5]:
                    p.is_key_player = True

            # Build squad status
            ctx.home_squad = self._build_squad_status(
                home_team, home_players, home_injuries)
            ctx.away_squad = self._build_squad_status(
                away_team, away_players, away_injuries)

            # Referee
            ref = self.api_football.get_fixture_referee(fixture_id)
            ctx.referee = ref

            # H2H
            h2h = self.api_football.get_h2h(home_team_id, away_team_id)
            ctx.h2h_home_wins = h2h["home_wins"]
            ctx.h2h_away_wins = h2h["away_wins"]
            ctx.h2h_draws = h2h["draws"]
            ctx.h2h_avg_goals = h2h["avg_goals"]
            ctx.h2h_matches = h2h["matches"]

            # xG
            home_xg = self.api_football.get_team_xg(
                home_team_id, league_id, season)
            away_xg = self.api_football.get_team_xg(
                away_team_id, league_id, season)
            ctx.home_xg_for = home_xg["xg_for"]
            ctx.home_xg_against = home_xg["xg_against"]
            ctx.away_xg_for = away_xg["xg_for"]
            ctx.away_xg_against = away_xg["xg_against"]

        # --- Odds ---
        if self.odds:
            odds_data = self.odds.get_match_odds()
            best = self.odds.find_best_odds(odds_data, home_team, away_team)
            ctx.odds_home = best["home"]
            ctx.odds_draw = best["draw"]
            ctx.odds_away = best["away"]
            ctx.odds_over25 = best["over25"]
            ctx.odds_under25 = best["under25"]

        # --- News sentiment ---
        if self.news:
            ctx.home_news_sentiment = self.news.get_team_sentiment(home_team)
            ctx.away_news_sentiment = self.news.get_team_sentiment(away_team)

        # --- WC ENHANCEMENTS (if league indicates WC) ---
        is_wc = (league_id == WC_LEAGUE_ID) or "world" in str(league_name).lower() or season == WC_SEASON
        ctx.is_national_team = is_wc

        if is_wc and self.api_football:
            # Try to attach venue from fixture if not passed
            if not ctx.venue:
                fix_data = self.api_football._request("fixtures", {"id": fixture_id})
                if fix_data and fix_data.get("response"):
                    fx = fix_data["response"][0]
                    ven = fx.get("fixture", {}).get("venue", {}) or {}
                    ctx.venue = ven.get("name") or ven.get("city")

            # Weather via config venue lookup (best effort)
            if ctx.venue and VENUES_2026:
                v = None
                for k, vv in VENUES_2026.items():
                    if ctx.venue and (k in ctx.venue.lower() or vv.get("city", "").lower() in ctx.venue.lower()):
                        v = vv
                        break
                if v:
                    ctx.weather_summary = {"altitude_m": v.get("alt_m"), "roof": v.get("roof_ac")}
                    # simple crowd proxy
                    ctx.expected_crowd = int(v.get("capacity", 50000) * 0.92)
                    ctx.crowd_factor = 1.03 if v.get("capacity", 0) > 60000 else 1.0

            # Enrich referee for WC refs
            if ctx.referee:
                ctx.referee = self.api_football.enrich_wc_referee(ctx.referee.name)

            # Quick national squad refresh (key players)
            try:
                wc_home = self.api_football.get_wc_squads(home_team_id)
                wc_away = self.api_football.get_wc_squads(away_team_id)
                if wc_home:
                    ctx.home_squad = self._build_squad_status(home_team, wc_home, [])
                if wc_away:
                    ctx.away_squad = self._build_squad_status(away_team, wc_away, [])
            except Exception:
                pass

            # Crude adaptation / travel (can be refined)
            ctx.travel_days = 4  # assume typical arrival
            ctx.climate_adapt_score = 0.95 if "mexico" in str(ctx.venue or "").lower() else 1.0

        # --- soccerdata enrichment (PRIMARY source for xG / per90 when available) ---
        if self.soccerdata and getattr(self.soccerdata, "has_soccerdata", False):
            try:
                sd_league = "INT-World Cup" if is_wc or "world" in str(league_name).lower() else None
                if sd_league:
                    team_xg_df = self.soccerdata.get_fbref_team_xg(league=sd_league, season=str(season) if season else "2022")
                    if not team_xg_df.empty and "team" in team_xg_df.columns:
                        for side, team_name in [("home", home_team), ("away", away_team)]:
                            mask = team_xg_df["team"].str.contains(team_name, case=False, na=False)
                            if mask.any():
                                row = team_xg_df[mask].iloc[0]
                                # Map common FBref / xG cols
                                xg_for = row.get("xG", row.get("Gls", 0)) or 0
                                if side == "home":
                                    if not ctx.home_xg_for:
                                        ctx.home_xg_for = float(xg_for) / max(1, float(row.get("90s", 1)) or 1)
                                else:
                                    if not ctx.away_xg_for:
                                        ctx.away_xg_for = float(xg_for) / max(1, float(row.get("90s", 1)) or 1)
                    # player level sample (for future workload)
                    pstats = self.soccerdata.get_fbref_player_stats(league=sd_league, season=str(season) if season else "2022")
                    if not pstats.empty:
                        ctx.__dict__.setdefault("sd_player_sample", pstats.head(5).to_dict("records"))

                    # WhoScored etc. enrichment (detailed ratings, shots, possession, cards)
                    try:
                        ws_data = self.soccerdata.get_whoscored_data(league=sd_league, season=str(season) if season else "2022")
                        if "error" not in ws_data:
                            ctx.__dict__.setdefault("whoscored_schedule", ws_data.get("schedule"))
                            pms = ws_data.get("player_match_stats")
                            if isinstance(pms, pd.DataFrame) and not pms.empty:
                                ctx.__dict__.setdefault("whoscored_player_sample", pms.head(8).to_dict("records"))
                    except Exception:
                        pass
                    # Quick ESPN / FotMob presence note for "etc"
                    for extra_src, method in [("espn", "get_espn_data"), ("fotmob", "get_fotmob_data")]:
                        try:
                            m = getattr(self.soccerdata, method, None)
                            if m:
                                res = m(league=sd_league, season=str(season) if season else "2022")
                                if "error" not in res:
                                    ctx.__dict__.setdefault(f"{extra_src}_available", True)
                        except Exception:
                            pass
            except Exception:
                pass

        # --- StatsBomb WC event enrichment (granular actions per analytics-handbook) ---
        if getattr(self, "statsbomb", None) and getattr(self.statsbomb, "has_statsbomb", False) and is_wc:
            try:
                wc_matches = self.statsbomb.get_wc_matches()
                if not wc_matches.empty:
                    mask = (wc_matches.get("home_team", pd.Series()).str.contains(home_team, case=False, na=False) |
                            wc_matches.get("away_team", pd.Series()).str.contains(home_team, case=False, na=False))
                    if mask.any():
                        sample_mid = int(wc_matches[mask].iloc[0]["match_id"])
                        hfeats = self.statsbomb.extract_team_event_features(sample_mid, home_team)
                        if hfeats:
                            ctx.__dict__.setdefault("statsbomb_event_sample", hfeats)
            except Exception:
                pass

        # --- socceraction VAEP / xT enrichment (action values for players) ---
        if getattr(self, "socceraction", None) and getattr(self.socceraction, "has_socceraction", False) and is_wc:
            try:
                # Use sample events if available in context or via statsbomb
                if ctx.home_squad or ctx.away_squad:
                    # Placeholder: in full impl convert events -> rate actions -> aggregate per player
                    ctx.__dict__.setdefault("vaep_note", "socceraction available for action valuation")
            except Exception:
                pass

        # --- roboflow/sports CV tracking enrichment (physical/workload from video) ---
        if getattr(self, "roboflow_sports", None) and getattr(self.roboflow_sports, "has_roboflow_sports", False) and is_wc:
            try:
                # If video available for match (highlights/broadcast), extract
                # For now, structure for when video_path provided
                if hasattr(ctx, "video_path") and ctx.video_path:
                    track_feats = self.roboflow_sports.extract_tracking_features(ctx.video_path)
                    ctx.__dict__.setdefault("tracking_workload", track_feats)
            except Exception:
                pass

        # --- eddwebster/football_analytics inspired: squad valuation & player similarity (TM values, clustering ideas) ---
        # Extend with real market value collection (Transfermarkt scrape or API-Football if available)
        # Useful for WC squad depth ("value" of bench) and comparing players to historical similar ones.
        if is_wc and (ctx.home_squad or ctx.away_squad):
            ctx.__dict__.setdefault("squad_valuation_note", "integrate TM market values for squad strength")

        # --- GRF RL sim enrichment (synthetic data / what-if from google-research/football) ---
        # Generate synthetic match features or "RL-expected" outcomes for data augmentation,
        # adaptation modeling (custom scenarios for heat/fatigue), or backtest validation.
        if is_wc:
            ctx.__dict__.setdefault("grf_sim_note", "use GoogleFootballSimulator for synthetic RL data")

        # Apply all gap data for WC (injuries, setpieces, sentiment, CLV, patterns)
        if is_wc:
            if self.injury_news:
                ctx.__dict__.setdefault("fresh_injuries", self.injury_news.get_fresh_injuries(home_team))
            if self.set_piece:
                ctx.__dict__.setdefault("setpiece_stats", self.set_piece.get_setpiece_stats())
            if self.sentiment:
                ctx.__dict__.setdefault("motivation_bias", self.sentiment.get_bias_score(home_team))
            if self.clv_tracker:
                ctx.__dict__.setdefault("clv_edge", self.clv_tracker.get_clv(0.65, 2.1))
            if self.wc_patterns:
                ctx.__dict__.setdefault("wc_historical", self.wc_patterns.get_wc_patterns(home_team))

            # Coach / tactical style (trainer data - how they let the team play) - important for both sides
            if getattr(self, "coach", None):
                try:
                    ctx.__dict__.setdefault("home_coach", self.coach.get_coach(home_team))
                    ctx.__dict__.setdefault("away_coach", self.coach.get_coach(away_team))
                    ctx.__dict__.setdefault("coach_matchup", self.coach.get_coach_matchup(home_team, away_team))
                except Exception:
                    pass

            # Multi-year historical WC (jfjelstul best structured source) + recent this-year (qualifiers 2024-26 + prep)
            if getattr(self, "jfjelstul", None):
                try:
                    ctx.__dict__.setdefault("home_wc_history", self.jfjelstul.get_multi_year_wc_stats(home_team))
                    ctx.__dict__.setdefault("away_wc_history", self.jfjelstul.get_multi_year_wc_stats(away_team))
                except Exception:
                    pass
            if getattr(self, "soccerdata", None):
                try:
                    ctx.__dict__.setdefault("home_recent_intl", self.soccerdata.get_recent_international_data(home_team))
                    ctx.__dict__.setdefault("away_recent_intl", self.soccerdata.get_recent_international_data(away_team))
                except Exception:
                    pass

        return ctx

    def _build_squad_status(
        self,
        team_name: str,
        players: List[PlayerInfo],
        injuries: List[PlayerInfo]
    ) -> TeamSquadStatus:
        """Build squad status from player data and injury list."""
        injured_names = {p.name.lower() for p in injuries}

        # Mark injured players
        for p in players:
            if p.name.lower() in injured_names:
                p.is_available = False

        injured = [p for p in players if not p.is_available]
        available_key = [p for p in players
                        if p.is_key_player and p.is_available]
        missing_key = [p for p in players
                      if p.is_key_player and not p.is_available]

        # Squad strength: reduce for each missing key player
        strength = 1.0
        for p in missing_key:
            # Weight by player contribution
            contribution = (p.goals_season + p.assists_season) / max(
                sum(pl.goals_season + pl.assists_season
                    for pl in players if pl.is_key_player), 1)
            strength -= contribution * 0.15  # Each key player = up to 15% impact

        return TeamSquadStatus(
            team_name=team_name,
            injured_players=injured + injuries,
            suspended_players=[p for p in players if p.is_suspended],
            available_key_players=available_key,
            missing_key_players=missing_key,
            squad_strength_score=max(strength, 0.5)
        )
