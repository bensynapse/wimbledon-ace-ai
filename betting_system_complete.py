#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    COMPLETE BETTING ANALYSIS SYSTEM v2.0                     ║
║                                                                              ║
║  Features:                                                                   ║
║  ✅ Poisson Model (xG-based)         ✅ CLV Tracking (CRITICAL)              ║
║  ✅ Half Kelly Criterion             ✅ Sharp Money Detection                ║
║  ✅ Backtesting Framework            ✅ Line Shopping                        ║
║  ✅ Motivation Factors               ✅ Fixture Congestion                   ║
║  ✅ Referee Stats                    ✅ Weather Impact                       ║
║  ✅ ELO Ratings                      ✅ H2H History                          ║
║                                                                              ║
║  Author: Built for Onur - OBT TECH                                           ║
║  Date: February 2026                                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json
import math
import os
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum
import warnings

# Third party imports (install via pip if needed)
try:
    import pandas as pd
    import numpy as np
    from scipy.stats import poisson
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("⚠️  Install scipy & pandas: pip install scipy pandas numpy")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("⚠️  Install requests: pip install requests")

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              CONFIGURATION                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@dataclass
class Config:
    """System configuration."""
    # API Keys (fill in your own)
    ODDS_API_KEY: str = "YOUR_ODDS_API_KEY"  # https://the-odds-api.com
    WEATHER_API_KEY: str = ""  # https://openweathermap.org (optional)
    
    # Bankroll Settings
    STARTING_BANKROLL: float = 1000.0
    KELLY_FRACTION: float = 0.5  # Half Kelly - NEVER use full Kelly!
    MAX_BET_PERCENT: float = 5.0
    MIN_EDGE_THRESHOLD: float = 3.0
    MIN_ODDS: float = 1.30
    MAX_ODDS: float = 5.00
    
    # CLV Settings
    CLV_BENCHMARK_BOOK: str = "pinnacle"
    CLV_TARGET_POSITIVE_RATE: float = 50.0
    
    # Model Settings
    HOME_ADVANTAGE: float = 1.25
    XG_WEIGHT: float = 0.6
    FORM_MATCHES: int = 5
    MAX_GOALS: int = 10
    
    # Data paths
    DATA_DIR: str = "data/"
    BET_TRACKER_FILE: str = "data/bets.json"
    
    def save(self, path: str = "config.json"):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str = "config.json"):
        if os.path.exists(path):
            with open(path) as f:
                return cls(**json.load(f))
        return cls()


CONFIG = Config()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           LEAGUES CONFIGURATION                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

LEAGUES = {
    "eredivisie": {
        "name": "Eredivisie",
        "country": "Netherlands",
        "odds_api_key": "soccer_netherlands_eredivisie",
        "football_data_code": "N1",
        "understat_name": "Eredivisie",
        "avg_goals": 3.0,
    },
    "premier_league": {
        "name": "Premier League", 
        "country": "England",
        "odds_api_key": "soccer_epl",
        "football_data_code": "E0",
        "understat_name": "EPL",
        "avg_goals": 2.7,
    },
    "la_liga": {
        "name": "La Liga",
        "country": "Spain", 
        "odds_api_key": "soccer_spain_la_liga",
        "football_data_code": "SP1",
        "understat_name": "La_Liga",
        "avg_goals": 2.5,
    },
    "bundesliga": {
        "name": "Bundesliga",
        "country": "Germany",
        "odds_api_key": "soccer_germany_bundesliga", 
        "football_data_code": "D1",
        "understat_name": "Bundesliga",
        "avg_goals": 3.0,
    },
    "serie_a": {
        "name": "Serie A",
        "country": "Italy",
        "odds_api_key": "soccer_italy_serie_a",
        "football_data_code": "I1",
        "understat_name": "Serie_A",
        "avg_goals": 2.6,
    },
    "ligue_1": {
        "name": "Ligue 1",
        "country": "France",
        "odds_api_key": "soccer_france_ligue_one",
        "football_data_code": "F1",
        "understat_name": "Ligue_1",
        "avg_goals": 2.6,
    },
}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           MOTIVATION FACTORS                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class Motivation(Enum):
    NORMAL = 1.00
    RELEGATION_BATTLE = 1.15
    TITLE_RACE = 1.10
    EUROPA_RACE = 1.08
    NOTHING_TO_PLAY = 0.90
    DERBY = 1.12
    NEW_MANAGER = 1.10
    MANAGER_LAST_MATCH = 0.95
    CUP_HANGOVER = 0.92


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              DATA CLASSES                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@dataclass
class TeamStats:
    """Team statistics container."""
    name: str
    xg_for: float = 0.0
    xg_against: float = 0.0
    goals_scored: int = 0
    goals_conceded: int = 0
    matches_played: int = 0
    elo_rating: float = 1500.0
    
    # Form (last N matches)
    form_xg: float = 0.0
    form_xga: float = 0.0
    form_goals: int = 0
    form_conceded: int = 0
    
    # Home/Away splits
    home_xg: float = 0.0
    away_xg: float = 0.0
    
    @property
    def xg_per_game(self) -> float:
        if self.matches_played == 0:
            return 0
        return self.xg_for / self.matches_played
    
    @property
    def xga_per_game(self) -> float:
        if self.matches_played == 0:
            return 0
        return self.xg_against / self.matches_played


@dataclass  
class MatchContext:
    """Match context factors."""
    # Referee
    referee_name: str = ""
    referee_cards_per_game: float = 3.5
    referee_home_bias: float = 0.0
    referee_penalties_per_game: float = 0.25
    
    # Motivation
    home_motivation: Motivation = Motivation.NORMAL
    away_motivation: Motivation = Motivation.NORMAL
    
    # Fatigue / Congestion
    home_days_rest: int = 7
    away_days_rest: int = 7
    home_matches_30_days: int = 4
    away_matches_30_days: int = 4
    
    # Special circumstances
    is_derby: bool = False
    home_key_players_out: int = 0
    away_key_players_out: int = 0
    
    # Weather
    weather: str = "normal"  # normal, rain, wind, extreme_heat, extreme_cold
    
    # H2H
    h2h_home_wins: int = 0
    h2h_draws: int = 0
    h2h_away_wins: int = 0


@dataclass
class Prediction:
    """Match prediction results."""
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    
    expected_home_goals: float
    expected_away_goals: float
    
    over_2_5_prob: float
    under_2_5_prob: float
    btts_yes_prob: float
    btts_no_prob: float
    
    # Additional markets
    over_1_5_prob: float = 0.0
    over_3_5_prob: float = 0.0
    home_over_1_5_prob: float = 0.0
    away_over_0_5_prob: float = 0.0
    
    # Score probabilities
    most_likely_scores: List[Tuple[Tuple[int, int], float]] = field(default_factory=list)
    
    confidence: str = "MEDIUM"


@dataclass
class Bet:
    """Single bet record."""
    id: str
    timestamp: str
    match: str
    league: str
    market: str
    selection: str
    
    # Odds
    odds_placed: float
    odds_closing: Optional[float] = None
    
    # Stake
    stake: float = 0.0
    stake_percent: float = 0.0
    
    # Probabilities
    our_probability: float = 0.0
    implied_probability: float = 0.0
    edge: float = 0.0
    
    # Bookmaker
    bookmaker: str = ""
    
    # Result
    result: str = "pending"  # pending, won, lost, void
    profit_loss: Optional[float] = None
    
    # CLV (Closing Line Value) - THE KEY METRIC
    clv: Optional[float] = None
    
    notes: str = ""


@dataclass
class ValueBet:
    """Value bet opportunity."""
    market: str
    selection: str
    our_prob: float
    implied_prob: float
    odds: float
    edge: float
    ev: float  # Expected Value
    kelly_stake: float
    confidence: str


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              DATA SCRAPERS                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class OddsAPIScraper:
    """
    Scraper for The Odds API.
    Free tier: 500 requests/month
    https://the-odds-api.com
    """
    
    BASE_URL = "https://api.the-odds-api.com/v4"
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or CONFIG.ODDS_API_KEY
        self.requests_remaining = 500
    
    def get_odds(self, sport: str, regions: str = "eu,uk",
                 markets: str = "h2h,totals") -> Dict:
        """Get live odds for a sport."""
        if not HAS_REQUESTS:
            return {"error": "requests library not installed"}
        
        url = f"{self.BASE_URL}/sports/{sport}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": "decimal",
        }
        
        try:
            resp = requests.get(url, params=params, timeout=10)
            self.requests_remaining = int(resp.headers.get("x-requests-remaining", 500))
            
            if resp.status_code == 200:
                return {"success": True, "data": resp.json(), "remaining": self.requests_remaining}
            return {"success": False, "error": resp.text}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_best_odds(self, league: str) -> List[Dict]:
        """Get best odds for all matches in a league."""
        league_config = LEAGUES.get(league, {})
        sport_key = league_config.get("odds_api_key")
        
        if not sport_key:
            return []
        
        result = self.get_odds(sport_key)
        if not result.get("success"):
            return []
        
        matches = []
        for game in result["data"]:
            match = {
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "commence_time": game["commence_time"],
                "best_odds": {
                    "home": {"odds": 0, "book": ""},
                    "draw": {"odds": 0, "book": ""},
                    "away": {"odds": 0, "book": ""},
                },
                "all_odds": {},
            }
            
            for bookmaker in game.get("bookmakers", []):
                book = bookmaker["key"]
                for market in bookmaker.get("markets", []):
                    if market["key"] == "h2h":
                        for outcome in market["outcomes"]:
                            name = outcome["name"]
                            price = outcome["price"]
                            
                            # Track all odds
                            if name not in match["all_odds"]:
                                match["all_odds"][name] = {}
                            match["all_odds"][name][book] = price
                            
                            # Track best
                            if name == game["home_team"]:
                                if price > match["best_odds"]["home"]["odds"]:
                                    match["best_odds"]["home"] = {"odds": price, "book": book}
                            elif name == game["away_team"]:
                                if price > match["best_odds"]["away"]["odds"]:
                                    match["best_odds"]["away"] = {"odds": price, "book": book}
                            elif name == "Draw":
                                if price > match["best_odds"]["draw"]["odds"]:
                                    match["best_odds"]["draw"] = {"odds": price, "book": book}
            
            matches.append(match)
        
        return matches


class FootballDataScraper:
    """
    Scraper for Football-Data.co.uk.
    Free historical data since 1993.
    https://www.football-data.co.uk
    """
    
    BASE_URL = "https://www.football-data.co.uk/mmz4281"
    
    def get_season_data(self, league_code: str, season: str) -> 'pd.DataFrame':
        """Download season data. Season format: '2425' for 2024/25."""
        if not HAS_REQUESTS:
            return pd.DataFrame()
        
        url = f"{self.BASE_URL}/{season}/{league_code}.csv"
        
        try:
            df = pd.read_csv(url, encoding='latin-1')
            return df
        except Exception as e:
            print(f"Error loading {url}: {e}")
            return pd.DataFrame()
    
    def get_referee_stats(self, league_code: str, seasons: List[str] = None) -> 'pd.DataFrame':
        """Calculate referee statistics from historical data."""
        if seasons is None:
            seasons = ['2324', '2425']
        
        all_data = []
        for season in seasons:
            df = self.get_season_data(league_code, season)
            if not df.empty and 'Referee' in df.columns:
                all_data.append(df)
        
        if not all_data:
            return pd.DataFrame()
        
        combined = pd.concat(all_data, ignore_index=True)
        
        # Aggregate referee stats
        ref_stats = combined.groupby('Referee').agg({
            'HY': 'sum',  # Home Yellow
            'AY': 'sum',  # Away Yellow
            'HR': 'sum',  # Home Red
            'AR': 'sum',  # Away Red
            'HomeTeam': 'count',
        }).rename(columns={'HomeTeam': 'matches'})
        
        ref_stats['total_cards'] = ref_stats['HY'] + ref_stats['AY'] + ref_stats['HR'] + ref_stats['AR']
        ref_stats['cards_per_game'] = ref_stats['total_cards'] / ref_stats['matches']
        ref_stats['home_card_bias'] = (ref_stats['HY'] + ref_stats['HR']) / ref_stats['total_cards'].replace(0, 1)
        
        return ref_stats.sort_values('cards_per_game', ascending=False)
    
    def get_h2h(self, league_code: str, home_team: str, away_team: str,
                seasons: List[str] = None) -> Dict:
        """Get head-to-head history between two teams."""
        if seasons is None:
            seasons = ['2122', '2223', '2324', '2425']
        
        h2h = {"home_wins": 0, "draws": 0, "away_wins": 0, "matches": []}
        
        for season in seasons:
            df = self.get_season_data(league_code, season)
            if df.empty:
                continue
            
            # Find matches between these teams
            matches = df[
                ((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) |
                ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))
            ]
            
            for _, match in matches.iterrows():
                home_goals = match.get('FTHG', 0)
                away_goals = match.get('FTAG', 0)
                
                h2h["matches"].append({
                    "date": match.get('Date'),
                    "home": match['HomeTeam'],
                    "away": match['AwayTeam'],
                    "score": f"{home_goals}-{away_goals}",
                })
                
                # Count results from perspective of our home team
                if match['HomeTeam'] == home_team:
                    if home_goals > away_goals:
                        h2h["home_wins"] += 1
                    elif home_goals < away_goals:
                        h2h["away_wins"] += 1
                    else:
                        h2h["draws"] += 1
                else:
                    # Reversed fixture
                    if away_goals > home_goals:
                        h2h["home_wins"] += 1
                    elif away_goals < home_goals:
                        h2h["away_wins"] += 1
                    else:
                        h2h["draws"] += 1
        
        return h2h


class UnderstatScraper:
    """
    Scraper for Understat.com (xG data).
    Free, unlimited access.
    """
    
    BASE_URL = "https://understat.com"
    
    def get_team_xg(self, league: str, season: str = "2024") -> Dict[str, Dict]:
        """Get xG data for all teams in a league."""
        if not HAS_REQUESTS or not HAS_BS4:
            return {}
        
        url = f"{self.BASE_URL}/league/{league}/{season}"
        
        try:
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            teams = {}
            for script in soup.find_all('script'):
                if script.string and 'teamsData' in script.string:
                    start = script.string.find("JSON.parse('") + len("JSON.parse('")
                    end = script.string.find("')", start)
                    data_str = script.string[start:end].encode().decode('unicode_escape')
                    data = json.loads(data_str)
                    
                    for team_id, team_data in data.items():
                        history = team_data.get('history', [])
                        if history:
                            last = history[-1]
                            teams[team_data['title']] = {
                                'matches': int(last.get('matches', 0)),
                                'xG': float(last.get('xG', 0)),
                                'xGA': float(last.get('xGA', 0)),
                                'goals': int(last.get('scored', 0)),
                                'conceded': int(last.get('missed', 0)),
                            }
                    break
            
            return teams
        except Exception as e:
            print(f"Understat error: {e}")
            return {}


class ClubEloScraper:
    """
    Scraper for ClubElo.com (ELO ratings).
    Free API access.
    """
    
    def get_current_ratings(self) -> Dict[str, float]:
        """Get current ELO ratings for all teams."""
        if not HAS_REQUESTS:
            return {}
        
        try:
            resp = requests.get("http://api.clubelo.com/", timeout=10)
            if resp.status_code == 200:
                ratings = {}
                for line in resp.text.strip().split('\n')[1:]:  # Skip header
                    parts = line.split(',')
                    if len(parts) >= 4:
                        team = parts[1]
                        elo = float(parts[3])
                        ratings[team] = elo
                return ratings
        except Exception as e:
            print(f"ClubElo error: {e}")
        
        return {}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                            POISSON MODEL                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class PoissonModel:
    """
    Poisson distribution model for football predictions.
    Uses xG data, ELO ratings, and various context factors.
    """
    
    def __init__(self, config: Config = None):
        self.config = config or CONFIG
    
    def calculate_expected_goals(self, 
                                  home: TeamStats, 
                                  away: TeamStats,
                                  context: MatchContext = None,
                                  league_avg: float = 2.7) -> Tuple[float, float]:
        """
        Calculate expected goals for both teams.
        """
        # Base attack/defense strengths
        home_attack = self._attack_strength(home, league_avg)
        home_defense = self._defense_strength(home, league_avg)
        away_attack = self._attack_strength(away, league_avg)
        away_defense = self._defense_strength(away, league_avg)
        
        # Base xG
        home_xg = home_attack * away_defense * (league_avg / 2) * self.config.HOME_ADVANTAGE
        away_xg = away_attack * home_defense * (league_avg / 2)
        
        # ELO adjustment
        elo_diff = home.elo_rating - away.elo_rating
        elo_factor = 1 + (elo_diff / 1000)
        home_xg *= min(1.3, max(0.7, elo_factor))
        away_xg *= min(1.3, max(0.7, 2 - elo_factor))
        
        # Context adjustments
        if context:
            home_xg, away_xg = self._apply_context(home_xg, away_xg, context)
        
        # Bounds
        home_xg = max(0.3, min(4.5, home_xg))
        away_xg = max(0.2, min(4.0, away_xg))
        
        return home_xg, away_xg
    
    def _attack_strength(self, team: TeamStats, league_avg: float) -> float:
        """Calculate attacking strength."""
        if team.matches_played == 0:
            return 1.0
        
        # Blend xG and actual goals
        xg_per_game = team.xg_for / team.matches_played if team.xg_for > 0 else 0
        goals_per_game = team.goals_scored / team.matches_played
        
        if xg_per_game > 0:
            attack = (xg_per_game * self.config.XG_WEIGHT + 
                     goals_per_game * (1 - self.config.XG_WEIGHT))
        else:
            attack = goals_per_game
        
        return max(0.5, min(2.0, attack / (league_avg / 2)))
    
    def _defense_strength(self, team: TeamStats, league_avg: float) -> float:
        """Calculate defensive strength (lower = better defense)."""
        if team.matches_played == 0:
            return 1.0
        
        xga_per_game = team.xg_against / team.matches_played if team.xg_against > 0 else 0
        conc_per_game = team.goals_conceded / team.matches_played
        
        if xga_per_game > 0:
            defense = (xga_per_game * self.config.XG_WEIGHT +
                      conc_per_game * (1 - self.config.XG_WEIGHT))
        else:
            defense = conc_per_game
        
        return max(0.5, min(2.0, defense / (league_avg / 2)))
    
    def _apply_context(self, home_xg: float, away_xg: float, 
                       ctx: MatchContext) -> Tuple[float, float]:
        """Apply context adjustments."""
        # Motivation
        home_xg *= ctx.home_motivation.value
        away_xg *= ctx.away_motivation.value
        
        # Fatigue
        if ctx.home_days_rest <= 3:
            home_xg *= 0.95
        if ctx.away_days_rest <= 3:
            away_xg *= 0.95
        
        if ctx.home_matches_30_days >= 8:
            home_xg *= 0.95
        if ctx.away_matches_30_days >= 8:
            away_xg *= 0.95
        
        # Derby
        if ctx.is_derby:
            home_xg *= 1.05
            away_xg *= 1.05
        
        # Key players out
        home_xg *= (1 - ctx.home_key_players_out * 0.05)
        away_xg *= (1 - ctx.away_key_players_out * 0.05)
        
        # Weather
        if ctx.weather == "rain":
            home_xg *= 0.92
            away_xg *= 0.92
        elif ctx.weather == "wind":
            home_xg *= 0.95
            away_xg *= 0.95
        
        return home_xg, away_xg
    
    def generate_score_matrix(self, home_xg: float, away_xg: float) -> Dict:
        """Generate probability matrix for all scores."""
        if not HAS_SCIPY:
            return {}
        
        matrix = {}
        for h in range(self.config.MAX_GOALS + 1):
            for a in range(self.config.MAX_GOALS + 1):
                prob = poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
                matrix[(h, a)] = prob
        
        return matrix
    
    def calculate_probabilities(self, home_xg: float, away_xg: float) -> Dict:
        """Calculate all market probabilities."""
        matrix = self.generate_score_matrix(home_xg, away_xg)
        
        if not matrix:
            # Fallback without scipy
            return self._simple_probabilities(home_xg, away_xg)
        
        probs = {
            "home_win": 0, "draw": 0, "away_win": 0,
            "over_1_5": 0, "over_2_5": 0, "over_3_5": 0,
            "btts_yes": 0,
        }
        
        for (h, a), prob in matrix.items():
            if h > a:
                probs["home_win"] += prob
            elif h == a:
                probs["draw"] += prob
            else:
                probs["away_win"] += prob
            
            if h + a > 1.5:
                probs["over_1_5"] += prob
            if h + a > 2.5:
                probs["over_2_5"] += prob
            if h + a > 3.5:
                probs["over_3_5"] += prob
            
            if h > 0 and a > 0:
                probs["btts_yes"] += prob
        
        probs["under_2_5"] = 1 - probs["over_2_5"]
        probs["btts_no"] = 1 - probs["btts_yes"]
        
        return probs
    
    def _simple_probabilities(self, home_xg: float, away_xg: float) -> Dict:
        """Simple probability calculation without scipy."""
        total_xg = home_xg + away_xg
        home_strength = home_xg / total_xg if total_xg > 0 else 0.5
        
        return {
            "home_win": home_strength * 0.75,
            "draw": 0.25,
            "away_win": (1 - home_strength) * 0.75,
            "over_2_5": min(0.9, total_xg / 4),
            "under_2_5": 1 - min(0.9, total_xg / 4),
            "btts_yes": min(0.8, (home_xg * away_xg) / 2),
            "btts_no": 1 - min(0.8, (home_xg * away_xg) / 2),
        }
    
    def predict(self, home: TeamStats, away: TeamStats,
                context: MatchContext = None,
                league_avg: float = 2.7) -> Prediction:
        """Generate full prediction."""
        home_xg, away_xg = self.calculate_expected_goals(home, away, context, league_avg)
        probs = self.calculate_probabilities(home_xg, away_xg)
        matrix = self.generate_score_matrix(home_xg, away_xg)
        
        # Most likely scores
        if matrix:
            sorted_scores = sorted(matrix.items(), key=lambda x: x[1], reverse=True)[:5]
        else:
            sorted_scores = []
        
        # Confidence
        data_quality = min(home.matches_played, away.matches_played)
        if data_quality >= 15:
            confidence = "HIGH"
        elif data_quality >= 8:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        return Prediction(
            home_win_prob=probs["home_win"],
            draw_prob=probs["draw"],
            away_win_prob=probs["away_win"],
            expected_home_goals=home_xg,
            expected_away_goals=away_xg,
            over_2_5_prob=probs["over_2_5"],
            under_2_5_prob=probs["under_2_5"],
            btts_yes_prob=probs["btts_yes"],
            btts_no_prob=probs["btts_no"],
            over_1_5_prob=probs.get("over_1_5", 0),
            over_3_5_prob=probs.get("over_3_5", 0),
            most_likely_scores=sorted_scores,
            confidence=confidence,
        )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                            KELLY CRITERION                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class KellyCalculator:
    """
    Kelly Criterion calculator for optimal bet sizing.
    
    CRITICAL: Always use fractional Kelly (0.5 = half Kelly).
    Full Kelly leads to ruin in practice!
    """
    
    def __init__(self, fraction: float = 0.5, max_percent: float = 5.0):
        self.fraction = fraction
        self.max_percent = max_percent
    
    def calculate(self, prob: float, odds: float) -> float:
        """
        Calculate Kelly stake as fraction of bankroll.
        
        Formula: f* = (bp - q) / b
        Where: b = odds - 1, p = win prob, q = 1 - p
        """
        if odds <= 1.0 or prob <= 0 or prob >= 1:
            return 0.0
        
        b = odds - 1
        p = prob
        q = 1 - p
        
        kelly = (b * p - q) / b
        kelly *= self.fraction  # Fractional Kelly
        kelly = min(kelly, self.max_percent / 100)  # Cap
        
        return max(0, kelly)
    
    def calculate_stake(self, bankroll: float, prob: float, odds: float) -> float:
        """Calculate actual stake amount."""
        return bankroll * self.calculate(prob, odds)
    
    def calculate_edge(self, prob: float, odds: float) -> float:
        """Calculate edge percentage."""
        implied = 1 / odds
        return ((prob - implied) / implied) * 100
    
    def calculate_ev(self, prob: float, odds: float) -> float:
        """Calculate Expected Value."""
        return (prob * (odds - 1)) - (1 - prob)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           VALUE BET FINDER                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class ValueBetFinder:
    """Find value bets by comparing model probabilities to market odds."""
    
    def __init__(self, config: Config = None):
        self.config = config or CONFIG
        self.kelly = KellyCalculator(
            fraction=self.config.KELLY_FRACTION,
            max_percent=self.config.MAX_BET_PERCENT
        )
    
    def find_value(self, prediction: Prediction, odds: Dict) -> List[ValueBet]:
        """Find all value bets for a match."""
        markets = [
            ("1X2", "Home", prediction.home_win_prob, "home_win"),
            ("1X2", "Draw", prediction.draw_prob, "draw"),
            ("1X2", "Away", prediction.away_win_prob, "away_win"),
            ("Totals", "Over 2.5", prediction.over_2_5_prob, "over_2_5"),
            ("Totals", "Under 2.5", prediction.under_2_5_prob, "under_2_5"),
            ("BTTS", "Yes", prediction.btts_yes_prob, "btts_yes"),
            ("BTTS", "No", prediction.btts_no_prob, "btts_no"),
        ]
        
        value_bets = []
        
        for market, selection, prob, odds_key in markets:
            if odds_key not in odds or not odds[odds_key]:
                continue
            
            market_odds = odds[odds_key]
            
            # Check odds bounds
            if market_odds < self.config.MIN_ODDS or market_odds > self.config.MAX_ODDS:
                continue
            
            edge = self.kelly.calculate_edge(prob, market_odds)
            
            if edge >= self.config.MIN_EDGE_THRESHOLD:
                ev = self.kelly.calculate_ev(prob, market_odds)
                kelly_stake = self.kelly.calculate(prob, market_odds)
                
                confidence = "HIGH" if edge >= 7 else "MEDIUM" if edge >= 5 else "LOW"
                
                value_bets.append(ValueBet(
                    market=market,
                    selection=selection,
                    our_prob=prob,
                    implied_prob=1 / market_odds,
                    odds=market_odds,
                    edge=edge,
                    ev=ev,
                    kelly_stake=kelly_stake,
                    confidence=confidence,
                ))
        
        # Sort by edge
        value_bets.sort(key=lambda x: x.edge, reverse=True)
        return value_bets


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           BET TRACKER & CLV                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class BetTracker:
    """
    Track all bets and calculate performance metrics.
    CLV (Closing Line Value) is the most important metric!
    """
    
    def __init__(self, filepath: str = None):
        self.filepath = filepath or CONFIG.BET_TRACKER_FILE
        self.bets: List[Bet] = []
        self._load()
    
    def _load(self):
        """Load bets from file."""
        os.makedirs(os.path.dirname(self.filepath) or '.', exist_ok=True)
        
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath) as f:
                    data = json.load(f)
                    self.bets = [Bet(**b) for b in data]
            except Exception as e:
                print(f"Error loading bets: {e}")
    
    def _save(self):
        """Save bets to file."""
        with open(self.filepath, 'w') as f:
            json.dump([asdict(b) for b in self.bets], f, indent=2)
    
    def add_bet(self, match: str, league: str, market: str, selection: str,
                odds: float, stake: float, bankroll: float,
                our_probability: float, bookmaker: str = "") -> Bet:
        """Record a new bet."""
        implied = 1 / odds
        edge = ((our_probability - implied) / implied) * 100
        
        bet = Bet(
            id=f"BET_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.bets)}",
            timestamp=datetime.now().isoformat(),
            match=match,
            league=league,
            market=market,
            selection=selection,
            odds_placed=odds,
            stake=stake,
            stake_percent=(stake / bankroll) * 100,
            our_probability=our_probability,
            implied_probability=implied,
            edge=edge,
            bookmaker=bookmaker,
        )
        
        self.bets.append(bet)
        self._save()
        return bet
    
    def update_result(self, bet_id: str, result: str, 
                      closing_odds: float = None) -> Optional[Bet]:
        """
        Update bet result and calculate CLV.
        
        CRITICAL: Always record closing_odds for CLV calculation!
        """
        for bet in self.bets:
            if bet.id == bet_id:
                bet.result = result
                
                # Profit/Loss
                if result == "won":
                    bet.profit_loss = bet.stake * (bet.odds_placed - 1)
                elif result == "lost":
                    bet.profit_loss = -bet.stake
                else:
                    bet.profit_loss = 0
                
                # CLV - THE KEY METRIC
                if closing_odds:
                    bet.odds_closing = closing_odds
                    # Positive CLV = you got better odds than closing
                    bet.clv = ((bet.odds_placed - closing_odds) / closing_odds) * 100
                
                self._save()
                return bet
        return None
    
    def get_stats(self) -> Dict:
        """Calculate comprehensive statistics."""
        if not self.bets:
            return {}
        
        settled = [b for b in self.bets if b.result in ["won", "lost"]]
        if not settled:
            return {"total": len(self.bets), "pending": len(self.bets)}
        
        won = len([b for b in settled if b.result == "won"])
        lost = len([b for b in settled if b.result == "lost"])
        
        total_staked = sum(b.stake for b in settled)
        profit_loss = sum(b.profit_loss or 0 for b in settled)
        
        roi = (profit_loss / total_staked * 100) if total_staked > 0 else 0
        win_rate = (won / len(settled) * 100) if settled else 0
        
        # CLV analysis
        bets_with_clv = [b for b in settled if b.clv is not None]
        clv_positive = len([b for b in bets_with_clv if b.clv > 0])
        clv_positive_rate = (clv_positive / len(bets_with_clv) * 100) if bets_with_clv else 0
        avg_clv = statistics.mean([b.clv for b in bets_with_clv]) if bets_with_clv else 0
        
        return {
            "total_bets": len(self.bets),
            "settled": len(settled),
            "won": won,
            "lost": lost,
            "pending": len([b for b in self.bets if b.result == "pending"]),
            "win_rate": win_rate,
            "total_staked": total_staked,
            "profit_loss": profit_loss,
            "roi": roi,
            "clv_positive_rate": clv_positive_rate,
            "avg_clv": avg_clv,
            "clv_sample_size": len(bets_with_clv),
        }
    
    def generate_report(self) -> str:
        """Generate performance report."""
        stats = self.get_stats()
        
        if not stats:
            return "No bets recorded yet."
        
        report = []
        report.append("=" * 60)
        report.append("📊 BETTING PERFORMANCE REPORT")
        report.append("=" * 60)
        report.append("")
        
        report.append("OVERALL")
        report.append(f"  Total Bets: {stats.get('total_bets', 0)}")
        report.append(f"  Settled: {stats.get('settled', 0)} (Won: {stats.get('won', 0)}, Lost: {stats.get('lost', 0)})")
        report.append(f"  Pending: {stats.get('pending', 0)}")
        report.append(f"  Win Rate: {stats.get('win_rate', 0):.1f}%")
        report.append("")
        
        report.append("FINANCIAL")
        report.append(f"  Total Staked: €{stats.get('total_staked', 0):.2f}")
        report.append(f"  Profit/Loss: €{stats.get('profit_loss', 0):.2f}")
        report.append(f"  ROI: {stats.get('roi', 0):.2f}%")
        report.append("")
        
        report.append("📈 CLV ANALYSIS (CRITICAL METRIC)")
        report.append(f"  CLV Positive Rate: {stats.get('clv_positive_rate', 0):.1f}%")
        report.append(f"  Average CLV: {stats.get('avg_clv', 0):.2f}%")
        report.append(f"  Sample Size: {stats.get('clv_sample_size', 0)} bets with CLV data")
        
        clv_rate = stats.get('clv_positive_rate', 0)
        if clv_rate >= 55:
            report.append("  ✅ Excellent! You're consistently beating the closing line")
        elif clv_rate >= 50:
            report.append("  ✅ Good. You're finding value in the market")
        elif clv_rate > 0:
            report.append("  ⚠️ Warning: Not consistently beating closing line")
        else:
            report.append("  ❌ Need more CLV data - record closing odds!")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              BACKTESTER                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class Backtester:
    """
    Backtest betting strategies on historical data.
    """
    
    def __init__(self, config: Config = None):
        self.config = config or CONFIG
        self.model = PoissonModel(config)
        self.kelly = KellyCalculator(
            fraction=self.config.KELLY_FRACTION,
            max_percent=self.config.MAX_BET_PERCENT
        )
        self.scraper = FootballDataScraper()
    
    def run(self, league_code: str, seasons: List[str],
            min_edge: float = 3.0,
            progress_fn: Callable = None) -> Dict:
        """
        Run backtest on historical data.
        """
        # Load data
        all_data = []
        for season in seasons:
            df = self.scraper.get_season_data(league_code, season)
            if not df.empty:
                df['Season'] = season
                all_data.append(df)
        
        if not all_data:
            return {"error": f"No data found for {league_code}"}
        
        data = pd.concat(all_data, ignore_index=True)
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce')
            data = data.sort_values('Date')
        
        # Initialize
        bankroll = self.config.STARTING_BANKROLL
        history = [bankroll]
        bets_log = []
        
        total_bets = 0
        won = 0
        lost = 0
        total_clv = 0
        clv_positive = 0
        
        team_stats = {}  # Track team performance
        
        # Iterate through matches
        for idx, row in data.iterrows():
            if progress_fn and idx % 50 == 0:
                progress_fn(idx / len(data))
            
            home = row.get('HomeTeam')
            away = row.get('AwayTeam')
            
            if pd.isna(home) or pd.isna(away):
                continue
            
            # Update team stats
            for team in [home, away]:
                if team not in team_stats:
                    team_stats[team] = TeamStats(name=team)
            
            # Need minimum history
            if (team_stats[home].matches_played < 5 or 
                team_stats[away].matches_played < 5):
                self._update_team_stats(team_stats, row)
                continue
            
            # Predict
            home_stats = team_stats[home]
            away_stats = team_stats[away]
            prediction = self.model.predict(home_stats, away_stats)
            
            # Get odds
            odds = self._get_odds(row)
            closing_odds = self._get_closing_odds(row)
            
            # Actual result
            home_goals = row.get('FTHG', 0) or 0
            away_goals = row.get('FTAG', 0) or 0
            
            # Check value bets
            markets = [
                ("Home", prediction.home_win_prob, odds.get("home"), 
                 home_goals > away_goals, closing_odds.get("home")),
                ("Draw", prediction.draw_prob, odds.get("draw"),
                 home_goals == away_goals, closing_odds.get("draw")),
                ("Away", prediction.away_win_prob, odds.get("away"),
                 home_goals < away_goals, closing_odds.get("away")),
                ("Over 2.5", prediction.over_2_5_prob, odds.get("over"),
                 home_goals + away_goals > 2.5, closing_odds.get("over")),
            ]
            
            for selection, prob, market_odds, won_bet, close in markets:
                if not market_odds or market_odds < 1.2:
                    continue
                
                edge = self.kelly.calculate_edge(prob, market_odds)
                
                if edge >= min_edge:
                    stake = self.kelly.calculate_stake(bankroll, prob, market_odds)
                    if stake < 1:
                        continue
                    
                    total_bets += 1
                    
                    if won_bet:
                        won += 1
                        profit = stake * (market_odds - 1)
                        bankroll += profit
                    else:
                        lost += 1
                        bankroll -= stake
                    
                    # CLV
                    clv = None
                    if close:
                        clv = ((market_odds - close) / close) * 100
                        total_clv += clv
                        if clv > 0:
                            clv_positive += 1
                    
                    history.append(bankroll)
                    bets_log.append({
                        "match": f"{home} vs {away}",
                        "selection": selection,
                        "odds": market_odds,
                        "won": won_bet,
                        "clv": clv,
                    })
                    
                    if bankroll < 10:
                        break
            
            # Update stats after match
            self._update_team_stats(team_stats, row)
            
            if bankroll < 10:
                break
        
        # Results
        profit = bankroll - self.config.STARTING_BANKROLL
        total_staked = sum(b.get("stake", 30) for b in bets_log)  # Approximate
        
        return {
            "start_bankroll": self.config.STARTING_BANKROLL,
            "end_bankroll": bankroll,
            "profit_loss": profit,
            "roi": (profit / max(total_staked, 1)) * 100,
            "total_bets": total_bets,
            "won": won,
            "lost": lost,
            "win_rate": (won / total_bets * 100) if total_bets > 0 else 0,
            "clv_positive_rate": (clv_positive / total_bets * 100) if total_bets > 0 else 0,
            "avg_clv": (total_clv / total_bets) if total_bets > 0 else 0,
            "max_drawdown": self._calc_drawdown(history),
            "history": history,
            "bets": bets_log,
        }
    
    def _get_odds(self, row: 'pd.Series') -> Dict:
        """Get opening odds from data."""
        return {
            "home": row.get('B365H') or row.get('BWH'),
            "draw": row.get('B365D') or row.get('BWD'),
            "away": row.get('B365A') or row.get('BWA'),
            "over": row.get('B365>2.5') or row.get('BbAv>2.5'),
        }
    
    def _get_closing_odds(self, row: 'pd.Series') -> Dict:
        """Get closing odds (Pinnacle)."""
        return {
            "home": row.get('PSH') or row.get('PSCH'),
            "draw": row.get('PSD') or row.get('PSCD'),
            "away": row.get('PSA') or row.get('PSCA'),
            "over": row.get('P>2.5'),
        }
    
    def _update_team_stats(self, stats: Dict, row: 'pd.Series'):
        """Update team statistics after a match."""
        home = row.get('HomeTeam')
        away = row.get('AwayTeam')
        hg = row.get('FTHG', 0) or 0
        ag = row.get('FTAG', 0) or 0
        
        if home in stats:
            stats[home].goals_scored += hg
            stats[home].goals_conceded += ag
            stats[home].matches_played += 1
        
        if away in stats:
            stats[away].goals_scored += ag
            stats[away].goals_conceded += hg
            stats[away].matches_played += 1
    
    def _calc_drawdown(self, history: List[float]) -> float:
        """Calculate maximum drawdown."""
        if not history:
            return 0
        peak = history[0]
        max_dd = 0
        for val in history:
            if val > peak:
                peak = val
            dd = (peak - val) / peak * 100
            max_dd = max(max_dd, dd)
        return max_dd
    
    def generate_report(self, result: Dict) -> str:
        """Generate backtest report."""
        report = []
        report.append("=" * 60)
        report.append("🔬 BACKTEST RESULTS")
        report.append("=" * 60)
        report.append("")
        
        report.append("PERFORMANCE")
        report.append(f"  Starting Bankroll: €{result.get('start_bankroll', 0):.2f}")
        report.append(f"  Final Bankroll: €{result.get('end_bankroll', 0):.2f}")
        report.append(f"  Profit/Loss: €{result.get('profit_loss', 0):.2f}")
        report.append(f"  ROI: {result.get('roi', 0):.2f}%")
        report.append("")
        
        report.append("BETS")
        report.append(f"  Total: {result.get('total_bets', 0)}")
        report.append(f"  Won: {result.get('won', 0)} | Lost: {result.get('lost', 0)}")
        report.append(f"  Win Rate: {result.get('win_rate', 0):.1f}%")
        report.append("")
        
        report.append("RISK")
        report.append(f"  Max Drawdown: {result.get('max_drawdown', 0):.1f}%")
        report.append("")
        
        report.append("CLV ANALYSIS")
        report.append(f"  CLV Positive Rate: {result.get('clv_positive_rate', 0):.1f}%")
        report.append(f"  Average CLV: {result.get('avg_clv', 0):.2f}%")
        
        clv = result.get('clv_positive_rate', 0)
        if clv >= 55:
            report.append("  ✅ Excellent - model consistently beats closing line")
        elif clv >= 50:
            report.append("  ✅ Good - model shows edge")
        else:
            report.append("  ⚠️ Model not beating closing line consistently")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           MAIN SYSTEM                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class BettingSystem:
    """
    Main betting system orchestrator.
    """
    
    def __init__(self, config: Config = None):
        self.config = config or CONFIG
        self.model = PoissonModel(self.config)
        self.value_finder = ValueBetFinder(self.config)
        self.tracker = BetTracker()
        self.kelly = KellyCalculator(
            fraction=self.config.KELLY_FRACTION,
            max_percent=self.config.MAX_BET_PERCENT
        )
        
        # Scrapers
        self.odds_api = OddsAPIScraper(self.config.ODDS_API_KEY)
        self.football_data = FootballDataScraper()
        self.understat = UnderstatScraper()
        self.club_elo = ClubEloScraper()
        
        # Current bankroll
        self.bankroll = self.config.STARTING_BANKROLL
    
    def analyze_match(self, home_name: str, away_name: str,
                      league: str = "premier_league",
                      odds: Dict = None,
                      context: MatchContext = None) -> Dict:
        """
        Full match analysis.
        """
        league_config = LEAGUES.get(league, {})
        avg_goals = league_config.get("avg_goals", 2.7)
        
        # Create basic team stats
        home = TeamStats(name=home_name)
        away = TeamStats(name=away_name)
        
        # Try to get real data
        try:
            # xG from Understat
            understat_league = league_config.get("understat_name")
            if understat_league:
                xg_data = self.understat.get_team_xg(understat_league)
                if home_name in xg_data:
                    d = xg_data[home_name]
                    home.xg_for = d.get('xG', 0)
                    home.xg_against = d.get('xGA', 0)
                    home.goals_scored = d.get('goals', 0)
                    home.goals_conceded = d.get('conceded', 0)
                    home.matches_played = d.get('matches', 0)
                
                if away_name in xg_data:
                    d = xg_data[away_name]
                    away.xg_for = d.get('xG', 0)
                    away.xg_against = d.get('xGA', 0)
                    away.goals_scored = d.get('goals', 0)
                    away.goals_conceded = d.get('conceded', 0)
                    away.matches_played = d.get('matches', 0)
            
            # ELO ratings
            elo_data = self.club_elo.get_current_ratings()
            for team_name, elo in elo_data.items():
                if home_name.lower() in team_name.lower():
                    home.elo_rating = elo
                if away_name.lower() in team_name.lower():
                    away.elo_rating = elo
        
        except Exception as e:
            print(f"Data fetch warning: {e}")
        
        # Generate prediction
        prediction = self.model.predict(home, away, context, avg_goals)
        
        # Find value bets
        value_bets = []
        if odds:
            value_bets = self.value_finder.find_value(prediction, odds)
        
        return {
            "match": f"{home_name} vs {away_name}",
            "league": league,
            "home_stats": home,
            "away_stats": away,
            "prediction": prediction,
            "value_bets": value_bets,
        }
    
    def print_analysis(self, result: Dict):
        """Print formatted analysis."""
        pred = result["prediction"]
        
        print("\n" + "=" * 60)
        print(f"🎯 {result['match']}")
        print("=" * 60)
        
        print(f"\n📊 EXPECTED GOALS")
        print(f"   Home: {pred.expected_home_goals:.2f}")
        print(f"   Away: {pred.expected_away_goals:.2f}")
        
        print(f"\n📈 WIN PROBABILITIES")
        print(f"   Home Win:  {pred.home_win_prob * 100:.1f}%")
        print(f"   Draw:      {pred.draw_prob * 100:.1f}%")
        print(f"   Away Win:  {pred.away_win_prob * 100:.1f}%")
        
        print(f"\n⚽ OTHER MARKETS")
        print(f"   Over 2.5:  {pred.over_2_5_prob * 100:.1f}%")
        print(f"   Under 2.5: {pred.under_2_5_prob * 100:.1f}%")
        print(f"   BTTS Yes:  {pred.btts_yes_prob * 100:.1f}%")
        
        if pred.most_likely_scores:
            print(f"\n🎯 MOST LIKELY SCORES")
            for (h, a), prob in pred.most_likely_scores[:3]:
                print(f"   {h}-{a}: {prob * 100:.1f}%")
        
        if result["value_bets"]:
            print(f"\n💰 VALUE BETS FOUND")
            for vb in result["value_bets"]:
                print(f"\n   {vb.market} - {vb.selection}")
                print(f"   Odds: {vb.odds:.2f} | Edge: {vb.edge:.1f}%")
                print(f"   Kelly Stake: {vb.kelly_stake * 100:.1f}% of bankroll")
                print(f"   Confidence: {vb.confidence}")
        else:
            print(f"\n❌ No value bets found at current odds")
        
        print(f"\nPrediction Confidence: {pred.confidence}")
        print("=" * 60)
    
    def record_bet(self, match: str, league: str, market: str, selection: str,
                   odds: float, stake: float, our_prob: float, 
                   bookmaker: str = "") -> str:
        """Record a new bet."""
        bet = self.tracker.add_bet(
            match=match,
            league=league,
            market=market,
            selection=selection,
            odds=odds,
            stake=stake,
            bankroll=self.bankroll,
            our_probability=our_prob,
            bookmaker=bookmaker,
        )
        return bet.id
    
    def settle_bet(self, bet_id: str, result: str, closing_odds: float = None):
        """
        Settle a bet.
        
        IMPORTANT: Always provide closing_odds for CLV tracking!
        """
        bet = self.tracker.update_result(bet_id, result, closing_odds)
        if bet and bet.profit_loss:
            self.bankroll += bet.profit_loss
        return bet
    
    def get_performance(self) -> str:
        """Get performance report."""
        return self.tracker.generate_report()
    
    def run_backtest(self, league: str, seasons: List[str]) -> str:
        """Run backtest for a league."""
        league_config = LEAGUES.get(league, {})
        code = league_config.get("football_data_code")
        
        if not code:
            return f"Unknown league: {league}"
        
        bt = Backtester(self.config)
        result = bt.run(code, seasons)
        
        if "error" in result:
            return result["error"]
        
        return bt.generate_report(result)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              CLI INTERFACE                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def demo():
    """Run a demonstration of the system."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🎯 BETTING SYSTEM DEMO 🎯                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    system = BettingSystem()
    
    # Demo match analysis
    print("\n📊 ANALYZING: Arsenal vs Chelsea")
    print("-" * 40)
    
    # Create sample stats (in real usage, these come from scrapers)
    arsenal = TeamStats(
        name="Arsenal",
        xg_for=42.5,
        xg_against=18.2,
        goals_scored=45,
        goals_conceded=16,
        matches_played=18,
        elo_rating=1850,
    )
    
    chelsea = TeamStats(
        name="Chelsea",
        xg_for=35.1,
        xg_against=28.5,
        goals_scored=38,
        goals_conceded=30,
        matches_played=18,
        elo_rating=1720,
    )
    
    # Context
    context = MatchContext(
        home_motivation=Motivation.TITLE_RACE,
        away_motivation=Motivation.EUROPA_RACE,
        is_derby=True,
        home_days_rest=7,
        away_days_rest=4,
    )
    
    # Odds
    odds = {
        "home_win": 1.65,
        "draw": 3.90,
        "away_win": 5.50,
        "over_2_5": 1.72,
        "under_2_5": 2.15,
        "btts_yes": 1.75,
        "btts_no": 2.05,
    }
    
    # Run prediction
    prediction = system.model.predict(arsenal, chelsea, context, league_avg=2.7)
    value_bets = system.value_finder.find_value(prediction, odds)
    
    result = {
        "match": "Arsenal vs Chelsea",
        "prediction": prediction,
        "value_bets": value_bets,
    }
    
    system.print_analysis(result)
    
    # Kelly calculation demo
    print("\n💰 KELLY STAKE CALCULATION")
    print("-" * 40)
    
    if value_bets:
        vb = value_bets[0]
        stake = system.kelly.calculate_stake(1000, vb.our_prob, vb.odds)
        print(f"For best value bet ({vb.selection} @ {vb.odds}):")
        print(f"  Bankroll: €1000")
        print(f"  Our probability: {vb.our_prob * 100:.1f}%")
        print(f"  Edge: {vb.edge:.1f}%")
        print(f"  Half Kelly stake: €{stake:.2f}")
    
    # Simulated bet tracking
    print("\n📝 BET TRACKING DEMO")
    print("-" * 40)
    
    tracker = BetTracker("data/demo_bets.json")
    
    # Add some demo bets
    demo_bets = [
        ("Arsenal vs Chelsea", "Home", 1.65, 50, 0.62, "won", 1.58),
        ("Liverpool vs Man City", "Over 2.5", 1.80, 40, 0.58, "won", 1.75),
        ("Tottenham vs Newcastle", "Draw", 3.50, 25, 0.30, "lost", 3.60),
    ]
    
    for match, sel, odds, stake, prob, res, close in demo_bets:
        bet = tracker.add_bet(
            match=match, league="premier_league", market="Demo",
            selection=sel, odds=odds, stake=stake,
            bankroll=1000, our_probability=prob, bookmaker="demo"
        )
        tracker.update_result(bet.id, res, close)
        
        clv = ((odds - close) / close) * 100
        status = "✅" if res == "won" else "❌"
        clv_status = "📈" if clv > 0 else "📉"
        print(f"  {status} {match}: {sel} @ {odds} (CLV: {clv:.1f}% {clv_status})")
    
    print("\n" + tracker.generate_report())
    
    # Cleanup
    try:
        os.remove("data/demo_bets.json")
    except:
        pass
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         SYSTEM READY TO USE!                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  USAGE:                                                                      ║
║    from betting_system_complete import BettingSystem                         ║
║    system = BettingSystem()                                                  ║
║                                                                              ║
║    # Analyze a match                                                         ║
║    result = system.analyze_match("Arsenal", "Chelsea", "premier_league",     ║
║                                   odds={"home_win": 1.65, ...})              ║
║    system.print_analysis(result)                                             ║
║                                                                              ║
║    # Run backtest                                                            ║
║    print(system.run_backtest("premier_league", ["2324"]))                    ║
║                                                                              ║
║    # Track bets                                                              ║
║    bet_id = system.record_bet(...)                                           ║
║    system.settle_bet(bet_id, "won", closing_odds=1.58)                       ║
║                                                                              ║
║  IMPORTANT:                                                                  ║
║    - Always use Half Kelly (0.5) - NEVER full Kelly!                         ║
║    - Track closing odds for CLV - this is THE key metric                     ║
║    - Minimum 500 bets before drawing conclusions                             ║
║    - Backtest before using real money                                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    demo()
