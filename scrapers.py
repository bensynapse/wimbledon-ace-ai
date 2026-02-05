"""
DATA SCRAPERS MODULE
====================
Scrapers for all free betting data sources.
"""

import requests
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os

# Try to import optional dependencies
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    print("Warning: beautifulsoup4 not installed. Run: pip install beautifulsoup4")

try:
    import soccerdata as sd
    HAS_SOCCERDATA = True
except ImportError:
    HAS_SOCCERDATA = False
    print("Warning: soccerdata not installed. Run: pip install soccerdata")

from config import API_KEYS, LEAGUES, DATA_SOURCES, PATHS, NEWS_SETTINGS


class OddsAPIScraper:
    """
    Scraper for The Odds API (500 free requests/month)
    https://the-odds-api.com
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or API_KEYS.get("odds_api")
        self.base_url = DATA_SOURCES["odds_api"]
        self.requests_used = 0
        self.requests_remaining = 500
        
    def get_odds(self, sport: str, regions: str = "eu,uk", 
                 markets: str = "h2h,spreads,totals") -> Dict:
        """
        Get live odds for a sport.
        
        Args:
            sport: Sport key (e.g., 'soccer_epl')
            regions: Regions for odds (eu, uk, us, au)
            markets: Markets to fetch (h2h, spreads, totals)
        
        Returns:
            Dictionary with odds data
        """
        url = f"{self.base_url}/sports/{sport}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": "decimal",
        }
        
        try:
            response = requests.get(url, params=params)
            
            # Track API usage
            self.requests_used = int(response.headers.get("x-requests-used", 0))
            self.requests_remaining = int(response.headers.get("x-requests-remaining", 500))
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "data": response.json(),
                    "requests_remaining": self.requests_remaining,
                }
            else:
                return {
                    "success": False,
                    "error": response.text,
                    "requests_remaining": self.requests_remaining,
                }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_upcoming_matches(self, league: str) -> List[Dict]:
        """Get upcoming matches with odds for a league."""
        league_config = LEAGUES.get(league, {})
        sport_key = league_config.get("odds_api_key")
        
        if not sport_key:
            return []
        
        result = self.get_odds(sport_key)
        
        if result["success"]:
            matches = []
            for game in result["data"]:
                match = {
                    "id": game["id"],
                    "home_team": game["home_team"],
                    "away_team": game["away_team"],
                    "commence_time": game["commence_time"],
                    "bookmakers": {},
                }
                
                for bookmaker in game.get("bookmakers", []):
                    book_name = bookmaker["key"]
                    match["bookmakers"][book_name] = {}
                    
                    for market in bookmaker.get("markets", []):
                        market_key = market["key"]
                        match["bookmakers"][book_name][market_key] = {
                            outcome["name"]: outcome["price"]
                            for outcome in market["outcomes"]
                        }
                
                matches.append(match)
            
            return matches
        return []
    
    def find_best_odds(self, matches: List[Dict]) -> List[Dict]:
        """Find the best odds across all bookmakers for each match."""
        best_odds = []
        
        for match in matches:
            best = {
                "home_team": match["home_team"],
                "away_team": match["away_team"],
                "commence_time": match["commence_time"],
                "best_home_odds": {"odds": 0, "bookmaker": ""},
                "best_draw_odds": {"odds": 0, "bookmaker": ""},
                "best_away_odds": {"odds": 0, "bookmaker": ""},
            }
            
            for bookmaker, markets in match.get("bookmakers", {}).items():
                h2h = markets.get("h2h", {})
                
                home_odds = h2h.get(match["home_team"], 0)
                away_odds = h2h.get(match["away_team"], 0)
                draw_odds = h2h.get("Draw", 0)
                
                if home_odds > best["best_home_odds"]["odds"]:
                    best["best_home_odds"] = {"odds": home_odds, "bookmaker": bookmaker}
                if draw_odds > best["best_draw_odds"]["odds"]:
                    best["best_draw_odds"] = {"odds": draw_odds, "bookmaker": bookmaker}
                if away_odds > best["best_away_odds"]["odds"]:
                    best["best_away_odds"] = {"odds": away_odds, "bookmaker": bookmaker}
            
            best_odds.append(best)
        
        return best_odds


class UnderstatScraper:
    """
    Scraper for Understat.com (xG data)
    Free, unlimited scraping
    """
    
    def __init__(self):
        self.base_url = DATA_SOURCES["understat"]
        
    def get_league_data(self, league: str, season: str = "2024") -> Dict:
        """
        Get xG data for a league.
        
        Args:
            league: League name (EPL, La_Liga, Bundesliga, Serie_A, Ligue_1)
            season: Season year (e.g., '2024' for 2024/25)
        """
        if not HAS_BS4:
            return {"error": "beautifulsoup4 not installed"}
        
        url = f"{self.base_url}/league/{league}/{season}"
        
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract JSON data from script tags
            scripts = soup.find_all("script")
            data = {}
            
            for script in scripts:
                if script.string and "teamsData" in script.string:
                    # Extract teams data
                    start = script.string.find("teamsData") + len("teamsData = JSON.parse('")
                    end = script.string.find("')", start)
                    teams_json = script.string[start:end].encode().decode('unicode_escape')
                    data["teams"] = json.loads(teams_json)
                    
                if script.string and "datesData" in script.string:
                    # Extract match data
                    start = script.string.find("datesData") + len("datesData = JSON.parse('")
                    end = script.string.find("')", start)
                    dates_json = script.string[start:end].encode().decode('unicode_escape')
                    data["matches"] = json.loads(dates_json)
            
            return data
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_team_xg(self, league: str, season: str = "2024") -> pd.DataFrame:
        """Get xG table for all teams in a league."""
        data = self.get_league_data(league, season)
        
        if "error" in data or "teams" not in data:
            return pd.DataFrame()
        
        teams_list = []
        for team_id, team_data in data["teams"].items():
            teams_list.append({
                "team": team_data["title"],
                "matches": int(team_data["history"][-1]["matches"]) if team_data["history"] else 0,
                "xG": float(team_data["history"][-1]["xG"]) if team_data["history"] else 0,
                "xGA": float(team_data["history"][-1]["xGA"]) if team_data["history"] else 0,
                "goals": int(team_data["history"][-1]["scored"]) if team_data["history"] else 0,
                "goals_against": int(team_data["history"][-1]["missed"]) if team_data["history"] else 0,
                "xG_diff": float(team_data["history"][-1]["xG"]) - float(team_data["history"][-1]["xGA"]) if team_data["history"] else 0,
            })
        
        df = pd.DataFrame(teams_list)
        df["xG_per_game"] = df["xG"] / df["matches"].replace(0, 1)
        df["xGA_per_game"] = df["xGA"] / df["matches"].replace(0, 1)
        
        return df.sort_values("xG_diff", ascending=False)


class FootballDataScraper:
    """
    Scraper for Football-Data.co.uk (Historical data since 1993)
    Free CSV downloads, unlimited
    """
    
    def __init__(self):
        self.base_url = DATA_SOURCES["football_data"]
        
    def get_season_data(self, league_code: str, season: str) -> pd.DataFrame:
        """
        Download historical data for a season.
        
        Args:
            league_code: League code (E0=EPL, SP1=La Liga, D1=Bundesliga, I1=Serie A, F1=Ligue 1, N1=Eredivisie)
            season: Season in format '2425' for 2024/25
        """
        url = f"{self.base_url}/mmz4281/{season}/{league_code}.csv"
        
        try:
            df = pd.read_csv(url, encoding='latin-1')
            return df
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return pd.DataFrame()
    
    def get_referee_stats(self, league_code: str, seasons: List[str] = None) -> pd.DataFrame:
        """
        Get referee statistics from historical data.
        
        Args:
            league_code: League code
            seasons: List of seasons to analyze (e.g., ['2324', '2425'])
        """
        if seasons is None:
            seasons = ['2223', '2324', '2425']
        
        all_data = []
        for season in seasons:
            df = self.get_season_data(league_code, season)
            if not df.empty and 'Referee' in df.columns:
                all_data.append(df)
        
        if not all_data:
            return pd.DataFrame()
        
        combined = pd.concat(all_data, ignore_index=True)
        
        # Calculate referee stats
        ref_stats = combined.groupby('Referee').agg({
            'HY': 'sum',  # Home Yellow
            'AY': 'sum',  # Away Yellow
            'HR': 'sum',  # Home Red
            'AR': 'sum',  # Away Red
            'HF': 'sum',  # Home Fouls
            'AF': 'sum',  # Away Fouls
            'HomeTeam': 'count',  # Number of matches
        }).rename(columns={'HomeTeam': 'matches'})
        
        ref_stats['total_yellows'] = ref_stats['HY'] + ref_stats['AY']
        ref_stats['total_reds'] = ref_stats['HR'] + ref_stats['AR']
        ref_stats['cards_per_game'] = (ref_stats['total_yellows'] + ref_stats['total_reds']) / ref_stats['matches']
        ref_stats['yellows_per_game'] = ref_stats['total_yellows'] / ref_stats['matches']
        ref_stats['home_yellow_bias'] = ref_stats['HY'] / ref_stats['total_yellows'].replace(0, 1)
        
        return ref_stats.sort_values('cards_per_game', ascending=False)
    
    def get_historical_odds(self, league_code: str, season: str) -> pd.DataFrame:
        """Get historical closing odds for backtesting."""
        df = self.get_season_data(league_code, season)
        
        if df.empty:
            return df
        
        # Select odds columns (Bet365, Pinnacle available in most files)
        odds_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
        
        # Pinnacle odds (sharpest - best for CLV)
        pinnacle_cols = ['PSH', 'PSD', 'PSA']  # Pinnacle Home/Draw/Away
        bet365_cols = ['B365H', 'B365D', 'B365A']  # Bet365 odds
        
        available_cols = [c for c in odds_cols + pinnacle_cols + bet365_cols if c in df.columns]
        
        return df[available_cols]


class ClubEloScraper:
    """
    Scraper for ClubElo.com (Team ELO ratings)
    Free, unlimited
    """
    
    def __init__(self):
        self.base_url = DATA_SOURCES["club_elo"]
        
    def get_current_ratings(self) -> pd.DataFrame:
        """Get current ELO ratings for all teams."""
        url = f"{self.base_url}/api/ranking"
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                # ClubElo returns CSV
                from io import StringIO
                df = pd.read_csv(StringIO(response.text))
                return df
        except Exception as e:
            print(f"Error: {e}")
        
        return pd.DataFrame()
    
    def get_team_history(self, team: str) -> pd.DataFrame:
        """Get historical ELO ratings for a team."""
        url = f"{self.base_url}/{team}"
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                from io import StringIO
                df = pd.read_csv(StringIO(response.text))
                return df
        except Exception as e:
            print(f"Error: {e}")
        
        return pd.DataFrame()


class WeatherScraper:
    """
    Scraper for OpenWeatherMap (Weather data)
    Free tier: 1000 calls/day
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or API_KEYS.get("weather_api")
        self.base_url = "https://api.openweathermap.org/data/2.5"
        
        # Stadium coordinates for major teams
        self.stadium_coords = {
            # Premier League
            "Arsenal": (51.5549, -0.1084),
            "Chelsea": (51.4817, -0.1910),
            "Liverpool": (53.4308, -2.9608),
            "Manchester City": (53.4831, -2.2004),
            "Manchester United": (53.4631, -2.2913),
            "Tottenham": (51.6043, -0.0664),
            # Add more as needed
        }
    
    def get_forecast(self, lat: float, lon: float, date: datetime) -> Dict:
        """Get weather forecast for a location and date."""
        if not self.api_key:
            return {"error": "No API key provided"}
        
        url = f"{self.base_url}/forecast"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "metric",
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                
                # Find forecast closest to match time
                forecasts = data.get("list", [])
                for forecast in forecasts:
                    forecast_time = datetime.fromtimestamp(forecast["dt"])
                    if abs((forecast_time - date).total_seconds()) < 10800:  # Within 3 hours
                        return {
                            "temperature": forecast["main"]["temp"],
                            "humidity": forecast["main"]["humidity"],
                            "wind_speed": forecast["wind"]["speed"],
                            "rain": forecast.get("rain", {}).get("3h", 0),
                            "description": forecast["weather"][0]["description"],
                        }
                
                return {"error": "No forecast available for this date"}
            else:
                return {"error": response.text}
        except Exception as e:
            return {"error": str(e)}
    
    def get_match_weather(self, home_team: str, match_datetime: datetime) -> Dict:
        """Get weather forecast for a match."""
        coords = self.stadium_coords.get(home_team)
        
        if not coords:
            return {"error": f"No coordinates for {home_team}"}
        
        return self.get_forecast(coords[0], coords[1], match_datetime)


class SoccerDataWrapper:
    """
    Wrapper for the soccerdata library.
    Provides easy access to FBref, Understat, WhoScored, etc.
    """
    
    def __init__(self):
        self.has_soccerdata = HAS_SOCCERDATA
        
    def get_fbref_stats(self, league: str, season: str) -> Dict[str, pd.DataFrame]:
        """Get comprehensive stats from FBref."""
        if not self.has_soccerdata:
            return {"error": "soccerdata not installed"}
        
        try:
            fbref = sd.FBref(leagues=league, seasons=season)
            
            return {
                "schedule": fbref.read_schedule(),
                "standings": fbref.read_standings(),
                "team_stats": fbref.read_team_season_stats(stat_type="standard"),
                "player_stats": fbref.read_player_season_stats(stat_type="standard"),
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_sofascore_data(self, league: str, season: str) -> pd.DataFrame:
        """Get data from SofaScore."""
        if not self.has_soccerdata:
            return pd.DataFrame()
        
        try:
            sofascore = sd.SoFIFA(leagues=league, seasons=season)
            return sofascore.read_player_ratings()
        except Exception as e:
            print(f"Error: {e}")
        return pd.DataFrame()


class TeamNewsLoader:
    """
    Load injuries, suspensions, and key player info from local JSON files.
    """

    def __init__(self, base_dir: str = None):
        self.base_dir = base_dir or PATHS["news_data"]

    def get_team_news(self, team_name: str, league: str) -> Dict:
        """Return normalized team news for a team."""
        news_blob = self._load_news_blob(league)
        teams_section = news_blob.get("teams", news_blob)
        team_data = teams_section.get(team_name, {})
        return self._summarize_team_news(team_data)

    def _load_news_blob(self, league: str) -> Dict:
        """Load league-specific news file or fall back to global."""
        league_path = os.path.join(self.base_dir, f"{league}.json")
        global_path = os.path.join(self.base_dir, "global.json")
        for path in (league_path, global_path):
            if os.path.exists(path):
                with open(path, "r") as handle:
                    return json.load(handle)
        return {}

    def _summarize_team_news(self, team_data: Dict) -> Dict:
        injuries = team_data.get("injuries", [])
        suspensions = team_data.get("suspensions", [])
        doubts = team_data.get("doubts", [])
        key_players = team_data.get("key_players", [])
        form = team_data.get("form", {})
        card_risk = team_data.get("card_risk", 0)

        injury_impact = self._sum_impact(injuries)
        suspension_impact = self._sum_impact(suspensions)
        doubtful_impact = self._sum_impact(doubts) * NEWS_SETTINGS["doubtful_weight"]

        absence_impact = injury_impact + suspension_impact + doubtful_impact
        absence_impact = min(absence_impact, NEWS_SETTINGS["max_absence_impact"])

        return {
            "injuries": injuries,
            "suspensions": suspensions,
            "doubts": doubts,
            "key_players": key_players,
            "summary": {
                "absence_impact": absence_impact,
                "key_players_out": len([player for player in key_players if player.get("status") == "out"]),
                "card_risk": card_risk,
                "form": {
                    "shots_per90": form.get("shots_per90", 0.0),
                    "xg_for_last5": form.get("xg_for_last5", 0.0),
                    "xg_against_last5": form.get("xg_against_last5", 0.0),
                },
            },
        }

    @staticmethod
    def _sum_impact(entries: List[Dict]) -> float:
        return sum(entry.get("impact", 0.04) for entry in entries)


class DataAggregator:
    """
    Aggregates data from all sources into a unified format.
    """
    
    def __init__(self):
        self.odds_scraper = OddsAPIScraper()
        self.understat = UnderstatScraper()
        self.football_data = FootballDataScraper()
        self.club_elo = ClubEloScraper()
        self.weather = WeatherScraper()
        self.soccerdata = SoccerDataWrapper()
        self.news_loader = TeamNewsLoader()
    
    def get_match_data(self, home_team: str, away_team: str, 
                       league: str, match_datetime: datetime = None) -> Dict:
        """
        Aggregate all available data for a match.
        
        Returns comprehensive match analysis data.
        """
        league_config = LEAGUES.get(league, {})
        
        data = {
            "home_team": home_team,
            "away_team": away_team,
            "league": league,
            "data_collected_at": datetime.now().isoformat(),
        }
        
        # 1. Get xG data from Understat
        understat_league = league_config.get("understat_name")
        if understat_league:
            xg_data = self.understat.get_team_xg(understat_league)
            if not xg_data.empty:
                home_xg = xg_data[xg_data["team"].str.contains(home_team, case=False, na=False)]
                away_xg = xg_data[xg_data["team"].str.contains(away_team, case=False, na=False)]
                
                data["xg"] = {
                    "home": home_xg.to_dict('records')[0] if not home_xg.empty else None,
                    "away": away_xg.to_dict('records')[0] if not away_xg.empty else None,
                }
        
        # 2. Get ELO ratings
        elo_data = self.club_elo.get_current_ratings()
        if not elo_data.empty:
            data["elo"] = {
                "home": None,
                "away": None,
            }
            # Try to find teams in ELO data
            for _, row in elo_data.iterrows():
                if home_team.lower() in str(row.get('Club', '')).lower():
                    data["elo"]["home"] = row.get('Elo', None)
                if away_team.lower() in str(row.get('Club', '')).lower():
                    data["elo"]["away"] = row.get('Elo', None)
        
        # 3. Get weather data
        if match_datetime:
            data["weather"] = self.weather.get_match_weather(home_team, match_datetime)
        
        # 4. Get referee stats
        fd_code = league_config.get("football_data_code")
        if fd_code:
            ref_stats = self.football_data.get_referee_stats(fd_code)
            data["referee_stats_available"] = not ref_stats.empty

        # 5. Get team news and availability (local data)
        home_news = self.news_loader.get_team_news(home_team, league)
        away_news = self.news_loader.get_team_news(away_team, league)
        if home_news or away_news:
            data["team_news"] = {
                "home": home_news,
                "away": away_news,
            }
        
        return data

    def assess_research_quality(
        self,
        match_data: Dict,
        match_datetime: datetime = None,
    ) -> Dict:
        """
        Assess data coverage and research quality for a match.
        """
        gaps = []
        sources = []
        score = 0

        xg_data = match_data.get("xg", {})
        has_home_xg = bool(xg_data.get("home"))
        has_away_xg = bool(xg_data.get("away"))

        if has_home_xg and has_away_xg:
            score += 35
            sources.append("understat_xg")
        elif has_home_xg or has_away_xg:
            score += 15
            sources.append("understat_xg_partial")
            gaps.append("xG data missing for one team")
        else:
            gaps.append("xG data missing for both teams")

        elo_data = match_data.get("elo", {})
        has_home_elo = bool(elo_data.get("home"))
        has_away_elo = bool(elo_data.get("away"))

        if has_home_elo and has_away_elo:
            score += 20
            sources.append("club_elo")
        elif has_home_elo or has_away_elo:
            score += 8
            sources.append("club_elo_partial")
            gaps.append("ELO rating missing for one team")
        else:
            gaps.append("ELO ratings missing for both teams")

        if match_data.get("referee_stats_available"):
            score += 10
            sources.append("football_data_referee")
        else:
            gaps.append("Referee stats unavailable")

        team_news = match_data.get("team_news", {})
        if team_news.get("home") and team_news.get("away"):
            score += 15
            sources.append("team_news")
        elif team_news.get("home") or team_news.get("away"):
            score += 7
            sources.append("team_news_partial")
            gaps.append("Team news missing for one side")
        else:
            gaps.append("Team news unavailable")

        if match_datetime:
            if match_data.get("weather"):
                score += 5
                sources.append("weather")
            else:
                gaps.append("Weather data unavailable for kickoff time")

        if sources:
            score += 20

        score = min(score, 100)

        if score >= 80:
            grade = "A"
        elif score >= 65:
            grade = "B"
        elif score >= 50:
            grade = "C"
        else:
            grade = "D"

        return {
            "score": score,
            "grade": grade,
            "sources": sources,
            "gaps": gaps,
        }

    def format_research_report(self, research: Dict, prediction_confidence: str) -> str:
        """
        Format research quality info into a readable report.
        """
        report = []
        report.append("🧠 RESEARCH QUALITY")
        report.append(
            f"   Score: {research['score']}/100 | Grade: {research['grade']} "
            f"| Model Confidence: {prediction_confidence}"
        )
        if research["sources"]:
            report.append(f"   Sources: {', '.join(research['sources'])}")
        if research["gaps"]:
            report.append(f"   Gaps: {', '.join(research['gaps'])}")
        return "\n".join(report)
    
    def save_data(self, data: Dict, filename: str):
        """Save data to JSON file."""
        os.makedirs(PATHS["data_dir"], exist_ok=True)
        filepath = os.path.join(PATHS["data_dir"], filename)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"Data saved to {filepath}")


# =============================================================================
# USAGE EXAMPLE
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("BETTING DATA SCRAPERS - TEST RUN")
    print("=" * 60)
    
    # Test Football-Data.co.uk (always works, no API key needed)
    print("\n1. Testing Football-Data.co.uk...")
    fd = FootballDataScraper()
    ref_stats = fd.get_referee_stats("E0", ["2324"])
    if not ref_stats.empty:
        print("✓ Referee stats loaded successfully!")
        print(f"  Found {len(ref_stats)} referees")
        print(f"  Highest cards/game: {ref_stats['cards_per_game'].max():.2f}")
    
    # Test Understat
    print("\n2. Testing Understat...")
    understat = UnderstatScraper()
    xg_data = understat.get_team_xg("EPL", "2024")
    if not xg_data.empty:
        print("✓ xG data loaded successfully!")
        print(f"  Found {len(xg_data)} teams")
        print(f"  Top xG team: {xg_data.iloc[0]['team']}")
    
    # Test Club ELO
    print("\n3. Testing Club ELO...")
    elo = ClubEloScraper()
    elo_data = elo.get_current_ratings()
    if not elo_data.empty:
        print("✓ ELO ratings loaded successfully!")
        print(f"  Found {len(elo_data)} teams")
    
    print("\n" + "=" * 60)
    print("All scrapers ready! Configure API keys in config.py for full functionality.")
    print("=" * 60)
