"""
MAIN APPLICATION
================
Entry point for the betting system.
"""

import argparse
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd

from config import LEAGUES, BANKROLL, PATHS, NEWS_SETTINGS
from model import BettingModel, TeamStats, MatchContext, LiveMatchState
from scrapers import DataAggregator, OddsAPIScraper, UnderstatScraper, FootballDataScraper
from tracker import BetTracker, CLVAnalyzer, BankrollManager
from backtest import Backtester, BacktestConfig, ParameterOptimizer
from enhanced_system import EnhancedBettingSystem


class BettingSystem:
    """
    Main betting system that orchestrates all components.
    """
    
    def __init__(self):
        self.model = BettingModel()
        self.aggregator = DataAggregator()
        self.tracker = BetTracker()
        self.bankroll = BankrollManager()
        self.clv_analyzer = CLVAnalyzer(self.tracker)
    
    def analyze_match(
        self,
        home_team: str,
        away_team: str,
        league: str,
        odds: Dict = None,
        match_datetime: datetime = None,
        live_state: Dict = None,
    ) -> Dict:
        """
        Full analysis of a single match.
        """
        # Get team data
        match_data = self.aggregator.get_match_data(
            home_team,
            away_team,
            league,
            match_datetime=match_datetime,
        )
        
        # Create team stats from data
        home_stats = self._create_team_stats(home_team, match_data.get("xg", {}).get("home"))
        away_stats = self._create_team_stats(away_team, match_data.get("xg", {}).get("away"))
        
        # Add ELO if available
        elo_data = match_data.get("elo", {})
        if elo_data.get("home"):
            home_stats.elo_rating = elo_data["home"]
        if elo_data.get("away"):
            away_stats.elo_rating = elo_data["away"]
        
        context = self._build_match_context(match_data)

        # Generate prediction
        if live_state:
            live = LiveMatchState(
                minute=live_state.get("minute", 0),
                home_goals=live_state.get("home_goals", 0),
                away_goals=live_state.get("away_goals", 0),
            )
            prediction = self.model.predict_live_match(home_stats, away_stats, live, context=context)
        else:
            prediction = self.model.predict_match(home_stats, away_stats, context=context)

        # Research quality assessment
        research = self.aggregator.assess_research_quality(
            match_data,
            match_datetime=match_datetime,
        )
        if research["score"] >= 75 and prediction.confidence == "HIGH":
            research["recommendation"] = "HIGH_CONFIDENCE_SHORTLIST"
        else:
            research["recommendation"] = "STANDARD_REVIEW"
        
        # Find value bets
        value_bets = []
        if odds:
            value_bets = self.model.find_value_bets(prediction, odds)

        report = self.model.generate_report(home_stats, away_stats, prediction, odds)
        if live_state:
            report = "\n".join(
                [
                    report,
                    "",
                    (
                        "⏱️ LIVE STATE"
                        f"\n   Minute: {live_state.get('minute', 0)}"
                        f"\n   Score: {home_team} {live_state.get('home_goals', 0)}"
                        f"-{live_state.get('away_goals', 0)} {away_team}"
                    ),
                ]
            )
        report = "\n".join(
            [
                report,
                "",
                self.aggregator.format_research_report(
                    research, prediction.confidence
                ),
            ]
        )
        
        return {
            "match": f"{home_team} vs {away_team}",
            "league": league,
            "prediction": prediction,
            "value_bets": value_bets,
            "research": research,
            "live_state": live_state,
            "report": report,
            "raw_data": match_data,
        }
    
    def _create_team_stats(self, team_name: str, xg_data: Dict) -> TeamStats:
        """Create TeamStats from scraped data."""
        if xg_data is None:
            return TeamStats(name=team_name)
        
        return TeamStats(
            name=team_name,
            xg_for=xg_data.get("xG", 0),
            xg_against=xg_data.get("xGA", 0),
            goals_scored=xg_data.get("goals", 0),
            goals_conceded=xg_data.get("goals_against", 0),
            matches_played=xg_data.get("matches", 0),
        )

    def _build_match_context(self, match_data: Dict) -> MatchContext:
        """Build MatchContext from available data sources."""
        weather_condition = self._normalize_weather(match_data.get("weather", {}))
        team_news = match_data.get("team_news", {})
        home_summary = team_news.get("home", {}).get("summary", {})
        away_summary = team_news.get("away", {}).get("summary", {})

        home_form = self._calculate_attack_form(home_summary.get("form", {}))
        away_form = self._calculate_attack_form(away_summary.get("form", {}))
        home_defense = self._calculate_defense_form(home_summary.get("form", {}))
        away_defense = self._calculate_defense_form(away_summary.get("form", {}))

        return MatchContext(
            weather_condition=weather_condition,
            home_key_players_out=home_summary.get("key_players_out", 0),
            away_key_players_out=away_summary.get("key_players_out", 0),
            home_absence_impact=home_summary.get("absence_impact", 0.0),
            away_absence_impact=away_summary.get("absence_impact", 0.0),
            home_attack_form=home_form,
            away_attack_form=away_form,
            home_defense_form=home_defense,
            away_defense_form=away_defense,
            home_card_risk=home_summary.get("card_risk", 0.0),
            away_card_risk=away_summary.get("card_risk", 0.0),
        )

    @staticmethod
    def _normalize_weather(weather: Dict) -> str:
        """Map weather data to model-friendly labels."""
        if not weather or weather.get("error"):
            return "normal"

        wind = weather.get("wind_speed", 0)
        rain = weather.get("rain", 0)
        temp = weather.get("temperature", 15)

        if rain >= 5:
            return "rain_heavy"
        if wind >= 12:
            return "wind_strong"
        if temp >= 30:
            return "extreme_heat"
        if temp <= 0:
            return "extreme_cold"
        return "normal"

    @staticmethod
    def _calculate_attack_form(form: Dict) -> float:
        """Calculate attacking form adjustment based on shots/xG trends."""
        shots = form.get("shots_per90", 0.0)
        xg_for = form.get("xg_for_last5", 0.0)

        shot_delta = shots - NEWS_SETTINGS["league_avg_shots_per90"]
        xg_delta = xg_for - NEWS_SETTINGS["league_avg_xg_last5"]

        adjustment = (
            shot_delta * NEWS_SETTINGS["form_shots_boost"]
            + (xg_delta / 0.1) * NEWS_SETTINGS["form_xg_boost"]
        )
        return max(-NEWS_SETTINGS["max_form_adjustment"],
                   min(NEWS_SETTINGS["max_form_adjustment"], adjustment))

    @staticmethod
    def _calculate_defense_form(form: Dict) -> float:
        """Calculate defensive form adjustment based on xGA trends."""
        xg_against = form.get("xg_against_last5", 0.0)
        delta = NEWS_SETTINGS["league_avg_xg_against_last5"] - xg_against
        adjustment = (delta / 0.1) * NEWS_SETTINGS["form_xg_boost"]
        return max(-NEWS_SETTINGS["max_form_adjustment"],
                   min(NEWS_SETTINGS["max_form_adjustment"], adjustment))
    
    def record_bet(self, match: str, league: str, market: str,
                   selection: str, odds: float, stake: float,
                   our_probability: float, bookmaker: str = "unknown") -> Dict:
        """Record a new bet."""
        bet = self.tracker.add_bet(
            match=match,
            league=league,
            market=market,
            selection=selection,
            odds=odds,
            stake=stake,
            bankroll=self.bankroll.current_bankroll,
            our_probability=our_probability,
            bookmaker=bookmaker,
        )
        
        return {
            "bet_id": bet.id,
            "message": f"Bet recorded: {selection} @ {odds} for €{stake}",
        }
    
    def settle_bet(self, bet_id: str, result: str, 
                   closing_odds: float = None) -> Dict:
        """Settle a bet with result and optional closing odds."""
        bet = self.tracker.update_result(bet_id, result, closing_odds)
        
        if bet:
            self.bankroll.update_bankroll(bet.profit_loss or 0)
            
            return {
                "success": True,
                "profit_loss": bet.profit_loss,
                "clv": bet.clv,
                "new_bankroll": self.bankroll.current_bankroll,
            }
        
        return {"success": False, "error": "Bet not found"}
    
    def get_performance_report(self) -> str:
        """Get comprehensive performance report."""
        report = []
        report.append(self.tracker.generate_report())
        report.append("\n")
        report.append(self.clv_analyzer.generate_clv_report())
        report.append(f"\n📊 Current Bankroll: €{self.bankroll.current_bankroll:.2f}")
        report.append(f"📈 ROI: {self.bankroll.get_roi():.2f}%")
        report.append(f"📉 Max Drawdown: {self.bankroll.get_drawdown():.2f}%")
        
        return "\n".join(report)
    
    def run_backtest(self, league: str, seasons: List[str]) -> str:
        """Run backtest for a league."""
        league_config = LEAGUES.get(league, {})
        league_code = league_config.get("football_data_code")
        
        if not league_code:
            return f"Unknown league: {league}"
        
        config = BacktestConfig(
            starting_bankroll=BANKROLL["starting_amount"],
            kelly_fraction=BANKROLL["kelly_fraction"],
            max_bet_percent=BANKROLL["max_bet_percent"],
            min_edge=3.0,
        )
        
        backtester = Backtester(config)
        result = backtester.run_backtest(league_code, seasons)
        
        return backtester.generate_report(result)


class LineShopper:
    """
    Compare odds across bookmakers to find best value.
    """
    
    def __init__(self):
        self.odds_scraper = OddsAPIScraper()
    
    def find_best_odds(self, league: str) -> List[Dict]:
        """
        Find best odds for all upcoming matches in a league.
        """
        matches = self.odds_scraper.get_upcoming_matches(league)
        return self.odds_scraper.find_best_odds(matches)
    
    def compare_odds(self, match_data: Dict) -> str:
        """Generate odds comparison report for a match."""
        report = []
        report.append(f"\n🔍 ODDS COMPARISON: {match_data.get('home_team')} vs {match_data.get('away_team')}")
        report.append("-" * 50)
        
        bookmakers = match_data.get("bookmakers", {})
        
        # Collect all odds for comparison
        home_odds = []
        draw_odds = []
        away_odds = []
        
        for book, markets in bookmakers.items():
            h2h = markets.get("h2h", {})
            if h2h:
                home = h2h.get(match_data.get("home_team"), 0)
                draw = h2h.get("Draw", 0)
                away = h2h.get(match_data.get("away_team"), 0)
                
                if home:
                    home_odds.append((book, home))
                if draw:
                    draw_odds.append((book, draw))
                if away:
                    away_odds.append((book, away))
        
        # Sort by odds (highest first)
        home_odds.sort(key=lambda x: x[1], reverse=True)
        draw_odds.sort(key=lambda x: x[1], reverse=True)
        away_odds.sort(key=lambda x: x[1], reverse=True)
        
        report.append("\nHOME WIN:")
        for book, odds in home_odds[:5]:
            marker = "👑" if odds == home_odds[0][1] else "  "
            report.append(f"  {marker} {book}: {odds:.2f}")
        
        report.append("\nDRAW:")
        for book, odds in draw_odds[:5]:
            marker = "👑" if odds == draw_odds[0][1] else "  "
            report.append(f"  {marker} {book}: {odds:.2f}")
        
        report.append("\nAWAY WIN:")
        for book, odds in away_odds[:5]:
            marker = "👑" if odds == away_odds[0][1] else "  "
            report.append(f"  {marker} {book}: {odds:.2f}")
        
        # Calculate overround
        if home_odds and draw_odds and away_odds:
            best_home = home_odds[0][1]
            best_draw = draw_odds[0][1]
            best_away = away_odds[0][1]
            
            overround = (1/best_home + 1/best_draw + 1/best_away - 1) * 100
            report.append(f"\n📊 Best Combined Overround: {overround:.2f}%")
            
            if overround < 2:
                report.append("   ✅ Very competitive market")
            elif overround < 5:
                report.append("   ⚠️ Normal market margin")
            else:
                report.append("   ❌ High margin - consider other markets")
        
        return "\n".join(report)


class FixtureAnalyzer:
    """
    Analyze fixture congestion and rest days.
    """
    
    def __init__(self):
        self.scraper = FootballDataScraper()
    
    def analyze_congestion(self, team: str, league_code: str, 
                           match_date: datetime) -> Dict:
        """
        Analyze fixture congestion for a team.
        """
        # Get recent fixtures
        df = self.scraper.get_season_data(league_code, "2425")
        
        if df.empty:
            return {"error": "No data available"}
        
        # Find team's matches
        team_matches = df[
            (df['HomeTeam'].str.contains(team, case=False, na=False)) |
            (df['AwayTeam'].str.contains(team, case=False, na=False))
        ].copy()
        
        if 'Date' in team_matches.columns:
            team_matches['Date'] = pd.to_datetime(team_matches['Date'], dayfirst=True)
        
        # Matches in last 30 days
        thirty_days_ago = match_date - timedelta(days=30)
        recent_matches = team_matches[
            (team_matches['Date'] >= thirty_days_ago) &
            (team_matches['Date'] < match_date)
        ]
        
        # Days since last match
        last_match = team_matches[team_matches['Date'] < match_date]['Date'].max()
        days_rest = (match_date - last_match).days if pd.notna(last_match) else 7
        
        return {
            "team": team,
            "matches_last_30_days": len(recent_matches),
            "days_since_last_match": days_rest,
            "fatigue_warning": len(recent_matches) >= 8 or days_rest <= 3,
        }


def create_sample_workflow():
    """Create a sample workflow showing system usage."""
    workflow = """
╔══════════════════════════════════════════════════════════════════╗
║                    BETTING SYSTEM WORKFLOW                        ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  1. DATA COLLECTION (Daily)                                       ║
║     └── Run scrapers to collect xG, ELO, odds, injuries          ║
║                                                                   ║
║  2. MATCH ANALYSIS (Before each matchday)                         ║
║     ├── Generate predictions using Poisson model                 ║
║     ├── Apply context factors (motivation, fatigue, weather)     ║
║     └── Calculate probabilities for all markets                  ║
║                                                                   ║
║  3. VALUE IDENTIFICATION                                          ║
║     ├── Compare model probabilities vs bookmaker odds            ║
║     ├── Find bets with edge > 3%                                 ║
║     └── Use line shopping to get best odds                       ║
║                                                                   ║
║  4. BET PLACEMENT                                                 ║
║     ├── Calculate stake using Half Kelly                         ║
║     ├── Record bet with odds, stake, and probability             ║
║     └── Track which bookmaker used                               ║
║                                                                   ║
║  5. POST-MATCH                                                    ║
║     ├── Record closing odds (from Pinnacle)                      ║
║     ├── Update bet result (won/lost)                             ║
║     ├── Calculate CLV for each bet                               ║
║     └── Update bankroll                                          ║
║                                                                   ║
║  6. PERFORMANCE REVIEW (Weekly)                                   ║
║     ├── Review ROI and profit/loss                               ║
║     ├── Analyze CLV positive rate (target: >50%)                 ║
║     ├── Check performance by market/league                       ║
║     └── Adjust strategy if needed                                ║
║                                                                   ║
║  7. BACKTESTING (Monthly)                                         ║
║     ├── Test model on new historical data                        ║
║     ├── Optimize parameters                                      ║
║     └── Validate edge still exists                               ║
║                                                                   ║
╚══════════════════════════════════════════════════════════════════╝
"""
    return workflow


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Betting Analysis System")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Analyze match
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a match")
    analyze_parser.add_argument("--home", required=True, help="Home team")
    analyze_parser.add_argument("--away", required=True, help="Away team")
    analyze_parser.add_argument("--league", required=True, help="League key")
    analyze_parser.add_argument("--kickoff", help="Kickoff time (YYYY-mm-dd HH:MM)")
    analyze_parser.add_argument("--minute", type=int, help="Live minute (0-90)")
    analyze_parser.add_argument("--home-goals", type=int, default=0, help="Live home goals")
    analyze_parser.add_argument("--away-goals", type=int, default=0, help="Live away goals")
    
    # Backtest
    backtest_parser = subparsers.add_parser("backtest", help="Run backtest")
    backtest_parser.add_argument("--league", required=True, help="League key")
    backtest_parser.add_argument("--seasons", nargs="+", required=True, help="Seasons (e.g., 2324)")
    
    # Performance
    perf_parser = subparsers.add_parser("performance", help="Show performance")
    
    # Line shop
    lines_parser = subparsers.add_parser("lines", help="Compare odds")
    lines_parser.add_argument("--league", required=True, help="League key")
    
    # Workflow
    workflow_parser = subparsers.add_parser("workflow", help="Show workflow")
    # Enhanced system
    enhanced_parser = subparsers.add_parser("enhanced", help="Run enhanced ML system")
    enhanced_parser.add_argument(
        "--mode", choices=["train", "predict", "analyze"], required=True,
        help="Operating mode"
    )
    enhanced_parser.add_argument(
        "--league", type=str, required=True,
        help="League code (e.g., E0, N1, T1)"
    )
    enhanced_parser.add_argument(
        "--start-year", type=int, default=2018,
        help="Start year for historical data"
    )
    enhanced_parser.add_argument(
        "--fixtures-file", type=str, default=None,
        help="JSON file with fixtures to predict"
    )
    
    args = parser.parse_args()
    
    if args.command == "analyze":
        system = BettingSystem()
        match_datetime = None
        if args.kickoff:
            match_datetime = datetime.strptime(args.kickoff, "%Y-%m-%d %H:%M")

        live_state = None
        if args.minute is not None:
            live_state = {
                "minute": args.minute,
                "home_goals": args.home_goals,
                "away_goals": args.away_goals,
            }

        result = system.analyze_match(
            args.home,
            args.away,
            args.league,
            match_datetime=match_datetime,
            live_state=live_state,
        )
        print(result["report"])
    
    elif args.command == "backtest":
        system = BettingSystem()
        report = system.run_backtest(args.league, args.seasons)
        print(report)
    
    elif args.command == "performance":
        system = BettingSystem()
        print(system.get_performance_report())
    
    elif args.command == "lines":
        shopper = LineShopper()
        matches = shopper.find_best_odds(args.league)
        for match in matches[:5]:
            print(f"\n{match['home_team']} vs {match['away_team']}")
            print(f"  Best Home: {match['best_home_odds']['odds']:.2f} @ {match['best_home_odds']['bookmaker']}")
            print(f"  Best Draw: {match['best_draw_odds']['odds']:.2f} @ {match['best_draw_odds']['bookmaker']}")
            print(f"  Best Away: {match['best_away_odds']['odds']:.2f} @ {match['best_away_odds']['bookmaker']}")
    
    elif args.command == "workflow":
        print(create_sample_workflow())
    elif args.command == "enhanced":
        system = EnhancedBettingSystem()
        if args.mode == "train":
            system.train(args.league, start_year=args.start_year)
        elif args.mode == "analyze":
            system.analyze_league(args.league, start_year=args.start_year)
        elif args.mode == "predict":
            if not args.fixtures_file:
                raise SystemExit("Provide --fixtures-file with match fixtures to predict")
            with open(args.fixtures_file) as f:
                fixtures = json.load(f)
            system.predict_matches(args.league, fixtures, start_year=args.start_year)
    
    else:
        parser.print_help()
        print("\n" + create_sample_workflow())


if __name__ == "__main__":
    main()
