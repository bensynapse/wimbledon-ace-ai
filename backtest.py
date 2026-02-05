"""
BACKTESTING FRAMEWORK
=====================
Test betting strategies on historical data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
import json
import os

from config import PATHS, BANKROLL, LEAGUES, VALUE_THRESHOLDS
from model import BettingModel, TeamStats, MatchContext, Prediction, KellyCalculator
from scrapers import FootballDataScraper


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    start_date: str
    end_date: str
    total_matches: int
    total_bets: int
    bets_won: int
    bets_lost: int
    total_staked: float
    total_returns: float
    profit_loss: float
    roi: float
    yield_per_bet: float
    win_rate: float
    avg_odds: float
    max_drawdown: float
    sharpe_ratio: float
    clv_positive_rate: float
    avg_clv: float
    bankroll_history: List[float]
    bets_log: List[Dict]


@dataclass
class BacktestConfig:
    """Configuration for a backtest."""
    starting_bankroll: float = 1000.0
    kelly_fraction: float = 0.5
    max_bet_percent: float = 5.0
    min_edge: float = 3.0
    min_odds: float = 1.30
    max_odds: float = 4.00
    markets: List[str] = field(default_factory=lambda: ["1X2", "Over/Under", "BTTS"])
    use_closing_odds: bool = True  # Use closing odds for CLV calculation


class Backtester:
    """
    Backtest betting strategies on historical data.
    """
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.model = BettingModel()
        self.kelly = KellyCalculator(
            fraction=self.config.kelly_fraction,
            max_bet_percent=self.config.max_bet_percent
        )
        self.scraper = FootballDataScraper()
    
    def load_historical_data(self, league_code: str, 
                             seasons: List[str]) -> pd.DataFrame:
        """
        Load historical match data from Football-Data.co.uk.
        
        Args:
            league_code: E0 (EPL), SP1 (La Liga), D1 (Bundesliga), etc.
            seasons: List of seasons like ['2223', '2324']
        """
        all_data = []
        
        for season in seasons:
            df = self.scraper.get_season_data(league_code, season)
            if not df.empty:
                df['Season'] = season
                all_data.append(df)
        
        if not all_data:
            return pd.DataFrame()
        
        combined = pd.concat(all_data, ignore_index=True)
        
        # Ensure date column is datetime
        if 'Date' in combined.columns:
            combined['Date'] = pd.to_datetime(combined['Date'], dayfirst=True, errors='coerce')
        
        return combined.sort_values('Date').reset_index(drop=True)
    
    def calculate_team_stats(self, df: pd.DataFrame, team: str, 
                             before_date: datetime, 
                             num_matches: int = 10) -> TeamStats:
        """
        Calculate team statistics from historical data.
        """
        # Get team's matches before this date
        home_matches = df[(df['HomeTeam'] == team) & (df['Date'] < before_date)]
        away_matches = df[(df['AwayTeam'] == team) & (df['Date'] < before_date)]
        
        # Combine and sort
        team_matches = pd.concat([
            home_matches.assign(is_home=True),
            away_matches.assign(is_home=False)
        ]).sort_values('Date', ascending=False).head(num_matches)
        
        if team_matches.empty:
            return TeamStats(name=team)
        
        # Calculate stats
        goals_for = 0
        goals_against = 0
        
        for _, match in team_matches.iterrows():
            if match['is_home']:
                goals_for += match.get('FTHG', 0) or 0
                goals_against += match.get('FTAG', 0) or 0
            else:
                goals_for += match.get('FTAG', 0) or 0
                goals_against += match.get('FTHG', 0) or 0
        
        matches_played = len(team_matches)
        
        return TeamStats(
            name=team,
            goals_scored=goals_for,
            goals_conceded=goals_against,
            matches_played=matches_played,
            # xG not available in Football-Data.co.uk, use goals as proxy
            xg_for=goals_for,
            xg_against=goals_against,
        )
    
    def simulate_bet(self, prediction: Prediction, actual_result: Dict,
                     odds: Dict, market: str) -> Optional[Dict]:
        """
        Simulate placing a bet and determine outcome.
        
        Args:
            prediction: Model prediction
            actual_result: Actual match result
            odds: Available odds for the match
            market: Market to bet on
        
        Returns:
            Bet result or None if no value found
        """
        # Map predictions to markets
        market_mapping = {
            "1X2": [
                ("Home", prediction.home_win_prob, "home_win"),
                ("Draw", prediction.draw_prob, "draw"),
                ("Away", prediction.away_win_prob, "away_win"),
            ],
            "Over/Under": [
                ("Over 2.5", prediction.over_2_5_prob, "over_2_5"),
                ("Under 2.5", prediction.under_2_5_prob, "under_2_5"),
            ],
            "BTTS": [
                ("BTTS Yes", prediction.btts_yes_prob, "btts_yes"),
                ("BTTS No", prediction.btts_no_prob, "btts_no"),
            ],
        }
        
        if market not in market_mapping:
            return None
        
        best_value = None
        best_edge = self.config.min_edge
        
        for selection, prob, odds_key in market_mapping[market]:
            if odds_key not in odds or odds[odds_key] is None:
                continue
            
            market_odds = odds[odds_key]
            
            # Check odds bounds
            if market_odds < self.config.min_odds or market_odds > self.config.max_odds:
                continue
            
            edge = self.kelly.calculate_edge(prob, market_odds)
            
            if edge > best_edge:
                best_value = {
                    "market": market,
                    "selection": selection,
                    "our_prob": prob,
                    "odds": market_odds,
                    "edge": edge,
                    "odds_key": odds_key,
                }
                best_edge = edge
        
        if not best_value:
            return None
        
        # Determine if bet won
        won = self._check_bet_result(
            best_value["selection"], 
            actual_result
        )
        
        return {
            **best_value,
            "won": won,
        }
    
    def _check_bet_result(self, selection: str, result: Dict) -> bool:
        """Check if a bet selection won."""
        home_goals = result.get("home_goals", 0)
        away_goals = result.get("away_goals", 0)
        total_goals = home_goals + away_goals
        
        if selection == "Home":
            return home_goals > away_goals
        elif selection == "Draw":
            return home_goals == away_goals
        elif selection == "Away":
            return away_goals > home_goals
        elif selection == "Over 2.5":
            return total_goals > 2.5
        elif selection == "Under 2.5":
            return total_goals < 2.5
        elif selection == "BTTS Yes":
            return home_goals > 0 and away_goals > 0
        elif selection == "BTTS No":
            return home_goals == 0 or away_goals == 0
        
        return False
    
    def run_backtest(self, league_code: str, seasons: List[str],
                     progress_callback: Callable = None) -> BacktestResult:
        """
        Run a full backtest on historical data.
        
        Args:
            league_code: Football-Data.co.uk league code
            seasons: List of seasons to test
            progress_callback: Optional callback for progress updates
        """
        # Load data
        df = self.load_historical_data(league_code, seasons)
        
        if df.empty:
            raise ValueError(f"No data found for {league_code} seasons {seasons}")
        
        # Initialize tracking
        bankroll = self.config.starting_bankroll
        bankroll_history = [bankroll]
        bets_log = []
        peak_bankroll = bankroll
        max_drawdown = 0
        
        total_matches = 0
        total_bets = 0
        bets_won = 0
        bets_lost = 0
        total_staked = 0
        total_returns = 0
        total_clv = 0
        clv_positive = 0
        all_odds = []
        
        # Skip first matches to build stats
        min_matches_for_stats = 5
        team_match_counts = {}
        
        for idx, row in df.iterrows():
            if progress_callback and idx % 50 == 0:
                progress_callback(idx / len(df))
            
            home_team = row.get('HomeTeam')
            away_team = row.get('AwayTeam')
            match_date = row.get('Date')
            
            if pd.isna(match_date) or pd.isna(home_team) or pd.isna(away_team):
                continue
            
            # Track match counts
            team_match_counts[home_team] = team_match_counts.get(home_team, 0) + 1
            team_match_counts[away_team] = team_match_counts.get(away_team, 0) + 1
            
            # Skip if not enough history
            if (team_match_counts.get(home_team, 0) < min_matches_for_stats or
                team_match_counts.get(away_team, 0) < min_matches_for_stats):
                continue
            
            total_matches += 1
            
            # Get team stats
            home_stats = self.calculate_team_stats(df, home_team, match_date)
            away_stats = self.calculate_team_stats(df, away_team, match_date)
            
            # Generate prediction
            prediction = self.model.predict_match(home_stats, away_stats)
            
            # Get odds from data
            odds = self._extract_odds(row)
            closing_odds = self._extract_closing_odds(row)
            
            # Actual result
            actual_result = {
                "home_goals": row.get('FTHG', 0) or 0,
                "away_goals": row.get('FTAG', 0) or 0,
            }
            
            # Try each market
            for market in self.config.markets:
                bet_result = self.simulate_bet(prediction, actual_result, odds, market)
                
                if bet_result:
                    # Calculate stake
                    stake = self.kelly.calculate_stake(
                        bankroll, 
                        bet_result["our_prob"],
                        bet_result["odds"]
                    )
                    
                    if stake < 1:  # Minimum €1 bet
                        continue
                    
                    total_bets += 1
                    total_staked += stake
                    all_odds.append(bet_result["odds"])
                    
                    if bet_result["won"]:
                        bets_won += 1
                        profit = stake * (bet_result["odds"] - 1)
                        bankroll += profit
                        total_returns += stake + profit
                    else:
                        bets_lost += 1
                        bankroll -= stake
                    
                    # Track drawdown
                    if bankroll > peak_bankroll:
                        peak_bankroll = bankroll
                    drawdown = (peak_bankroll - bankroll) / peak_bankroll * 100
                    max_drawdown = max(max_drawdown, drawdown)
                    
                    bankroll_history.append(bankroll)
                    
                    # Calculate CLV if closing odds available
                    clv = None
                    if self.config.use_closing_odds and bet_result["odds_key"] in closing_odds:
                        close_odds = closing_odds[bet_result["odds_key"]]
                        if close_odds:
                            clv = ((bet_result["odds"] - close_odds) / close_odds) * 100
                            total_clv += clv
                            if clv > 0:
                                clv_positive += 1
                    
                    bets_log.append({
                        "date": str(match_date),
                        "match": f"{home_team} vs {away_team}",
                        "market": market,
                        "selection": bet_result["selection"],
                        "odds": bet_result["odds"],
                        "stake": stake,
                        "edge": bet_result["edge"],
                        "won": bet_result["won"],
                        "profit": stake * (bet_result["odds"] - 1) if bet_result["won"] else -stake,
                        "clv": clv,
                        "bankroll": bankroll,
                    })
                    
                    # Stop if bankrupt
                    if bankroll < 10:
                        break
            
            if bankroll < 10:
                break
        
        # Calculate final metrics
        profit_loss = bankroll - self.config.starting_bankroll
        roi = (profit_loss / total_staked * 100) if total_staked > 0 else 0
        yield_per_bet = (profit_loss / total_bets) if total_bets > 0 else 0
        win_rate = (bets_won / total_bets * 100) if total_bets > 0 else 0
        avg_odds = np.mean(all_odds) if all_odds else 0
        
        clv_positive_rate = (clv_positive / total_bets * 100) if total_bets > 0 else 0
        avg_clv = (total_clv / total_bets) if total_bets > 0 else 0
        
        # Sharpe ratio (simplified)
        if len(bankroll_history) > 1:
            returns = np.diff(bankroll_history) / bankroll_history[:-1]
            sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
        else:
            sharpe = 0
        
        return BacktestResult(
            start_date=str(df['Date'].min()),
            end_date=str(df['Date'].max()),
            total_matches=total_matches,
            total_bets=total_bets,
            bets_won=bets_won,
            bets_lost=bets_lost,
            total_staked=total_staked,
            total_returns=total_returns,
            profit_loss=profit_loss,
            roi=roi,
            yield_per_bet=yield_per_bet,
            win_rate=win_rate,
            avg_odds=avg_odds,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe,
            clv_positive_rate=clv_positive_rate,
            avg_clv=avg_clv,
            bankroll_history=bankroll_history,
            bets_log=bets_log,
        )
    
    def _extract_odds(self, row: pd.Series) -> Dict:
        """Extract opening/early odds from a data row."""
        return {
            "home_win": row.get('B365H') or row.get('BWH') or row.get('IWH'),
            "draw": row.get('B365D') or row.get('BWD') or row.get('IWD'),
            "away_win": row.get('B365A') or row.get('BWA') or row.get('IWA'),
            "over_2_5": row.get('B365>2.5') or row.get('BbAv>2.5'),
            "under_2_5": row.get('B365<2.5') or row.get('BbAv<2.5'),
            # BTTS not always available, calculate from other data if needed
        }
    
    def _extract_closing_odds(self, row: pd.Series) -> Dict:
        """Extract closing odds (Pinnacle is sharpest)."""
        return {
            "home_win": row.get('PSH') or row.get('PSCH'),
            "draw": row.get('PSD') or row.get('PSCD'),
            "away_win": row.get('PSA') or row.get('PSCA'),
            "over_2_5": row.get('P>2.5'),
            "under_2_5": row.get('P<2.5'),
        }
    
    def generate_report(self, result: BacktestResult) -> str:
        """Generate a comprehensive backtest report."""
        report = []
        report.append("=" * 70)
        report.append("BACKTEST RESULTS")
        report.append("=" * 70)
        report.append(f"Period: {result.start_date} to {result.end_date}")
        report.append("")
        
        # Overview
        report.append("📊 OVERVIEW")
        report.append(f"   Total Matches Analyzed: {result.total_matches}")
        report.append(f"   Total Bets Placed: {result.total_bets}")
        report.append(f"   Won: {result.bets_won} | Lost: {result.bets_lost}")
        report.append(f"   Win Rate: {result.win_rate:.1f}%")
        report.append("")
        
        # Financial Performance
        report.append("💰 FINANCIAL PERFORMANCE")
        report.append(f"   Starting Bankroll: €{self.config.starting_bankroll:.2f}")
        report.append(f"   Final Bankroll: €{result.bankroll_history[-1]:.2f}")
        report.append(f"   Profit/Loss: €{result.profit_loss:.2f}")
        report.append(f"   ROI: {result.roi:.2f}%")
        report.append(f"   Yield per Bet: €{result.yield_per_bet:.2f}")
        report.append(f"   Total Staked: €{result.total_staked:.2f}")
        report.append(f"   Average Odds: {result.avg_odds:.2f}")
        report.append("")
        
        # Risk Metrics
        report.append("📈 RISK METRICS")
        report.append(f"   Max Drawdown: {result.max_drawdown:.1f}%")
        report.append(f"   Sharpe Ratio: {result.sharpe_ratio:.2f}")
        report.append("")
        
        # CLV Analysis (THE KEY METRIC)
        report.append("🎯 CLOSING LINE VALUE (CLV)")
        report.append(f"   CLV Positive Rate: {result.clv_positive_rate:.1f}%")
        report.append(f"   Average CLV: {result.avg_clv:.2f}%")
        
        if result.clv_positive_rate >= 55:
            report.append("   ✅ Excellent! Model consistently beats closing line")
        elif result.clv_positive_rate >= 50:
            report.append("   ✅ Good. Model shows edge over market")
        else:
            report.append("   ⚠️ Warning: Model not consistently beating closing line")
        report.append("")
        
        # Assessment
        report.append("📋 OVERALL ASSESSMENT")
        
        if result.roi > 10:
            report.append("   ✅ Excellent ROI - strategy appears very profitable")
        elif result.roi > 5:
            report.append("   ✅ Good ROI - strategy shows consistent profit")
        elif result.roi > 0:
            report.append("   ⚠️ Marginal profit - consider optimizing parameters")
        else:
            report.append("   ❌ Negative ROI - strategy needs significant revision")
        
        if result.max_drawdown > 30:
            report.append("   ⚠️ High drawdown risk - consider reducing stake sizes")
        
        if result.total_bets < 100:
            report.append("   ⚠️ Sample size too small for reliable conclusions")
        elif result.total_bets < 500:
            report.append("   ⚠️ Sample size adequate but could be larger")
        else:
            report.append("   ✅ Good sample size for statistical significance")
        
        report.append("")
        report.append("=" * 70)
        
        return "\n".join(report)
    
    def save_results(self, result: BacktestResult, filepath: str = None):
        """Save backtest results to file."""
        if filepath is None:
            os.makedirs(PATHS["backtest_results"], exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(PATHS["backtest_results"], f"backtest_{timestamp}.json")
        
        data = {
            "config": {
                "starting_bankroll": self.config.starting_bankroll,
                "kelly_fraction": self.config.kelly_fraction,
                "max_bet_percent": self.config.max_bet_percent,
                "min_edge": self.config.min_edge,
                "markets": self.config.markets,
            },
            "results": {
                "start_date": result.start_date,
                "end_date": result.end_date,
                "total_matches": result.total_matches,
                "total_bets": result.total_bets,
                "bets_won": result.bets_won,
                "bets_lost": result.bets_lost,
                "profit_loss": result.profit_loss,
                "roi": result.roi,
                "win_rate": result.win_rate,
                "max_drawdown": result.max_drawdown,
                "clv_positive_rate": result.clv_positive_rate,
                "avg_clv": result.avg_clv,
            },
            "bankroll_history": result.bankroll_history,
            "bets_log": result.bets_log,
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Results saved to {filepath}")


class ParameterOptimizer:
    """
    Optimize backtest parameters to find best configuration.
    """
    
    def __init__(self):
        self.results = []
    
    def grid_search(self, league_code: str, seasons: List[str],
                    param_grid: Dict[str, List]) -> List[Dict]:
        """
        Run grid search over parameter combinations.
        
        Args:
            league_code: League to test
            seasons: Seasons to test
            param_grid: Dictionary of parameters to test
                e.g., {
                    "kelly_fraction": [0.25, 0.5, 0.75],
                    "min_edge": [2, 3, 5],
                }
        """
        from itertools import product
        
        # Generate all combinations
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(product(*values))
        
        results = []
        
        for i, combo in enumerate(combinations):
            params = dict(zip(keys, combo))
            print(f"\nTesting combination {i+1}/{len(combinations)}: {params}")
            
            config = BacktestConfig(**params)
            backtester = Backtester(config)
            
            try:
                result = backtester.run_backtest(league_code, seasons)
                
                results.append({
                    "params": params,
                    "roi": result.roi,
                    "win_rate": result.win_rate,
                    "max_drawdown": result.max_drawdown,
                    "clv_positive_rate": result.clv_positive_rate,
                    "total_bets": result.total_bets,
                    "profit_loss": result.profit_loss,
                })
                
                print(f"   ROI: {result.roi:.2f}% | CLV+: {result.clv_positive_rate:.1f}%")
                
            except Exception as e:
                print(f"   Error: {e}")
        
        # Sort by ROI
        results.sort(key=lambda x: x["roi"], reverse=True)
        self.results = results
        
        return results
    
    def get_best_params(self) -> Optional[Dict]:
        """Get best performing parameter set."""
        if not self.results:
            return None
        return self.results[0]["params"]
    
    def generate_report(self) -> str:
        """Generate optimization report."""
        if not self.results:
            return "No optimization results available."
        
        report = []
        report.append("=" * 70)
        report.append("PARAMETER OPTIMIZATION RESULTS")
        report.append("=" * 70)
        report.append("")
        
        report.append("TOP 5 CONFIGURATIONS:")
        report.append("-" * 70)
        
        for i, r in enumerate(self.results[:5]):
            report.append(f"\n{i+1}. ROI: {r['roi']:.2f}%")
            report.append(f"   Parameters: {r['params']}")
            report.append(f"   Win Rate: {r['win_rate']:.1f}%")
            report.append(f"   CLV+: {r['clv_positive_rate']:.1f}%")
            report.append(f"   Max Drawdown: {r['max_drawdown']:.1f}%")
            report.append(f"   Total Bets: {r['total_bets']}")
        
        report.append("")
        report.append("=" * 70)
        
        return "\n".join(report)


# =============================================================================
# USAGE EXAMPLE
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("BACKTESTING FRAMEWORK - DEMO")
    print("=" * 60)
    
    # Configure backtest
    config = BacktestConfig(
        starting_bankroll=1000,
        kelly_fraction=0.5,
        max_bet_percent=5.0,
        min_edge=3.0,
        markets=["1X2", "Over/Under"],
    )
    
    # Run backtest on Premier League
    backtester = Backtester(config)
    
    print("\nLoading Premier League data for 2023/24 season...")
    
    try:
        result = backtester.run_backtest(
            league_code="E0",  # Premier League
            seasons=["2324"],  # 2023/24 season
            progress_callback=lambda p: print(f"Progress: {p*100:.0f}%", end="\r")
        )
        
        # Print report
        print("\n" + backtester.generate_report(result))
        
        # Save results
        backtester.save_results(result)
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: Make sure you have internet connection to download data.")
