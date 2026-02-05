"""
BETTING MODEL
=============
Core prediction model using Poisson distribution, ELO ratings, and multiple factors.
"""

import math
from scipy.stats import poisson
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np

from config import (
    POISSON_SETTINGS, ELO_SETTINGS, BANKROLL, VALUE_THRESHOLDS,
    MOTIVATION_FACTORS, CONGESTION_SETTINGS, WEATHER_IMPACT,
    REFEREE_SETTINGS, KEY_NUMBERS
)


@dataclass
class TeamStats:
    """Container for team statistics."""
    name: str
    xg_for: float = 0.0
    xg_against: float = 0.0
    goals_scored: float = 0.0
    goals_conceded: float = 0.0
    matches_played: int = 0
    elo_rating: float = 1500.0
    form_xg: float = 0.0  # xG in last 5 games
    form_xga: float = 0.0  # xGA in last 5 games
    home_xg: float = 0.0
    away_xg: float = 0.0


@dataclass
class MatchContext:
    """Additional context factors for a match."""
    referee_cards_per_game: float = 3.5
    home_motivation: str = "normal"  # relegation_battle, title_race, etc.
    away_motivation: str = "normal"
    home_days_rest: int = 7
    away_days_rest: int = 7
    home_matches_30_days: int = 4
    away_matches_30_days: int = 4
    is_derby: bool = False
    weather_condition: str = "normal"  # rain_heavy, wind_strong, etc.
    home_key_players_out: int = 0
    away_key_players_out: int = 0


@dataclass
class LiveMatchState:
    """Live match state for in-play analysis."""
    minute: int
    home_goals: int
    away_goals: int


@dataclass
class Prediction:
    """Container for match prediction results."""
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    expected_home_goals: float
    expected_away_goals: float
    over_2_5_prob: float
    under_2_5_prob: float
    btts_yes_prob: float
    btts_no_prob: float
    score_matrix: Dict[Tuple[int, int], float]
    most_likely_scores: List[Tuple[Tuple[int, int], float]]
    confidence: str


class PoissonModel:
    """
    Poisson distribution model for goal prediction.
    """
    
    def __init__(self, settings: Dict = None):
        self.settings = settings or POISSON_SETTINGS
        self.max_goals = self.settings["max_goals"]
        self.home_advantage = self.settings["home_advantage"]
        self.xg_weight = self.settings["xg_weight"]
        self.form_weight = self.settings["form_weight"]
        self.season_weight = self.settings["season_weight"]
    
    def calculate_expected_goals(self, 
                                  home_team: TeamStats, 
                                  away_team: TeamStats,
                                  league_avg_goals: float = 2.7) -> Tuple[float, float]:
        """
        Calculate expected goals for both teams.
        
        Uses weighted combination of:
        - xG data (if available)
        - Recent form
        - Season averages
        - ELO difference
        """
        # Calculate attack and defense strengths
        home_attack = self._calculate_attack_strength(home_team, league_avg_goals, is_home=True)
        home_defense = self._calculate_defense_strength(home_team, league_avg_goals)
        away_attack = self._calculate_attack_strength(away_team, league_avg_goals, is_home=False)
        away_defense = self._calculate_defense_strength(away_team, league_avg_goals)
        
        # Base expected goals
        home_xg = home_attack * away_defense * (league_avg_goals / 2) * self.home_advantage
        away_xg = away_attack * home_defense * (league_avg_goals / 2)
        
        # Adjust for ELO difference
        elo_diff = home_team.elo_rating - away_team.elo_rating
        elo_factor = 1 + (elo_diff / 1000)  # Small adjustment based on ELO
        
        home_xg *= elo_factor
        away_xg *= (2 - elo_factor)  # Inverse for away team
        
        # Ensure reasonable bounds
        home_xg = max(0.3, min(4.0, home_xg))
        away_xg = max(0.3, min(4.0, away_xg))
        
        return home_xg, away_xg
    
    def _calculate_attack_strength(self, team: TeamStats, league_avg: float, 
                                    is_home: bool) -> float:
        """Calculate team's attacking strength."""
        if team.matches_played == 0:
            return 1.0
        
        # Use xG if available, otherwise goals
        if team.xg_for > 0:
            xg_per_game = team.xg_for / team.matches_played
            goals_per_game = team.goals_scored / team.matches_played
            attack = (xg_per_game * self.xg_weight + 
                     goals_per_game * (1 - self.xg_weight))
        else:
            attack = team.goals_scored / team.matches_played
        
        # Normalize to league average
        attack_strength = attack / (league_avg / 2)
        
        return max(0.5, min(2.0, attack_strength))
    
    def _calculate_defense_strength(self, team: TeamStats, league_avg: float) -> float:
        """Calculate team's defensive strength (lower is better)."""
        if team.matches_played == 0:
            return 1.0
        
        # Use xGA if available
        if team.xg_against > 0:
            xga_per_game = team.xg_against / team.matches_played
            goals_against_per_game = team.goals_conceded / team.matches_played
            defense = (xga_per_game * self.xg_weight + 
                      goals_against_per_game * (1 - self.xg_weight))
        else:
            defense = team.goals_conceded / team.matches_played
        
        # Normalize (higher value = worse defense)
        defense_strength = defense / (league_avg / 2)
        
        return max(0.5, min(2.0, defense_strength))
    
    def generate_score_matrix(self, home_xg: float, away_xg: float) -> Dict[Tuple[int, int], float]:
        """
        Generate probability matrix for all possible scores.
        """
        matrix = {}
        
        for home_goals in range(self.max_goals + 1):
            for away_goals in range(self.max_goals + 1):
                home_prob = poisson.pmf(home_goals, home_xg)
                away_prob = poisson.pmf(away_goals, away_xg)
                matrix[(home_goals, away_goals)] = home_prob * away_prob
        
        return matrix
    
    def calculate_probabilities(self, score_matrix: Dict[Tuple[int, int], float]) -> Dict:
        """
        Calculate all market probabilities from score matrix.
        """
        home_win = 0.0
        draw = 0.0
        away_win = 0.0
        over_2_5 = 0.0
        btts_yes = 0.0
        
        for (home, away), prob in score_matrix.items():
            if home > away:
                home_win += prob
            elif home == away:
                draw += prob
            else:
                away_win += prob
            
            if home + away > 2.5:
                over_2_5 += prob
            
            if home > 0 and away > 0:
                btts_yes += prob
        
        return {
            "home_win": home_win,
            "draw": draw,
            "away_win": away_win,
            "over_2_5": over_2_5,
            "under_2_5": 1 - over_2_5,
            "btts_yes": btts_yes,
            "btts_no": 1 - btts_yes,
        }


class ContextAdjuster:
    """
    Adjusts predictions based on contextual factors.
    """
    
    def apply_motivation_factor(self, base_xg: float, motivation: str, 
                                 is_attacking: bool = True) -> float:
        """Apply motivation multiplier to expected goals."""
        factor = MOTIVATION_FACTORS.get(motivation, 1.0)
        
        if is_attacking:
            return base_xg * factor
        else:
            # Better motivation = better defense = lower xG against
            return base_xg / factor
    
    def apply_fatigue_factor(self, base_xg: float, days_rest: int, 
                              matches_30_days: int) -> float:
        """Adjust xG based on fatigue."""
        # Less rest = less effective
        if days_rest <= 3:
            factor = 0.95
        elif days_rest <= 4:
            factor = 0.97
        else:
            factor = 1.0
        
        # Many matches = fatigue
        if matches_30_days >= 8:
            factor *= 0.95
        elif matches_30_days >= 6:
            factor *= 0.97
        
        return base_xg * factor
    
    def apply_weather_factor(self, base_xg: float, weather: str) -> float:
        """Adjust xG based on weather conditions."""
        if weather in WEATHER_IMPACT:
            return base_xg * WEATHER_IMPACT[weather].get("goals_modifier", 1.0)
        return base_xg
    
    def apply_key_player_factor(self, base_xg: float, players_out: int) -> float:
        """Adjust xG based on missing key players."""
        # Each key player out reduces effectiveness by ~5%
        factor = 1.0 - (players_out * 0.05)
        return base_xg * max(0.7, factor)
    
    def adjust_prediction(self, home_xg: float, away_xg: float, 
                          context: MatchContext) -> Tuple[float, float]:
        """Apply all context adjustments to predictions."""
        
        # Home team adjustments
        home_xg = self.apply_motivation_factor(home_xg, context.home_motivation)
        home_xg = self.apply_fatigue_factor(home_xg, context.home_days_rest, 
                                            context.home_matches_30_days)
        home_xg = self.apply_key_player_factor(home_xg, context.home_key_players_out)
        
        # Away team adjustments
        away_xg = self.apply_motivation_factor(away_xg, context.away_motivation)
        away_xg = self.apply_fatigue_factor(away_xg, context.away_days_rest,
                                            context.away_matches_30_days)
        away_xg = self.apply_key_player_factor(away_xg, context.away_key_players_out)
        
        # Weather affects both teams
        home_xg = self.apply_weather_factor(home_xg, context.weather_condition)
        away_xg = self.apply_weather_factor(away_xg, context.weather_condition)
        
        # Derby boost (more goals typically)
        if context.is_derby:
            home_xg *= 1.05
            away_xg *= 1.05
        
        return home_xg, away_xg


class KellyCalculator:
    """
    Kelly Criterion calculator for optimal bet sizing.
    """
    
    def __init__(self, fraction: float = 0.5, max_bet_percent: float = 5.0):
        """
        Initialize Kelly calculator.
        
        Args:
            fraction: Kelly fraction (0.5 = half Kelly, recommended)
            max_bet_percent: Maximum bet as percentage of bankroll
        """
        self.fraction = fraction
        self.max_bet_percent = max_bet_percent
    
    def calculate_kelly(self, prob: float, odds: float) -> float:
        """
        Calculate Kelly stake percentage.
        
        Formula: f* = (bp - q) / b
        Where:
            f* = fraction of bankroll to bet
            b = decimal odds - 1 (net odds)
            p = probability of winning
            q = probability of losing (1 - p)
        """
        if odds <= 1.0 or prob <= 0 or prob >= 1:
            return 0.0
        
        b = odds - 1  # Net odds
        p = prob
        q = 1 - p
        
        kelly = (b * p - q) / b
        
        # Apply fractional Kelly
        kelly *= self.fraction
        
        # Apply maximum cap
        kelly = min(kelly, self.max_bet_percent / 100)
        
        # No negative bets
        return max(0, kelly)
    
    def calculate_stake(self, bankroll: float, prob: float, odds: float) -> float:
        """Calculate actual stake amount."""
        kelly_fraction = self.calculate_kelly(prob, odds)
        return bankroll * kelly_fraction
    
    def calculate_ev(self, prob: float, odds: float) -> float:
        """
        Calculate Expected Value (EV) of a bet.
        
        EV = (prob * (odds - 1)) - (1 - prob)
        """
        if odds <= 1.0:
            return -1.0
        
        return (prob * (odds - 1)) - (1 - prob)
    
    def calculate_edge(self, prob: float, odds: float) -> float:
        """
        Calculate edge percentage.
        
        Edge = (implied_prob - our_prob) / implied_prob * 100
        """
        implied_prob = 1 / odds
        edge = (prob - implied_prob) / implied_prob * 100
        return edge


class ValueBetFinder:
    """
    Identifies value bets by comparing model probabilities to market odds.
    """
    
    def __init__(self):
        self.kelly = KellyCalculator(
            fraction=BANKROLL["kelly_fraction"],
            max_bet_percent=BANKROLL["max_bet_percent"]
        )
        self.min_edge = VALUE_THRESHOLDS["minimum_ev"]
    
    def find_value(self, probabilities: Dict, odds: Dict) -> List[Dict]:
        """
        Find value bets in a match.
        
        Args:
            probabilities: Dict with our calculated probabilities
            odds: Dict with bookmaker odds for each market
        
        Returns:
            List of value bet opportunities
        """
        value_bets = []
        
        markets = [
            ("home_win", "1X2", "Home"),
            ("draw", "1X2", "Draw"),
            ("away_win", "1X2", "Away"),
            ("over_2_5", "Totals", "Over 2.5"),
            ("under_2_5", "Totals", "Under 2.5"),
            ("btts_yes", "BTTS", "Yes"),
            ("btts_no", "BTTS", "No"),
        ]
        
        for prob_key, market_type, selection in markets:
            if prob_key in probabilities and prob_key in odds:
                prob = probabilities[prob_key]
                market_odds = odds[prob_key]
                
                edge = self.kelly.calculate_edge(prob, market_odds)
                ev = self.kelly.calculate_ev(prob, market_odds)
                
                if edge >= self.min_edge:
                    kelly_stake = self.kelly.calculate_kelly(prob, market_odds)
                    
                    value_bets.append({
                        "market": market_type,
                        "selection": selection,
                        "our_probability": round(prob * 100, 2),
                        "implied_probability": round((1 / market_odds) * 100, 2),
                        "odds": market_odds,
                        "edge_percent": round(edge, 2),
                        "expected_value": round(ev * 100, 2),
                        "kelly_stake_percent": round(kelly_stake * 100, 2),
                        "confidence": self._get_confidence(edge),
                    })
        
        # Sort by edge (highest first)
        value_bets.sort(key=lambda x: x["edge_percent"], reverse=True)
        
        return value_bets
    
    def _get_confidence(self, edge: float) -> str:
        """Determine confidence level based on edge."""
        if edge >= VALUE_THRESHOLDS["confidence_levels"]["high"]:
            return "HIGH"
        elif edge >= VALUE_THRESHOLDS["confidence_levels"]["medium"]:
            return "MEDIUM"
        else:
            return "LOW"


class BettingModel:
    """
    Main betting model that combines all components.
    """
    
    def __init__(self):
        self.poisson = PoissonModel()
        self.adjuster = ContextAdjuster()
        self.value_finder = ValueBetFinder()
        self.kelly = KellyCalculator(
            fraction=BANKROLL["kelly_fraction"],
            max_bet_percent=BANKROLL["max_bet_percent"]
        )
    
    def predict_match(self, 
                      home_team: TeamStats, 
                      away_team: TeamStats,
                      context: MatchContext = None,
                      league_avg_goals: float = 2.7) -> Prediction:
        """
        Generate full prediction for a match.
        """
        # Calculate base expected goals
        home_xg, away_xg = self.poisson.calculate_expected_goals(
            home_team, away_team, league_avg_goals
        )
        
        # Apply context adjustments
        if context:
            home_xg, away_xg = self.adjuster.adjust_prediction(
                home_xg, away_xg, context
            )
        
        # Generate score matrix
        score_matrix = self.poisson.generate_score_matrix(home_xg, away_xg)
        
        # Calculate probabilities
        probs = self.poisson.calculate_probabilities(score_matrix)
        
        # Get most likely scores
        sorted_scores = sorted(score_matrix.items(), key=lambda x: x[1], reverse=True)
        most_likely = sorted_scores[:5]
        
        # Determine confidence
        confidence = self._calculate_confidence(home_team, away_team, context)
        
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
            score_matrix=score_matrix,
            most_likely_scores=most_likely,
            confidence=confidence,
        )

    def predict_live_match(
        self,
        home_team: TeamStats,
        away_team: TeamStats,
        live_state: LiveMatchState,
        context: MatchContext = None,
        league_avg_goals: float = 2.7,
    ) -> Prediction:
        """
        Generate in-play prediction using remaining-time Poisson adjustments.
        """
        home_xg, away_xg = self.poisson.calculate_expected_goals(
            home_team, away_team, league_avg_goals
        )

        if context:
            home_xg, away_xg = self.adjuster.adjust_prediction(
                home_xg, away_xg, context
            )

        minute = max(0, min(90, live_state.minute))
        remaining_ratio = max(0.0, (90 - minute) / 90)
        remaining_home_xg = home_xg * remaining_ratio
        remaining_away_xg = away_xg * remaining_ratio

        score_matrix = self.poisson.generate_score_matrix(
            remaining_home_xg, remaining_away_xg
        )
        probs = self._calculate_live_probabilities(
            score_matrix, live_state.home_goals, live_state.away_goals
        )

        most_likely = self._calculate_live_scorelines(
            score_matrix, live_state.home_goals, live_state.away_goals
        )

        confidence = self._calculate_live_confidence(
            home_team, away_team, context, minute
        )

        return Prediction(
            home_win_prob=probs["home_win"],
            draw_prob=probs["draw"],
            away_win_prob=probs["away_win"],
            expected_home_goals=live_state.home_goals + remaining_home_xg,
            expected_away_goals=live_state.away_goals + remaining_away_xg,
            over_2_5_prob=probs["over_2_5"],
            under_2_5_prob=probs["under_2_5"],
            btts_yes_prob=probs["btts_yes"],
            btts_no_prob=probs["btts_no"],
            score_matrix=score_matrix,
            most_likely_scores=most_likely,
            confidence=confidence,
        )
    
    def find_value_bets(self, prediction: Prediction, odds: Dict) -> List[Dict]:
        """Find value bets for a match."""
        probabilities = {
            "home_win": prediction.home_win_prob,
            "draw": prediction.draw_prob,
            "away_win": prediction.away_win_prob,
            "over_2_5": prediction.over_2_5_prob,
            "under_2_5": prediction.under_2_5_prob,
            "btts_yes": prediction.btts_yes_prob,
            "btts_no": prediction.btts_no_prob,
        }
        
        return self.value_finder.find_value(probabilities, odds)
    
    def _calculate_confidence(self, home_team: TeamStats, away_team: TeamStats,
                              context: MatchContext) -> str:
        """
        Calculate overall prediction confidence.
        """
        confidence_score = 100
        
        # Less data = less confidence
        min_matches = min(home_team.matches_played, away_team.matches_played)
        if min_matches < 5:
            confidence_score -= 30
        elif min_matches < 10:
            confidence_score -= 15
        
        # Missing key players reduces confidence
        if context:
            if context.home_key_players_out > 2 or context.away_key_players_out > 2:
                confidence_score -= 20
            
            # Weather uncertainty
            if context.weather_condition in ["rain_heavy", "wind_strong"]:
                confidence_score -= 10
        
        if confidence_score >= 80:
            return "HIGH"
        elif confidence_score >= 60:
            return "MEDIUM"
        else:
            return "LOW"

    def _calculate_live_probabilities(
        self,
        score_matrix: Dict[Tuple[int, int], float],
        home_goals: int,
        away_goals: int,
    ) -> Dict[str, float]:
        """Calculate live probabilities from remaining-goals matrix."""
        home_win = draw = away_win = 0.0
        over_2_5 = under_2_5 = 0.0
        btts_yes = btts_no = 0.0

        for (home_add, away_add), prob in score_matrix.items():
            final_home = home_goals + home_add
            final_away = away_goals + away_add

            if final_home > final_away:
                home_win += prob
            elif final_home == final_away:
                draw += prob
            else:
                away_win += prob

            total_goals = final_home + final_away
            if total_goals > 2.5:
                over_2_5 += prob
            else:
                under_2_5 += prob

            if final_home > 0 and final_away > 0:
                btts_yes += prob
            else:
                btts_no += prob

        return {
            "home_win": home_win,
            "draw": draw,
            "away_win": away_win,
            "over_2_5": over_2_5,
            "under_2_5": under_2_5,
            "btts_yes": btts_yes,
            "btts_no": btts_no,
        }

    def _calculate_live_scorelines(
        self,
        score_matrix: Dict[Tuple[int, int], float],
        home_goals: int,
        away_goals: int,
    ) -> List[Tuple[Tuple[int, int], float]]:
        """Convert remaining goals matrix to full-time scorelines."""
        full_time = {}
        for (home_add, away_add), prob in score_matrix.items():
            final_score = (home_goals + home_add, away_goals + away_add)
            full_time[final_score] = full_time.get(final_score, 0) + prob

        return sorted(full_time.items(), key=lambda x: x[1], reverse=True)[:5]

    def _calculate_live_confidence(
        self,
        home_team: TeamStats,
        away_team: TeamStats,
        context: MatchContext,
        minute: int,
    ) -> str:
        """Adjust confidence based on live match minute."""
        base = self._calculate_confidence(home_team, away_team, context)
        if minute < 15 and base == "HIGH":
            return "MEDIUM"
        if minute >= 60 and base == "MEDIUM":
            return "HIGH"
        return base
    
    def generate_report(self, home_team: TeamStats, away_team: TeamStats,
                        prediction: Prediction, odds: Dict = None) -> str:
        """Generate a text report for a match prediction."""
        
        report = []
        report.append("=" * 60)
        report.append(f"MATCH PREDICTION: {home_team.name} vs {away_team.name}")
        report.append("=" * 60)
        report.append("")
        
        # Expected Goals
        report.append("📊 EXPECTED GOALS")
        report.append(f"   {home_team.name}: {prediction.expected_home_goals:.2f}")
        report.append(f"   {away_team.name}: {prediction.expected_away_goals:.2f}")
        report.append("")
        
        # Win Probabilities
        report.append("📈 WIN PROBABILITIES")
        report.append(f"   Home Win:  {prediction.home_win_prob * 100:.1f}%")
        report.append(f"   Draw:      {prediction.draw_prob * 100:.1f}%")
        report.append(f"   Away Win:  {prediction.away_win_prob * 100:.1f}%")
        report.append("")
        
        # Other Markets
        report.append("⚽ OTHER MARKETS")
        report.append(f"   Over 2.5:  {prediction.over_2_5_prob * 100:.1f}%")
        report.append(f"   Under 2.5: {prediction.under_2_5_prob * 100:.1f}%")
        report.append(f"   BTTS Yes:  {prediction.btts_yes_prob * 100:.1f}%")
        report.append(f"   BTTS No:   {prediction.btts_no_prob * 100:.1f}%")
        report.append("")
        
        # Most Likely Scores
        report.append("🎯 MOST LIKELY SCORES")
        for (home, away), prob in prediction.most_likely_scores[:5]:
            report.append(f"   {home}-{away}: {prob * 100:.1f}%")
        report.append("")
        
        # Value Bets
        if odds:
            value_bets = self.find_value_bets(prediction, odds)
            if value_bets:
                report.append("💰 VALUE BETS FOUND")
                for bet in value_bets:
                    report.append(f"   {bet['market']} - {bet['selection']}")
                    report.append(f"      Odds: {bet['odds']} | Edge: {bet['edge_percent']}%")
                    report.append(f"      Kelly: {bet['kelly_stake_percent']}% | Confidence: {bet['confidence']}")
            else:
                report.append("❌ NO VALUE BETS FOUND")
        
        report.append("")
        report.append(f"Prediction Confidence: {prediction.confidence}")
        report.append("=" * 60)
        
        return "\n".join(report)


# =============================================================================
# USAGE EXAMPLE
# =============================================================================
if __name__ == "__main__":
    # Create sample teams
    arsenal = TeamStats(
        name="Arsenal",
        xg_for=45.5,
        xg_against=22.3,
        goals_scored=48,
        goals_conceded=20,
        matches_played=20,
        elo_rating=1850,
    )
    
    chelsea = TeamStats(
        name="Chelsea",
        xg_for=35.2,
        xg_against=30.5,
        goals_scored=38,
        goals_conceded=32,
        matches_played=20,
        elo_rating=1720,
    )
    
    # Create context
    context = MatchContext(
        referee_cards_per_game=4.2,
        home_motivation="title_race",
        away_motivation="europa_race",
        home_days_rest=7,
        away_days_rest=4,
        is_derby=True,
    )
    
    # Sample odds
    odds = {
        "home_win": 1.65,
        "draw": 3.80,
        "away_win": 5.50,
        "over_2_5": 1.75,
        "under_2_5": 2.10,
        "btts_yes": 1.80,
        "btts_no": 2.00,
    }
    
    # Generate prediction
    model = BettingModel()
    prediction = model.predict_match(arsenal, chelsea, context)
    
    # Print report
    print(model.generate_report(arsenal, chelsea, prediction, odds))
