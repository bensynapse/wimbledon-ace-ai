"""
CLV TRACKER & BET MANAGEMENT
=============================
Track bets, closing line value, and performance metrics.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import statistics

from config import PATHS, CLV_SETTINGS, BANKROLL


@dataclass
class Bet:
    """Single bet record."""
    id: str
    timestamp: str
    match: str
    league: str
    market: str
    selection: str
    odds_placed: float
    odds_closing: Optional[float]  # Filled in after match
    stake: float
    stake_percent: float
    our_probability: float
    implied_probability: float
    edge_at_placement: float
    bookmaker: str
    result: Optional[str]  # "won", "lost", "void", "pending"
    profit_loss: Optional[float]
    clv: Optional[float]  # Closing Line Value
    notes: str = ""


@dataclass
class BetStats:
    """Aggregate betting statistics."""
    total_bets: int
    won: int
    lost: int
    void: int
    pending: int
    total_staked: float
    total_returns: float
    profit_loss: float
    roi: float
    yield_percent: float
    win_rate: float
    avg_odds: float
    clv_positive_percent: float
    avg_clv: float
    longest_winning_streak: int
    longest_losing_streak: int


class BetTracker:
    """
    Tracks all bets and calculates performance metrics.
    """
    
    def __init__(self, filepath: str = None):
        self.filepath = filepath or PATHS["bet_tracker"]
        self.bets: List[Bet] = []
        self._load()
    
    def _load(self):
        """Load bets from file."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                    self.bets = [Bet(**bet) for bet in data]
            except Exception as e:
                print(f"Error loading bets: {e}")
                self.bets = []
    
    def _save(self):
        """Save bets to file."""
        os.makedirs(os.path.dirname(self.filepath) or '.', exist_ok=True)
        with open(self.filepath, 'w') as f:
            json.dump([asdict(bet) for bet in self.bets], f, indent=2)
    
    def add_bet(self, 
                match: str,
                league: str,
                market: str,
                selection: str,
                odds: float,
                stake: float,
                bankroll: float,
                our_probability: float,
                bookmaker: str = "unknown",
                notes: str = "") -> Bet:
        """
        Record a new bet.
        """
        implied_prob = 1 / odds
        edge = ((our_probability - implied_prob) / implied_prob) * 100
        
        bet = Bet(
            id=f"BET_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.bets)}",
            timestamp=datetime.now().isoformat(),
            match=match,
            league=league,
            market=market,
            selection=selection,
            odds_placed=odds,
            odds_closing=None,
            stake=stake,
            stake_percent=(stake / bankroll) * 100,
            our_probability=our_probability,
            implied_probability=implied_prob,
            edge_at_placement=edge,
            bookmaker=bookmaker,
            result="pending",
            profit_loss=None,
            clv=None,
            notes=notes,
        )
        
        self.bets.append(bet)
        self._save()
        
        return bet
    
    def update_result(self, bet_id: str, result: str, 
                      closing_odds: float = None) -> Optional[Bet]:
        """
        Update bet result and calculate CLV.
        
        Args:
            bet_id: Bet identifier
            result: "won", "lost", or "void"
            closing_odds: Final odds before match started
        """
        for bet in self.bets:
            if bet.id == bet_id:
                bet.result = result
                
                # Calculate profit/loss
                if result == "won":
                    bet.profit_loss = bet.stake * (bet.odds_placed - 1)
                elif result == "lost":
                    bet.profit_loss = -bet.stake
                else:  # void
                    bet.profit_loss = 0
                
                # Calculate CLV if closing odds provided
                if closing_odds:
                    bet.odds_closing = closing_odds
                    bet.clv = self._calculate_clv(bet.odds_placed, closing_odds)
                
                self._save()
                return bet
        
        return None
    
    def _calculate_clv(self, odds_placed: float, odds_closing: float) -> float:
        """
        Calculate Closing Line Value.
        
        CLV = (closing_implied_prob - placed_implied_prob) / placed_implied_prob * 100
        
        Positive CLV = you beat the closing line (good)
        Negative CLV = closing line was better (bad)
        """
        placed_implied = 1 / odds_placed
        closing_implied = 1 / odds_closing
        
        # If you got higher odds than closing, that's positive CLV
        # Because you're getting paid more for the same probability
        clv = ((odds_placed - odds_closing) / odds_closing) * 100
        
        return clv
    
    def get_bet(self, bet_id: str) -> Optional[Bet]:
        """Get a specific bet."""
        for bet in self.bets:
            if bet.id == bet_id:
                return bet
        return None
    
    def get_pending_bets(self) -> List[Bet]:
        """Get all pending bets."""
        return [bet for bet in self.bets if bet.result == "pending"]
    
    def get_bets_by_league(self, league: str) -> List[Bet]:
        """Get bets filtered by league."""
        return [bet for bet in self.bets if bet.league == league]
    
    def get_bets_by_market(self, market: str) -> List[Bet]:
        """Get bets filtered by market type."""
        return [bet for bet in self.bets if bet.market == market]
    
    def get_bets_in_range(self, start_date: datetime, end_date: datetime) -> List[Bet]:
        """Get bets within a date range."""
        return [
            bet for bet in self.bets
            if start_date <= datetime.fromisoformat(bet.timestamp) <= end_date
        ]
    
    def calculate_stats(self, bets: List[Bet] = None) -> BetStats:
        """
        Calculate comprehensive statistics.
        """
        if bets is None:
            bets = self.bets
        
        if not bets:
            return BetStats(
                total_bets=0, won=0, lost=0, void=0, pending=0,
                total_staked=0, total_returns=0, profit_loss=0,
                roi=0, yield_percent=0, win_rate=0, avg_odds=0,
                clv_positive_percent=0, avg_clv=0,
                longest_winning_streak=0, longest_losing_streak=0
            )
        
        settled = [b for b in bets if b.result in ["won", "lost"]]
        
        won = len([b for b in bets if b.result == "won"])
        lost = len([b for b in bets if b.result == "lost"])
        void = len([b for b in bets if b.result == "void"])
        pending = len([b for b in bets if b.result == "pending"])
        
        total_staked = sum(b.stake for b in settled)
        total_returns = sum(
            b.stake * b.odds_placed if b.result == "won" else 0 
            for b in settled
        )
        profit_loss = sum(b.profit_loss or 0 for b in settled)
        
        roi = (profit_loss / total_staked * 100) if total_staked > 0 else 0
        yield_percent = (profit_loss / len(settled)) if settled else 0
        win_rate = (won / len(settled) * 100) if settled else 0
        avg_odds = statistics.mean([b.odds_placed for b in settled]) if settled else 0
        
        # CLV stats
        bets_with_clv = [b for b in settled if b.clv is not None]
        clv_positive = len([b for b in bets_with_clv if b.clv > 0])
        clv_positive_percent = (clv_positive / len(bets_with_clv) * 100) if bets_with_clv else 0
        avg_clv = statistics.mean([b.clv for b in bets_with_clv]) if bets_with_clv else 0
        
        # Streaks
        winning_streak, losing_streak = self._calculate_streaks(settled)
        
        return BetStats(
            total_bets=len(bets),
            won=won,
            lost=lost,
            void=void,
            pending=pending,
            total_staked=total_staked,
            total_returns=total_returns,
            profit_loss=profit_loss,
            roi=roi,
            yield_percent=yield_percent,
            win_rate=win_rate,
            avg_odds=avg_odds,
            clv_positive_percent=clv_positive_percent,
            avg_clv=avg_clv,
            longest_winning_streak=winning_streak,
            longest_losing_streak=losing_streak,
        )
    
    def _calculate_streaks(self, settled_bets: List[Bet]) -> Tuple[int, int]:
        """Calculate longest winning and losing streaks."""
        if not settled_bets:
            return 0, 0
        
        # Sort by timestamp
        sorted_bets = sorted(settled_bets, key=lambda b: b.timestamp)
        
        max_win = current_win = 0
        max_loss = current_loss = 0
        
        for bet in sorted_bets:
            if bet.result == "won":
                current_win += 1
                current_loss = 0
                max_win = max(max_win, current_win)
            else:
                current_loss += 1
                current_win = 0
                max_loss = max(max_loss, current_loss)
        
        return max_win, max_loss
    
    def generate_report(self, bets: List[Bet] = None) -> str:
        """Generate a comprehensive performance report."""
        stats = self.calculate_stats(bets)
        
        report = []
        report.append("=" * 60)
        report.append("BETTING PERFORMANCE REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append("=" * 60)
        report.append("")
        
        # Overall Performance
        report.append("📊 OVERALL PERFORMANCE")
        report.append(f"   Total Bets: {stats.total_bets}")
        report.append(f"   Won: {stats.won} | Lost: {stats.lost} | Void: {stats.void} | Pending: {stats.pending}")
        report.append(f"   Win Rate: {stats.win_rate:.1f}%")
        report.append("")
        
        # Financial
        report.append("💰 FINANCIAL")
        report.append(f"   Total Staked: €{stats.total_staked:.2f}")
        report.append(f"   Total Returns: €{stats.total_returns:.2f}")
        report.append(f"   Profit/Loss: €{stats.profit_loss:.2f}")
        report.append(f"   ROI: {stats.roi:.2f}%")
        report.append(f"   Average Odds: {stats.avg_odds:.2f}")
        report.append("")
        
        # CLV Analysis (THE KEY METRIC)
        report.append("📈 CLOSING LINE VALUE (CLV)")
        report.append(f"   CLV Positive Rate: {stats.clv_positive_percent:.1f}%")
        report.append(f"   Average CLV: {stats.avg_clv:.2f}%")
        if stats.clv_positive_percent >= 50:
            report.append("   ✅ Good! You're beating the closing line")
        else:
            report.append("   ⚠️ Warning: You're not consistently beating the closing line")
        report.append("")
        
        # Streaks
        report.append("🔥 STREAKS")
        report.append(f"   Longest Winning Streak: {stats.longest_winning_streak}")
        report.append(f"   Longest Losing Streak: {stats.longest_losing_streak}")
        report.append("")
        
        # Assessment
        report.append("📋 ASSESSMENT")
        if stats.roi > 5:
            report.append("   ✅ Excellent performance! Keep it up.")
        elif stats.roi > 0:
            report.append("   ✅ Profitable, but room for improvement.")
        elif stats.roi > -5:
            report.append("   ⚠️ Slightly losing. Review your edge calculations.")
        else:
            report.append("   ❌ Significant losses. Consider adjusting your strategy.")
        
        # CLV is the real indicator
        if stats.clv_positive_percent >= 55:
            report.append("   ✅ CLV indicates long-term profitability expected.")
        elif stats.clv_positive_percent >= 50:
            report.append("   ⚠️ CLV is borderline. Need larger sample size.")
        else:
            report.append("   ❌ CLV indicates potential issues with bet selection.")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def export_csv(self, filepath: str):
        """Export bets to CSV."""
        import csv
        
        with open(filepath, 'w', newline='') as f:
            if self.bets:
                writer = csv.DictWriter(f, fieldnames=asdict(self.bets[0]).keys())
                writer.writeheader()
                for bet in self.bets:
                    writer.writerow(asdict(bet))


class CLVAnalyzer:
    """
    Specialized analyzer for Closing Line Value.
    """
    
    def __init__(self, tracker: BetTracker):
        self.tracker = tracker
    
    def analyze_by_bookmaker(self) -> Dict[str, Dict]:
        """Analyze CLV performance by bookmaker."""
        results = {}
        
        bets_with_clv = [b for b in self.tracker.bets if b.clv is not None]
        
        for bet in bets_with_clv:
            if bet.bookmaker not in results:
                results[bet.bookmaker] = {
                    "bets": 0,
                    "positive_clv": 0,
                    "total_clv": 0,
                }
            
            results[bet.bookmaker]["bets"] += 1
            results[bet.bookmaker]["total_clv"] += bet.clv
            if bet.clv > 0:
                results[bet.bookmaker]["positive_clv"] += 1
        
        # Calculate averages
        for book in results:
            results[book]["avg_clv"] = results[book]["total_clv"] / results[book]["bets"]
            results[book]["positive_rate"] = results[book]["positive_clv"] / results[book]["bets"] * 100
        
        return results
    
    def analyze_by_market(self) -> Dict[str, Dict]:
        """Analyze CLV performance by market type."""
        results = {}
        
        bets_with_clv = [b for b in self.tracker.bets if b.clv is not None]
        
        for bet in bets_with_clv:
            if bet.market not in results:
                results[bet.market] = {
                    "bets": 0,
                    "positive_clv": 0,
                    "total_clv": 0,
                    "won": 0,
                    "total_profit": 0,
                }
            
            results[bet.market]["bets"] += 1
            results[bet.market]["total_clv"] += bet.clv
            if bet.clv > 0:
                results[bet.market]["positive_clv"] += 1
            if bet.result == "won":
                results[bet.market]["won"] += 1
            results[bet.market]["total_profit"] += bet.profit_loss or 0
        
        # Calculate averages
        for market in results:
            results[market]["avg_clv"] = results[market]["total_clv"] / results[market]["bets"]
            results[market]["positive_rate"] = results[market]["positive_clv"] / results[market]["bets"] * 100
            results[market]["win_rate"] = results[market]["won"] / results[market]["bets"] * 100
            results[market]["roi"] = results[market]["total_profit"] / results[market]["bets"]
        
        return results
    
    def get_clv_trend(self, window: int = 50) -> List[float]:
        """
        Get rolling average CLV trend.
        
        Args:
            window: Number of bets for rolling average
        """
        bets_with_clv = sorted(
            [b for b in self.tracker.bets if b.clv is not None],
            key=lambda b: b.timestamp
        )
        
        if len(bets_with_clv) < window:
            return [b.clv for b in bets_with_clv]
        
        rolling_avg = []
        for i in range(window - 1, len(bets_with_clv)):
            window_bets = bets_with_clv[i - window + 1:i + 1]
            avg = statistics.mean([b.clv for b in window_bets])
            rolling_avg.append(avg)
        
        return rolling_avg
    
    def generate_clv_report(self) -> str:
        """Generate detailed CLV analysis report."""
        report = []
        report.append("=" * 60)
        report.append("CLOSING LINE VALUE (CLV) ANALYSIS")
        report.append("=" * 60)
        report.append("")
        
        # Overall CLV
        bets_with_clv = [b for b in self.tracker.bets if b.clv is not None]
        if not bets_with_clv:
            report.append("No bets with CLV data yet.")
            return "\n".join(report)
        
        positive = len([b for b in bets_with_clv if b.clv > 0])
        avg_clv = statistics.mean([b.clv for b in bets_with_clv])
        
        report.append("📊 OVERALL CLV METRICS")
        report.append(f"   Bets Analyzed: {len(bets_with_clv)}")
        report.append(f"   Positive CLV Rate: {positive / len(bets_with_clv) * 100:.1f}%")
        report.append(f"   Average CLV: {avg_clv:.2f}%")
        report.append("")
        
        # By Bookmaker
        by_book = self.analyze_by_bookmaker()
        if by_book:
            report.append("📚 CLV BY BOOKMAKER")
            for book, data in sorted(by_book.items(), key=lambda x: x[1]["avg_clv"], reverse=True):
                report.append(f"   {book}: Avg CLV {data['avg_clv']:.2f}% | Positive Rate {data['positive_rate']:.1f}%")
            report.append("")
        
        # By Market
        by_market = self.analyze_by_market()
        if by_market:
            report.append("🎯 CLV BY MARKET")
            for market, data in sorted(by_market.items(), key=lambda x: x[1]["avg_clv"], reverse=True):
                report.append(f"   {market}: Avg CLV {data['avg_clv']:.2f}% | Win Rate {data['win_rate']:.1f}%")
            report.append("")
        
        # Assessment
        report.append("📋 CLV ASSESSMENT")
        target = CLV_SETTINGS["positive_clv_target"]
        current = positive / len(bets_with_clv) * 100
        
        if current >= 55:
            report.append(f"   ✅ Excellent! {current:.1f}% positive CLV (target: {target}%)")
            report.append("   You're consistently finding value before the market.")
        elif current >= 50:
            report.append(f"   ⚠️ Okay. {current:.1f}% positive CLV (target: {target}%)")
            report.append("   Borderline - need more bets to confirm edge.")
        else:
            report.append(f"   ❌ Warning! {current:.1f}% positive CLV (target: {target}%)")
            report.append("   You're not beating the closing line consistently.")
            report.append("   Consider: betting earlier, using sharper books, or reviewing model.")
        
        report.append("=" * 60)
        
        return "\n".join(report)


class BankrollManager:
    """
    Manages bankroll and calculates optimal stakes.
    """
    
    def __init__(self, starting_bankroll: float = None):
        self.starting_bankroll = starting_bankroll or BANKROLL["starting_amount"]
        self.current_bankroll = self.starting_bankroll
        self.history = [{
            "timestamp": datetime.now().isoformat(),
            "bankroll": self.starting_bankroll,
            "action": "initial",
        }]
    
    def update_bankroll(self, profit_loss: float, action: str = "bet_result"):
        """Update bankroll after a bet settles."""
        self.current_bankroll += profit_loss
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "bankroll": self.current_bankroll,
            "action": action,
            "change": profit_loss,
        })
    
    def calculate_stake(self, edge: float, odds: float) -> float:
        """
        Calculate stake using fractional Kelly.
        """
        if edge <= 0:
            return 0
        
        # Convert edge percentage to probability advantage
        implied_prob = 1 / odds
        our_prob = implied_prob * (1 + edge / 100)
        
        # Kelly formula
        b = odds - 1
        p = our_prob
        q = 1 - p
        
        kelly = (b * p - q) / b
        
        # Apply fraction
        kelly *= BANKROLL["kelly_fraction"]
        
        # Apply caps
        kelly = min(kelly, BANKROLL["max_bet_percent"] / 100)
        kelly = max(kelly, 0)
        
        return self.current_bankroll * kelly
    
    def get_roi(self) -> float:
        """Calculate current ROI."""
        return ((self.current_bankroll - self.starting_bankroll) / 
                self.starting_bankroll * 100)
    
    def get_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        peak = self.starting_bankroll
        max_drawdown = 0
        
        for record in self.history:
            if record["bankroll"] > peak:
                peak = record["bankroll"]
            
            drawdown = (peak - record["bankroll"]) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown


# =============================================================================
# USAGE EXAMPLE
# =============================================================================
if __name__ == "__main__":
    # Initialize tracker
    tracker = BetTracker("data/test_bets.json")
    
    # Add some sample bets
    bet1 = tracker.add_bet(
        match="Arsenal vs Chelsea",
        league="premier_league",
        market="1X2",
        selection="Home",
        odds=1.85,
        stake=50,
        bankroll=1000,
        our_probability=0.58,
        bookmaker="bet365",
    )
    print(f"Added bet: {bet1.id}")
    
    # Update with result and closing odds
    tracker.update_result(bet1.id, "won", closing_odds=1.75)
    
    # Add another bet
    bet2 = tracker.add_bet(
        match="Liverpool vs Man City",
        league="premier_league",
        market="Over/Under",
        selection="Over 2.5",
        odds=1.90,
        stake=40,
        bankroll=1050,
        our_probability=0.56,
        bookmaker="pinnacle",
    )
    
    tracker.update_result(bet2.id, "lost", closing_odds=1.85)
    
    # Generate report
    print("\n" + tracker.generate_report())
    
    # CLV Analysis
    analyzer = CLVAnalyzer(tracker)
    print("\n" + analyzer.generate_clv_report())
