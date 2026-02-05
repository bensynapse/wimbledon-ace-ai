#!/usr/bin/env python3
"""
QUICK START DEMO
================
Demonstrates the full betting system workflow.
Run this script to see everything in action.
"""

import os
import sys

# Ensure we're in the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print("""
╔══════════════════════════════════════════════════════════════════╗
║           🎯 BETTING ANALYSIS SYSTEM - QUICK START 🎯             ║
╠══════════════════════════════════════════════════════════════════╣
║  Dit script demonstreert alle features van het systeem.          ║
║  Volg de stappen om te zien hoe alles werkt.                     ║
╚══════════════════════════════════════════════════════════════════╝
""")

input("Druk ENTER om te beginnen...")

# =============================================================================
# STAP 1: Model Test
# =============================================================================
print("\n" + "=" * 60)
print("📊 STAP 1: PREDICTION MODEL TEST")
print("=" * 60)

from model import BettingModel, TeamStats, MatchContext, KellyCalculator

# Create sample teams with realistic stats
print("\nTeam data laden...")

ajax = TeamStats(
    name="Ajax",
    xg_for=42.5,
    xg_against=18.3,
    goals_scored=45,
    goals_conceded=16,
    matches_played=18,
    elo_rating=1780,
)

psv = TeamStats(
    name="PSV",
    xg_for=48.2,
    xg_against=22.1,
    goals_scored=52,
    goals_conceded=20,
    matches_played=18,
    elo_rating=1820,
)

# Add context
context = MatchContext(
    referee_cards_per_game=4.1,
    home_motivation="title_race",
    away_motivation="title_race",
    home_days_rest=7,
    away_days_rest=5,
    is_derby=True,
)

# Sample odds (zoals je ze bij een bookmaker zou vinden)
odds = {
    "home_win": 2.40,
    "draw": 3.50,
    "away_win": 2.85,
    "over_2_5": 1.65,
    "under_2_5": 2.20,
    "btts_yes": 1.55,
    "btts_no": 2.40,
}

print("\n🏠 AJAX (Thuis)")
print(f"   xG: {ajax.xg_for} | xGA: {ajax.xg_against}")
print(f"   Goals: {ajax.goals_scored} | Conceded: {ajax.goals_conceded}")
print(f"   ELO: {ajax.elo_rating}")

print("\n🚗 PSV (Uit)")
print(f"   xG: {psv.xg_for} | xGA: {psv.xg_against}")
print(f"   Goals: {psv.goals_scored} | Conceded: {psv.goals_conceded}")
print(f"   ELO: {psv.elo_rating}")

print("\n📋 Context:")
print(f"   Derby match: {'Ja' if context.is_derby else 'Nee'}")
print(f"   Home motivatie: {context.home_motivation}")
print(f"   Away rust dagen: {context.away_days_rest}")

input("\nDruk ENTER voor de voorspelling...")

# Generate prediction
model = BettingModel()
prediction = model.predict_match(ajax, psv, context)

# Print report
print(model.generate_report(ajax, psv, prediction, odds))

input("\nDruk ENTER voor Kelly berekeningen...")

# =============================================================================
# STAP 2: Kelly Criterion Demo
# =============================================================================
print("\n" + "=" * 60)
print("💰 STAP 2: KELLY CRITERION - STAKE SIZING")
print("=" * 60)

kelly = KellyCalculator(fraction=0.5, max_bet_percent=5.0)
bankroll = 1000

print(f"\nBankroll: €{bankroll}")
print(f"Kelly Fraction: 0.5 (Half Kelly)")
print(f"Max Bet: 5% van bankroll = €{bankroll * 0.05}")

print("\n📊 STAKE BEREKENINGEN VOOR VALUE BETS:")
print("-" * 50)

# Check each market
value_bets = model.find_value_bets(prediction, odds)

if value_bets:
    for bet in value_bets:
        stake = kelly.calculate_stake(bankroll, bet["our_probability"] / 100, bet["odds"])
        print(f"\n{bet['market']} - {bet['selection']}")
        print(f"   Onze kans: {bet['our_probability']:.1f}%")
        print(f"   Bookmaker kans: {bet['implied_probability']:.1f}%")
        print(f"   Edge: {bet['edge_percent']:.2f}%")
        print(f"   Odds: {bet['odds']}")
        print(f"   Kelly stake: €{stake:.2f} ({stake/bankroll*100:.1f}%)")
else:
    print("\n❌ Geen value bets gevonden bij deze odds.")
    print("   Dit betekent dat de bookmaker odds te scherp zijn.")

input("\nDruk ENTER voor bet tracking demo...")

# =============================================================================
# STAP 3: Bet Tracking & CLV
# =============================================================================
print("\n" + "=" * 60)
print("📝 STAP 3: BET TRACKING & CLV ANALYSE")
print("=" * 60)

from tracker import BetTracker, CLVAnalyzer

# Create a temporary tracker for demo
tracker = BetTracker("data/demo_bets.json")

print("\nSimulatie: 5 bets plaatsen en resultaten invoeren...")

# Simulate some bets
demo_bets = [
    {"match": "Ajax vs PSV", "selection": "Home", "odds": 2.40, "closing": 2.25, "result": "won", "prob": 0.45},
    {"match": "Feyenoord vs AZ", "selection": "Over 2.5", "odds": 1.75, "closing": 1.70, "result": "won", "prob": 0.60},
    {"match": "Utrecht vs Twente", "selection": "BTTS Yes", "odds": 1.65, "closing": 1.60, "result": "lost", "prob": 0.62},
    {"match": "Vitesse vs Heerenveen", "selection": "Home", "odds": 1.90, "closing": 2.00, "result": "won", "prob": 0.55},
    {"match": "Groningen vs Sparta", "selection": "Draw", "odds": 3.40, "closing": 3.30, "result": "lost", "prob": 0.32},
]

demo_bankroll = 1000
for i, d in enumerate(demo_bets):
    stake = 30 + (i * 5)  # Variërende stakes
    bet = tracker.add_bet(
        match=d["match"],
        league="eredivisie",
        market="Demo",
        selection=d["selection"],
        odds=d["odds"],
        stake=stake,
        bankroll=demo_bankroll,
        our_probability=d["prob"],
        bookmaker="demo_book",
    )
    tracker.update_result(bet.id, d["result"], d["closing"])
    
    clv = ((d["odds"] - d["closing"]) / d["closing"]) * 100
    status = "✅" if d["result"] == "won" else "❌"
    clv_status = "📈" if clv > 0 else "📉"
    
    print(f"   {status} {d['match']}: {d['selection']} @ {d['odds']} (CLV: {clv:.1f}% {clv_status})")

print("\n" + tracker.generate_report())

# CLV Analysis
analyzer = CLVAnalyzer(tracker)
print(analyzer.generate_clv_report())

input("\nDruk ENTER voor backtest info...")

# =============================================================================
# STAP 4: Backtest Info
# =============================================================================
print("\n" + "=" * 60)
print("🔬 STAP 4: BACKTESTING")
print("=" * 60)

print("""
Backtesting laat je je strategie testen op historische data.

VOORBEELD GEBRUIK:
```python
from backtest import Backtester, BacktestConfig

config = BacktestConfig(
    starting_bankroll=1000,
    kelly_fraction=0.5,
    min_edge=3.0,
)

backtester = Backtester(config)

# Test op Premier League 2023/24
result = backtester.run_backtest("E0", ["2324"])

print(backtester.generate_report(result))
```

BESCHIKBARE LEAGUES:
- E0  = Premier League
- SP1 = La Liga
- D1  = Bundesliga
- I1  = Serie A
- F1  = Ligue 1
- N1  = Eredivisie

Om een echte backtest te draaien:
```bash
python backtest.py
```
""")

input("\nDruk ENTER voor line shopping info...")

# =============================================================================
# STAP 5: Line Shopping
# =============================================================================
print("\n" + "=" * 60)
print("🔍 STAP 5: LINE SHOPPING")
print("=" * 60)

print("""
Line shopping = bij meerdere bookmakers de beste odds zoeken.

WAAROM BELANGRIJK:
- 0.05 verschil in odds kan 20% meer winst betekenen over tijd
- Sommige books zijn scherper dan andere
- Pinnacle heeft de laagste marge (beste voor CLV benchmark)

VOORBEELD:

Match: Ajax vs PSV

Bookmaker    | Home  | Draw  | Away
-------------|-------|-------|------
Bet365       | 2.35  | 3.45  | 2.80
Unibet       | 2.40  | 3.50  | 2.75  👑 Best Home
Pinnacle     | 2.38  | 3.55  | 2.82  👑 Best Draw/Away
888sport     | 2.30  | 3.40  | 2.78

→ Voor Home bet: Unibet (2.40)
→ Voor Draw bet: Pinnacle (3.55)
→ Voor Away bet: Pinnacle (2.82)

CODE VOORBEELD:
```python
from main import LineShopper

shopper = LineShopper()
best_odds = shopper.find_best_odds("premier_league")

for match in best_odds:
    print(f"{match['home_team']} vs {match['away_team']}")
    print(f"  Best Home: {match['best_home_odds']}")
```

NOTE: Vereist The Odds API key in config.py
""")

input("\nDruk ENTER voor final summary...")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("✅ SYSTEEM OVERZICHT")
print("=" * 60)

print("""
JE HEBT NU:

1. ✅ Prediction Model
   - Poisson distributie voor goals
   - Context adjustments (motivatie, vermoeidheid, weer)
   - Value bet identificatie

2. ✅ Kelly Criterion Calculator
   - Half Kelly voor veilige stake sizing
   - Automatic edge berekening
   - Max bet cap (5%)

3. ✅ Bet Tracker
   - Alle bets loggen met details
   - CLV tracking per bet
   - Performance metrics

4. ✅ CLV Analyzer
   - Closing Line Value analyse
   - Performance per market/bookmaker
   - Long-term edge validatie

5. ✅ Backtest Framework
   - Test op historische data
   - Parameter optimalisatie
   - ROI en CLV validatie

VOLGENDE STAPPEN:

1. Vul je API keys in config.py
2. Run een backtest: python backtest.py
3. Start met paper trading (fake bets loggen)
4. Na 100+ bets: evalueer CLV positief rate
5. Als CLV > 50%: overweeg echt geld

ONTHOUD:
- CLV > win/loss record
- Half Kelly, NOOIT full Kelly
- Minimum 500 bets voor conclusies
- Track ALTIJD closing odds

""")

print("=" * 60)
print("🎯 HAPPY BETTING! May the odds be in your favor. 🍀")
print("=" * 60)

# Cleanup demo file
try:
    os.remove("data/demo_bets.json")
except:
    pass
