"""
Configuration for API keys and settings.
Fill in your API keys before running.
"""

# ============================================================
# API KEYS - Fill these in with your own keys
# ============================================================

# Football data APIs
FOOTBALL_API_KEY = ""          # https://www.football-data.org/ (free tier: 10 req/min)
API_FOOTBALL_KEY = ""          # https://www.api-football.com/ (free tier: 100 req/day)
ODDS_API_KEY = ""              # https://the-odds-api.com/ (free tier: 500 req/month)

# Optional: For news sentiment analysis
NEWS_API_KEY = ""              # https://newsapi.org/ (free tier: 100 req/day)

# ============================================================
# MODEL SETTINGS
# ============================================================

MATCH_HISTORY_WINDOW = 10       # Number of recent matches to consider
GOAL_DIFF_MARGIN = 2            # Margin for "outstanding performance"
MIN_VALUE_THRESHOLD = 0.05      # Minimum edge for value bet (5%)
KELLY_FRACTION = 0.25           # Quarter-Kelly for conservative sizing
CONFIDENCE_THRESHOLD = 0.60     # Minimum model confidence to flag a bet
BANKROLL = 1000.0               # Default bankroll for Kelly calculations

# ============================================================
# DATA SOURCES
# ============================================================

FOOTBALL_DATA_CO_UK_BASE = "https://www.football-data.co.uk/mmz4281/{}/{}csv"
RAPID_API_HOST = "api-football-v1.p.rapidapi.com"
