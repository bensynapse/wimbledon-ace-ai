"""
WimbledonAce AI — Grand Slam Tennis Betting Predictor 2026
Configuration. Fill in API keys here or via environment variables.
"""

import os

# ============================================================
# PROJECT IDENTITY
# ============================================================

PROJECT_NAME = "WimbledonAce AI"
PROJECT_TAGLINE = "Grand Slam Tennis Betting Predictor 2026"
PROJECT_SLUG = "wimbledon-ace-ai"
GITHUB_TOPICS = [
    "wimbledon", "wimbledon-2026", "tennis", "tennis-betting", "sports-betting",
    "machine-learning", "ai", "atp", "wta", "grand-slam", "value-betting",
    "kelly-criterion", "prediction", "roland-garros", "us-open", "australian-open",
]

# ============================================================
# API KEYS
# ============================================================

# Paste keys here, or set ODDS_API_KEY / TENNIS_API_KEY env vars
ODDS_API_KEY = os.getenv("ODDS_API_KEY") or ""
TENNIS_API_KEY = os.getenv("TENNIS_API_KEY") or ""

# ============================================================
# MODEL SETTINGS
# ============================================================

MATCH_HISTORY_WINDOW = 20
MIN_VALUE_THRESHOLD = 0.05
KELLY_FRACTION = 0.25
CONFIDENCE_THRESHOLD = 0.60
BANKROLL = 1000.0

# ============================================================
# TENNIS TOURS
# ============================================================

TOURS = {
    "atp": {"name": "ATP", "odds_api_key": "tennis_atp"},
    "wta": {"name": "WTA", "odds_api_key": "tennis_wta"},
}

# Elo-style surface weights
SURFACE_ELO_WEIGHT = 0.35
OVERALL_ELO_WEIGHT = 0.65

# ============================================================
# DATA SOURCES
# ============================================================

DATA_SOURCES = {
    "odds_api": "https://api.the-odds-api.com/v4",
}

PATHS = {
    "data_dir": "data/",
    "output_dir": "output/",
}

API_KEYS = {
    "odds_api": ODDS_API_KEY,
    "tennis_api": TENNIS_API_KEY,
}