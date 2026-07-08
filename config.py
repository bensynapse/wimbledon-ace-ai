"""
WimbledonAce AI — Grand Slam Tennis Betting Predictor 2026
Configuration. Fill in API keys here or via environment variables.
"""

import os
from pathlib import Path


def _load_dotenv() -> None:
    env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

# ============================================================
# PROJECT IDENTITY
# ============================================================

PROJECT_NAME = "WimbledonAce AI"
PROJECT_TAGLINE = "Tennis Market Intelligence — Error, Fatigue & Value Detection"
PROJECT_SLUG = "wimbledon-ace-ai"
GITHUB_TOPICS = [
    "wimbledon", "wimbledon-2026", "tennis", "tennis-betting", "sports-betting",
    "machine-learning", "ai", "atp", "wta", "grand-slam", "value-betting",
    "kelly-criterion", "prediction", "roland-garros", "us-open", "australian-open",
]

# ============================================================
# API KEYS
# ============================================================

# Paste keys here, or set env vars (see .env.example)
ODDS_API_KEY = os.getenv("ODDS_API_KEY") or ""
TENNIS_API_KEY = os.getenv("TENNIS_API_KEY") or ""
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or ""
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") or ""

# Data source preference: github (free) | tennis_api (paid fallback)
DATA_SOURCE = os.getenv("DATA_SOURCE", "github")

# GitHub ATP database (free, no key) — Tennismylife/TML-Database
# Replaces Jeff Sackmann tennis_atp + api-tennis.com for ATP history/results/rankings
GITHUB_TENNIS_REPO = os.getenv(
    "GITHUB_TENNIS_REPO", "Tennismylife/TML-Database"
)

# Jeff Sackmann charting — UE/winners/rally (contextual error intelligence)
GITHUB_CHARTING_REPO = os.getenv(
    "GITHUB_CHARTING_REPO", "JeffSackmann/tennis_MatchChartingProject"
)

# Kaggle — ATP + WTA (guillemservera/tennis), requires ~/.kaggle/kaggle.json
KAGGLE_TENNIS_DATASET = os.getenv("KAGGLE_TENNIS_DATASET", "guillemservera/tennis")

# Kaggle — ATP with historical bookmaker odds (dissfya), for backtesting
KAGGLE_ATP_ODDS_DATASET = os.getenv(
    "KAGGLE_ATP_ODDS_DATASET", "dissfya/atp-tennis-2000-2023daily-pull"
)

# Recommended stack (see README):
# 1. GitHub TML-Database  → ATP matches, results, rankings, serve stats (FREE)
# 2. The Odds API           → upcoming fixtures + bookmaker odds
# 3. Kaggle guillemservera/tennis → WTA + historical ATP backup
# 4. Kaggle dissfya/atp-tennis → ATP with Odd_1/Odd_2 for backtesting
# 4. api-tennis.com         → optional paid fallback
# 4. Open-Meteo             → weather (free)
# 5. Telegram Bot API       → alerts (free)

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
    "tennis_api": "https://api.api-tennis.com/tennis/",
    "odds_api": "https://api.the-odds-api.com/v4",
    "weather_api": "https://api.open-meteo.com/v1/forecast",
}

PATHS = {
    "data_dir": "data/",
    "output_dir": "output/",
}

API_KEYS = {
    "odds_api": ODDS_API_KEY,
    "tennis_api": TENNIS_API_KEY,
    "telegram": TELEGRAM_BOT_TOKEN,
}

def api_status() -> dict:
    """Quick check which data sources are configured."""
    return {
        "github_atp": True,
        "github_charting": True,
        "kaggle": Path(f"{PATHS['data_dir']}kaggle_tennis/").exists()
        or Path.home().joinpath(".kaggle", "kaggle.json").exists(),
        "kaggle_odds": Path(f"{PATHS['data_dir']}kaggle_odds/atp_tennis.csv").exists()
        or any(Path(f"{PATHS['data_dir']}kaggle_odds/").glob("*.csv"))
        if Path(f"{PATHS['data_dir']}kaggle_odds/").exists()
        else False,
        "tennis_abstract_elo": Path(f"{PATHS['data_dir']}tennis_abstract/atp_elo.parquet").exists(),
        "data_source": DATA_SOURCE,
        "tennis_api": bool(TENNIS_API_KEY),
        "odds_api": bool(ODDS_API_KEY),
        "weather_api": True,
        "telegram": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
    }