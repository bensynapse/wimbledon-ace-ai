"""
BETTING SYSTEM CONFIGURATION
============================
Central configuration for the complete betting analysis system.
"""

# =============================================================================
# API KEYS (Vul hier je eigen keys in)
# =============================================================================
API_KEYS = {
    "odds_api": "YOUR_ODDS_API_KEY",  # https://the-odds-api.com (500 free/month)
    "weather_api": "YOUR_OPENWEATHER_KEY",  # https://openweathermap.org (free tier)
}

# =============================================================================
# SUPPORTED LEAGUES
# =============================================================================
LEAGUES = {
    "eredivisie": {
        "name": "Eredivisie",
        "country": "Netherlands",
        "odds_api_key": "soccer_netherlands_eredivisie",
        "fbref_id": "23",
        "understat_name": "Eredivisie",
        "football_data_code": "N1",
    },
    "premier_league": {
        "name": "Premier League",
        "country": "England",
        "odds_api_key": "soccer_epl",
        "fbref_id": "9",
        "understat_name": "EPL",
        "football_data_code": "E0",
    },
    "la_liga": {
        "name": "La Liga",
        "country": "Spain",
        "odds_api_key": "soccer_spain_la_liga",
        "fbref_id": "12",
        "understat_name": "La_Liga",
        "football_data_code": "SP1",
    },
    "bundesliga": {
        "name": "Bundesliga",
        "country": "Germany",
        "odds_api_key": "soccer_germany_bundesliga",
        "fbref_id": "20",
        "understat_name": "Bundesliga",
        "football_data_code": "D1",
    },
    "serie_a": {
        "name": "Serie A",
        "country": "Italy",
        "odds_api_key": "soccer_italy_serie_a",
        "fbref_id": "11",
        "understat_name": "Serie_A",
        "football_data_code": "I1",
    },
    "ligue_1": {
        "name": "Ligue 1",
        "country": "France",
        "odds_api_key": "soccer_france_ligue_one",
        "fbref_id": "13",
        "understat_name": "Ligue_1",
        "football_data_code": "F1",
    },
}

# =============================================================================
# BANKROLL MANAGEMENT SETTINGS
# =============================================================================
BANKROLL = {
    "starting_amount": 1000,  # Starting bankroll in EUR
    "kelly_fraction": 0.5,    # Use Half Kelly (0.5) - NEVER full Kelly
    "max_bet_percent": 5.0,   # Maximum 5% of bankroll per bet
    "min_edge_threshold": 3.0,  # Minimum edge % to place bet
    "min_odds": 1.30,         # Minimum odds to consider
    "max_odds": 4.00,         # Maximum odds (higher = more variance)
}

# =============================================================================
# VALUE BET THRESHOLDS
# =============================================================================
VALUE_THRESHOLDS = {
    "minimum_ev": 3.0,        # Minimum Expected Value %
    "minimum_prob_diff": 5.0,  # Minimum probability difference %
    "confidence_levels": {
        "high": 7.0,          # EV >= 7% = high confidence
        "medium": 5.0,        # EV >= 5% = medium confidence
        "low": 3.0,           # EV >= 3% = low confidence
    }
}

# =============================================================================
# CLV (CLOSING LINE VALUE) SETTINGS
# =============================================================================
CLV_SETTINGS = {
    "benchmark_book": "pinnacle",  # Sharpest book for CLV comparison
    "track_all_bets": True,
    "positive_clv_target": 50.0,   # Target: >50% of bets should beat closing line
}

# =============================================================================
# MOTIVATION FACTORS (Impact multipliers)
# =============================================================================
MOTIVATION_FACTORS = {
    "relegation_battle": 1.15,      # Teams fighting relegation try harder
    "title_race": 1.10,             # Championship contenders
    "europa_race": 1.08,            # Fighting for European spots
    "nothing_to_play": 0.90,        # Mid-table, season over
    "derby_match": 1.12,            # Local rivalry
    "manager_first_match": 1.10,    # New manager bounce
    "manager_last_match": 0.95,     # Manager about to be fired
    "cup_final_hangover": 0.92,     # After big cup match
}

# =============================================================================
# FIXTURE CONGESTION SETTINGS
# =============================================================================
CONGESTION_SETTINGS = {
    "days_rest_optimal": 7,         # Optimal rest days
    "days_rest_minimum": 3,         # Minimum acceptable
    "matches_30_days_warning": 8,   # Warning if >8 matches in 30 days
    "europa_thursday_penalty": 0.95,  # Playing Sunday after Europa Thursday
    "champions_league_travel": {
        "short": 0.98,              # <2000km travel
        "medium": 0.95,             # 2000-4000km
        "long": 0.90,               # >4000km (e.g., Russia, Turkey)
    }
}

# =============================================================================
# REFEREE IMPACT SETTINGS
# =============================================================================
REFEREE_SETTINGS = {
    "cards_per_game_high": 4.5,     # Above this = card-happy referee
    "cards_per_game_low": 3.0,      # Below this = lenient referee
    "home_bias_threshold": 0.15,    # >15% difference = home bias
    "penalty_prone_threshold": 0.35,  # >0.35 penalties/game = penalty prone
}

# =============================================================================
# WEATHER IMPACT SETTINGS
# =============================================================================
WEATHER_IMPACT = {
    "rain_heavy": {
        "goals_modifier": 0.90,     # 10% fewer goals
        "btts_modifier": 0.95,
    },
    "wind_strong": {
        "goals_modifier": 0.92,
        "corners_modifier": 0.85,
    },
    "extreme_heat": {
        "goals_modifier": 0.95,
        "late_goals_modifier": 1.10,  # More goals in 2nd half (fatigue)
    },
    "extreme_cold": {
        "goals_modifier": 0.93,
    }
}

# =============================================================================
# POISSON MODEL SETTINGS
# =============================================================================
POISSON_SETTINGS = {
    "max_goals": 10,                # Maximum goals to calculate
    "home_advantage": 1.25,         # Home team scores ~25% more
    "xg_weight": 0.6,               # Weight for xG vs actual goals
    "form_weight": 0.3,             # Weight for recent form
    "season_weight": 0.1,           # Weight for full season stats
    "form_matches": 5,              # Number of recent matches for form
}

# =============================================================================
# ELO RATING SETTINGS
# =============================================================================
ELO_SETTINGS = {
    "k_factor": 20,                 # How quickly ratings change
    "home_advantage_elo": 100,      # Home team gets +100 ELO boost
    "default_elo": 1500,            # Starting ELO for new teams
}

# =============================================================================
# DATA SOURCES URLS
# =============================================================================
DATA_SOURCES = {
    "understat": "https://understat.com",
    "fbref": "https://fbref.com",
    "football_data": "https://www.football-data.co.uk",
    "transfermarkt": "https://www.transfermarkt.com",
    "club_elo": "http://clubelo.com",
    "sofascore": "https://www.sofascore.com",
    "odds_api": "https://api.the-odds-api.com/v4",
}

# =============================================================================
# FILE PATHS
# =============================================================================
PATHS = {
    "data_dir": "data/",
    "historical_data": "data/historical/",
    "odds_data": "data/odds/",
    "predictions": "data/predictions/",
    "backtest_results": "data/backtest/",
    "bet_tracker": "data/bet_tracker.json",
    "clv_tracker": "data/clv_tracker.json",
}

# =============================================================================
# SCORING PATTERNS (When teams typically score)
# =============================================================================
SCORING_PATTERNS = {
    # Percentage of goals scored in each 15-min period
    "default": {
        "0-15": 0.10,
        "16-30": 0.12,
        "31-45": 0.15,
        "46-60": 0.18,
        "61-75": 0.20,
        "76-90": 0.25,
    }
}

# =============================================================================
# KEY NUMBERS (Important score margins)
# =============================================================================
KEY_NUMBERS = {
    "football": [0, 1, 2],  # Most common goal differences
    "most_common_scores": [
        (1, 0), (2, 1), (1, 1), (2, 0), (0, 0),
        (2, 2), (3, 1), (1, 2), (3, 0), (0, 1)
    ]
}
