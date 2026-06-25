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

# ============================================================
# WC / WORLD CUP 2026 SPECIFIC (for WK bot)
# ============================================================

# API-Football identifiers for the tournament (see guide)
WC_LEAGUE_ID = 1
WC_SEASON = 2026

# Weather (OpenWeatherMap)
WEATHER_API_KEY = ""   # https://openweathermap.org

# ============================================================
# VENUES_2026 - 16 stadiums with key environmental data
# Keys chosen for flexible matching (city, stadium short names, teams)
# lat/lon for forecasts; alt_m critical for performance; roof_ac for heat mitigation
# ============================================================

VENUES_2026 = {
    # Mexico - high altitude + heat factors
    "mexico city": {
        "name": "Estadio Azteca",
        "city": "Mexico City",
        "country": "Mexico",
        "lat": 19.303,
        "lon": -99.151,
        "alt_m": 2200,
        "capacity": 87800,
        "roof_ac": False,
        "june_high_c": 26,
        "notes": "Highest altitude WC venue; thin air reduces stamina, favors adapted teams or low-intensity styles"
    },
    "azteca": {
        "name": "Estadio Azteca",
        "city": "Mexico City",
        "country": "Mexico",
        "lat": 19.303,
        "lon": -99.151,
        "alt_m": 2200,
        "capacity": 87800,
        "roof_ac": False,
        "june_high_c": 26,
        "notes": "Highest altitude WC venue"
    },
    "guadalajara": {
        "name": "Estadio Akron",
        "city": "Zapopan / Guadalajara",
        "country": "Mexico",
        "lat": 20.682,
        "lon": -103.462,
        "alt_m": 1566,
        "capacity": 45664,
        "roof_ac": False,
        "june_high_c": 30,
        "notes": "Moderate altitude + heat"
    },
    "monterrey": {
        "name": "Estadio BBVA",
        "city": "Monterrey",
        "country": "Mexico",
        "lat": 25.669,
        "lon": -100.282,
        "alt_m": 540,
        "capacity": 51200,
        "roof_ac": False,
        "june_high_c": 34,
        "notes": "Hot, humid potential"
    },
    # Canada - cooler
    "vancouver": {
        "name": "BC Place",
        "city": "Vancouver",
        "country": "Canada",
        "lat": 49.276,
        "lon": -123.112,
        "alt_m": 0,
        "capacity": 52500,
        "roof_ac": True,
        "june_high_c": 20,
        "notes": "Retractable roof; cooler maritime climate"
    },
    "toronto": {
        "name": "BMO Field",
        "city": "Toronto",
        "country": "Canada",
        "lat": 43.633,
        "lon": -79.419,
        "alt_m": 76,
        "capacity": 45000,
        "roof_ac": False,
        "june_high_c": 23,
        "notes": "Lakeside; moderate"
    },
    # USA - mix of heat, roofs, large crowds
    "atlanta": {
        "name": "Mercedes-Benz Stadium",
        "city": "Atlanta",
        "country": "USA",
        "lat": 33.756,
        "lon": -84.400,
        "alt_m": 320,
        "capacity": 71000,
        "roof_ac": True,
        "june_high_c": 30,
        "notes": "Retractable roof + AC; strong home atmosphere possible"
    },
    "dallas": {
        "name": "AT&T Stadium",
        "city": "Arlington / Dallas",
        "country": "USA",
        "lat": 32.748,
        "lon": -97.093,
        "alt_m": 180,
        "capacity": 80000,
        "roof_ac": True,
        "june_high_c": 34,
        "notes": "Large retractable; huge crowds"
    },
    "houston": {
        "name": "NRG Stadium",
        "city": "Houston",
        "country": "USA",
        "lat": 29.685,
        "lon": -95.411,
        "alt_m": 10,
        "capacity": 72000,
        "roof_ac": True,
        "june_high_c": 33,
        "notes": "Retractable AC; extreme humidity risk outside"
    },
    "miami": {
        "name": "Hard Rock Stadium",
        "city": "Miami Gardens",
        "country": "USA",
        "lat": 25.958,
        "lon": -80.239,
        "alt_m": 0,
        "capacity": 65000,
        "roof_ac": False,
        "june_high_c": 31,
        "notes": "Heat + humidity; high crowd energy"
    },
    "los angeles": {
        "name": "SoFi Stadium",
        "city": "Inglewood / Los Angeles",
        "country": "USA",
        "lat": 33.953,
        "lon": -118.339,
        "alt_m": 20,
        "capacity": 70000,
        "roof_ac": "partial",
        "june_high_c": 24,
        "notes": "Modern, covered elements"
    },
    "new york": {
        "name": "MetLife Stadium",
        "city": "East Rutherford",
        "country": "USA",
        "lat": 40.813,
        "lon": -74.074,
        "alt_m": 3,
        "capacity": 82500,
        "roof_ac": False,
        "june_high_c": 27,
        "notes": "High capacity; Northeast summer"
    },
    "seattle": {
        "name": "Lumen Field",
        "city": "Seattle",
        "country": "USA",
        "lat": 47.595,
        "lon": -122.331,
        "alt_m": 0,
        "capacity": 67000,
        "roof_ac": False,
        "june_high_c": 22,
        "notes": "Cooler, rainy potential"
    },
    "boston": {
        "name": "Gillette Stadium",
        "city": "Foxborough",
        "country": "USA",
        "lat": 42.091,
        "lon": -71.264,
        "alt_m": 70,
        "capacity": 65000,
        "roof_ac": False,
        "june_high_c": 25,
        "notes": "Northeast conditions"
    },
    "kansas city": {
        "name": "Arrowhead Stadium",
        "city": "Kansas City",
        "country": "USA",
        "lat": 39.049,
        "lon": -94.484,
        "alt_m": 270,
        "capacity": 76000,
        "roof_ac": False,
        "june_high_c": 29,
        "notes": "Loud home crowd potential"
    },
    "philadelphia": {
        "name": "Lincoln Financial Field",
        "city": "Philadelphia",
        "country": "USA",
        "lat": 39.901,
        "lon": -75.168,
        "alt_m": 10,
        "capacity": 69000,
        "roof_ac": False,
        "june_high_c": 28,
        "notes": "Intense atmosphere"
    },
    "san francisco": {
        "name": "Levi's Stadium",
        "city": "Santa Clara",
        "country": "USA",
        "lat": 37.403,
        "lon": -121.970,
        "alt_m": 5,
        "capacity": 68500,
        "roof_ac": False,
        "june_high_c": 24,
        "notes": "Bay area mild"
    },
}

# ============================================================
# ENVIRONMENTAL & TOURNAMENT IMPACT FACTORS (for xG / prob adjustments)
# ============================================================

HEAT_MODIFIERS = {
    "extreme_heat": {"goals_modifier": 0.88, "notes": "High fatigue, fewer goals, more subs"},
    "heat_humidity": {"goals_modifier": 0.92, "notes": "Moderate suppression"},
    "high_altitude": {"goals_modifier": 0.93, "notes": "Lower oxygen, affects away sides more initially"},
    "normal": {"goals_modifier": 1.0},
    "cool_rain": {"goals_modifier": 0.97},
}

CROWD_FACTORS = {
    "high_attendance_host_confed": 1.06,   # partisan boost
    "high_attendance_neutral": 1.02,
    "low_attendance": 0.98,
}

ADAPTATION_MODIFIERS = {
    # Simple multipliers based on team region vs venue conditions
    "same_climate": 1.03,
    "mild_to_hot": 0.96,
    "euro_to_high_alt": 0.90,
    "default": 1.0,
}

# Existing advanced settings for compatibility with model.py / betting_system_complete
POISSON_SETTINGS = {
    "max_goals": 10,
    "home_advantage": 1.25,
    "xg_weight": 0.6,
    "form_weight": 0.25,
    "season_weight": 0.15,
}

ELO_SETTINGS = {"k_factor": 20, "home_adv": 50}

MOTIVATION_FACTORS = {
    "normal": 1.0,
    "must_win": 1.08,
    "dead_rubber": 0.92,
    "title_race": 1.05,
    "relegation_battle": 1.05,
}

CONGESTION_SETTINGS = {
    "high": 0.95,   # many matches in short time
    "normal": 1.0,
}

WEATHER_IMPACT = {  # legacy mapping
    "normal": {"goals_modifier": 1.0},
    "rain": {"goals_modifier": 0.97},
    "wind": {"goals_modifier": 0.96},
    "extreme_heat": {"goals_modifier": 0.88},
    "extreme_cold": {"goals_modifier": 0.95},
}

REFEREE_SETTINGS = {
    "strict_yellows": 5.5,
    "lenient": 3.0,
}

KEY_NUMBERS = {"over_2_5": 2.5}

# ==========================================================
# BACK-COMPAT FOR scrapers / backtest / model (prevent import errors)
# ==========================================================

API_KEYS = {
    "football_data": FOOTBALL_API_KEY,
    "api_football": API_FOOTBALL_KEY,
    "odds_api": ODDS_API_KEY,
    "news_api": NEWS_API_KEY,
    "weather_api": WEATHER_API_KEY,
}

# Minimal league config (extend as needed)
LEAGUES = {
    "premier_league": {"name": "Premier League", "country": "England", "football_data_code": "E0", "understat_name": "EPL", "soccerdata_name": "ENG-Premier League"},
    "world_cup": {"name": "FIFA World Cup 2026", "country": "World", "football_data_code": None, "understat_name": None, "soccerdata_name": "INT-World Cup"},
    "wc": {"name": "FIFA World Cup 2026", "country": "World", "football_data_code": None, "understat_name": None, "soccerdata_name": "INT-World Cup"},
}

# Soccerdata primary config (use for FBref, Understat, ClubElo, WhoScored, ESPN, FotMob, SoFIFA etc as unified source)
SOCCERDATA_CONFIG = {
    "enabled": True,
    "wc_league": "INT-World Cup",
    # seasons can be like "2022", "2018" for past WC; "2026" for upcoming (will use latest available + qualifiers)
    "default_wc_seasons": ["2022", "2018", "2014"],
    "cache_note": "Data cached under ~/soccerdata or SOCCERDATA_DIR; use no_cache=True for fresh pulls",
    "sources": ["FBref", "WhoScored", "Understat", "ClubElo", "ESPN", "FotMob", "SoFIFA"],
}

# StatsBomb Open Data (https://github.com/statsbomb/open-data)
# Clone the repo locally for full offline access to all historical WC events + 360 data
# Path to local clone, e.g. "/path/to/statsbomb/open-data"
STATSBOMB_OPEN_DATA_DIR = None  # or set to local path
STATSBOMB_WC_COMPETITION_ID = 43
STATSBOMB_WC_SEASONS = {
    106: "2022",
    3: "2018",
    # older historical available too
}

DATA_SOURCES = {
    "odds_api": "https://api.the-odds-api.com/v4",
    "football_data_co_uk": FOOTBALL_DATA_CO_UK_BASE,
    # Legacy keys kept for fallbacks (soccerdata is now preferred for xG/ELO/stats)
    "understat": "https://understat.com",
    "club_elo": "http://clubelo.com",
}

PATHS = {
    "data_dir": "data/",
    "models_dir": "models/",
    "output_dir": "output/",
}

VALUE_THRESHOLDS = {"min_edge": MIN_VALUE_THRESHOLD}
CLV_SETTINGS = {"benchmark": "pinnacle"}
