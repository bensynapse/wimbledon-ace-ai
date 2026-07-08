"""Open-Meteo weather — free, no API key required."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# Major tournament → (lat, lon, city label)
TOURNAMENT_LOCATIONS = {
    "wimbledon": (51.4344, -0.2142, "London"),
    "roland garros": (48.8467, 2.2531, "Paris"),
    "french open": (48.8467, 2.2531, "Paris"),
    "us open": (40.7498, -73.8458, "New York"),
    "australian open": (-37.8225, 144.9784, "Melbourne"),
    "indian wells": (33.7206, -116.3265, "Indian Wells"),
    "miami": (25.7617, -80.1918, "Miami"),
    "monte carlo": (43.7401, 7.4266, "Monte Carlo"),
    "madrid": (40.4168, -3.7038, "Madrid"),
    "rome": (41.9028, 12.4964, "Rome"),
    "halle": (52.0379, 8.9756, "Halle"),
    "queens": (51.4875, -0.2147, "London"),
}


class WeatherSource:
    def __init__(self):
        self._cache: Dict[str, Dict] = {}

    def for_tournament(
        self,
        tournament: str,
        match_date: Optional[str] = None,
        match_hour: Optional[int] = None,
        surface: str = "",
    ) -> Dict:
        key = (tournament or "").lower().strip()
        if not key:
            return {}

        cache_key = f"{key}|{match_date or ''}|{match_hour or ''}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        coords = self._resolve_coords(key)
        if not coords:
            return {}

        lat, lon, city = coords
        try:
            data = self._fetch(lat, lon, match_date=match_date)
            result = {
                "temperature_c": data.get("temperature_c"),
                "humidity_pct": data.get("humidity_pct"),
                "wind_kmh": data.get("wind_kmh"),
                "wind_gusts_kmh": data.get("wind_gusts_kmh"),
                "heat_index": data.get("heat_index"),
                "weather_city": city,
                "indoor_outdoor": "outdoor",
                "match_date": match_date,
            }
            if match_date and data.get("hourly"):
                slot = _pick_hour_slot(data["hourly"], match_hour)
                if slot:
                    result.update(slot)
                    result["forecast_source"] = "hourly"
            impact = _grass_weather_impact(result, surface)
            result.update(impact)
            self._cache[cache_key] = result
            return result
        except Exception as exc:
            logger.warning("Weather fetch failed for %s: %s", tournament, exc)
            return {}

    def _resolve_coords(self, tournament: str) -> Optional[Tuple[float, float, str]]:
        for name, coords in TOURNAMENT_LOCATIONS.items():
            if name in tournament:
                return coords
        return None

    def _fetch(self, lat: float, lon: float, match_date: Optional[str] = None) -> Dict:
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_gusts_10m",
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_gusts_10m,precipitation",
            "wind_speed_unit": "kmh",
            "timezone": "auto",
        }
        if match_date:
            params["start_date"] = match_date[:10]
            params["end_date"] = match_date[:10]

        response = requests.get(OPEN_METEO_URL, params=params, timeout=15)
        response.raise_for_status()
        payload = response.json()
        current = payload.get("current", {})
        temp = current.get("temperature_2m")
        humidity = current.get("relative_humidity_2m")
        wind = current.get("wind_speed_10m")
        gusts = current.get("wind_gusts_10m")
        heat_index = _approx_heat_index(temp, humidity) if temp and humidity else None
        out = {
            "temperature_c": temp,
            "humidity_pct": humidity,
            "wind_kmh": wind,
            "wind_gusts_kmh": gusts,
            "heat_index": heat_index,
            "hourly": payload.get("hourly"),
        }
        return out


def _pick_hour_slot(hourly: Dict, match_hour: Optional[int]) -> Optional[Dict]:
    times: List[str] = hourly.get("time", [])
    if not times:
        return None
    idx = 0
    if match_hour is not None:
        for i, t in enumerate(times):
            try:
                hour = int(t[11:13])
            except (ValueError, IndexError):
                continue
            if hour == match_hour:
                idx = i
                break
        else:
            afternoon = [i for i, t in enumerate(times) if 13 <= int(t[11:13]) <= 18]
            idx = afternoon[len(afternoon) // 2] if afternoon else len(times) // 2
    else:
        afternoon = [i for i, t in enumerate(times) if 13 <= int(t[11:13]) <= 18]
        idx = afternoon[len(afternoon) // 2] if afternoon else len(times) // 2

    temp = hourly.get("temperature_2m", [None])[idx]
    humidity = hourly.get("relative_humidity_2m", [None])[idx]
    wind = hourly.get("wind_speed_10m", [None])[idx]
    gusts = hourly.get("wind_gusts_10m", [None])[idx]
    precip = hourly.get("precipitation", [0])[idx]
    return {
        "temperature_c": temp,
        "humidity_pct": humidity,
        "wind_kmh": wind,
        "wind_gusts_kmh": gusts,
        "precipitation_mm": precip,
        "forecast_hour": times[idx] if idx < len(times) else None,
        "heat_index": _approx_heat_index(temp, humidity) if temp and humidity else None,
    }


def _grass_weather_impact(weather: Dict, surface: str) -> Dict:
    """Grass-specific ball speed / UE risk from heat, humidity, wind."""
    surface = (surface or "").lower()
    temp = weather.get("temperature_c") or 20
    humidity = weather.get("humidity_pct") or 50
    wind = weather.get("wind_kmh") or 0
    gusts = weather.get("wind_gusts_kmh") or wind

    heat_level = "low"
    if temp >= 33:
        heat_level = "extreme"
    elif temp >= 29:
        heat_level = "high"
    elif temp >= 24:
        heat_level = "medium"

    wind_level = "low"
    effective_wind = max(wind, gusts * 0.6)
    if effective_wind >= 25 or gusts >= 35:
        wind_level = "high"
    elif effective_wind >= 15 or gusts >= 28:
        wind_level = "medium"

    ball_speed = "normal"
    notes = []
    if surface == "grass":
        if temp >= 28 and humidity <= 40:
            ball_speed = "fast"
            notes.append(f"Hot dry grass (~{temp:.0f}°C, {humidity}% humidity) — faster court, lower bounce")
        elif temp >= 26:
            ball_speed = "medium-fast"
            notes.append(f"Warm grass — lively ball speed")
        if wind_level == "high":
            notes.append(f"Wind {wind:.0f} km/h, gusts {gusts:.0f} km/h — timing errors, serve toss risk")
        elif wind_level == "medium":
            notes.append(f"Breezy ({wind:.0f} km/h) — slight UE uptick on returns and net play")

    flags = []
    if heat_level in ("high", "extreme"):
        flags.append("heat_high" if heat_level == "extreme" else "heat_medium")
    if wind_level == "high":
        flags.append("wind_high")
    elif wind_level == "medium":
        flags.append("wind_medium")

    return {
        "weather_heat_level": heat_level,
        "weather_wind_level": wind_level,
        "grass_ball_speed": ball_speed if surface == "grass" else "n/a",
        "weather_impact_notes": notes,
        "weather_context_flags": flags,
        "weather_summary": _weather_summary(temp, humidity, wind, gusts, ball_speed, surface),
    }


def _weather_summary(
    temp: float,
    humidity: float,
    wind: float,
    gusts: float,
    ball_speed: str,
    surface: str,
) -> str:
    parts = [f"{temp:.0f}°C", f"wind {wind:.0f} km/h", f"gusts {gusts:.0f} km/h", f"humidity {humidity:.0f}%"]
    if surface == "grass" and ball_speed != "n/a":
        parts.append(f"grass ball speed: {ball_speed}")
    return " | ".join(parts)


def _approx_heat_index(temp_c: float, humidity: float) -> float:
    """Simple heat stress proxy in °C."""
    if temp_c < 27:
        return temp_c
    return round(temp_c + max(0, (humidity - 40) * 0.05), 1)