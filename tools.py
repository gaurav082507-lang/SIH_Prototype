import json
import math
import os
from datetime import datetime, timezone
from typing import List, Tuple

import pandas as pd
import requests
import requests_cache
import openmeteo_requests
from dotenv import load_dotenv
from retry_requests import retry
from langchain_core.tools import tool

load_dotenv()


# ============================================================
# Shared Open-Meteo client
# (previously created twice — once for weather, once for ocean —
#  which pointed two separate CachedSession objects at the same
#  ".cache" file for no reason. One shared client is used by both.)
# ============================================================

cache_session = requests_cache.CachedSession(
    ".cache",
    expire_after=3600
)

retry_session = retry(
    cache_session,
    retries=5,
    backoff_factor=0.2
)

openmeteo = openmeteo_requests.Client(
    session=retry_session
)

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
OCEAN_URL = "https://marine-api.open-meteo.com/v1/marine"


# ============================================================
# TOOL: generate_grid
# Deterministic — the Planner calls this instead of computing
# lat/lon offsets itself, which LLMs are unreliable at.
# ============================================================

@tool
def generate_grid(center_lat: float, center_lon: float,
                   radius_km: float = 40, num_points: int = 10) -> list[dict]:
    """
    Generate candidate coordinates evenly spaced in a ring around a
    center point, for area-based marine scans (e.g. fishing-zone
    recommendations, broad marine-safety checks).

    Use this whenever an area scan is needed instead of computing
    lat/lon offsets yourself — this tool returns exact coordinates,
    while manual arithmetic is error-prone and imprecise.

    Args:
        center_lat: Latitude of the center point (the user's location).
        center_lon: Longitude of the center point (the user's location).
        radius_km: Distance from center to each candidate point, in km.
            Defaults to 40 km, a reasonable range for a fishing vessel.
        num_points: How many points to generate, evenly spaced around
            the circle. Defaults to 10.

    Returns:
        A list of dicts, each with: "id", "latitude", "longitude",
        "bearing_deg" (0-359, direction from center), and "distance_km".
    """
    points = []
    for i in range(num_points):
        bearing_deg = (360 / num_points) * i
        bearing_rad = math.radians(bearing_deg)

        delta_lat = (radius_km / 111.0) * math.cos(bearing_rad)
        delta_lon = (radius_km / (111.0 * math.cos(math.radians(center_lat)))) * math.sin(bearing_rad)

        points.append({
            "id": f"P{i + 1}",
            "latitude": round(center_lat + delta_lat, 4),
            "longitude": round(center_lon + delta_lon, 4),
            "bearing_deg": round(bearing_deg),
            "distance_km": radius_km,
        })
    return points


# ============================================================
# Weather (Open-Meteo forecast API)
# ============================================================

_WEATHER_HOURLY_VARS = [
    "temperature_2m",
    "precipitation",
    "precipitation_probability",
    "rain",
    "weather_code",
    "cloud_cover",
    "visibility",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "surface_pressure",
]

_WEATHER_DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "precipitation_probability_max",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
    "weather_code",
]


def _parse_weather_response(response, latitude, longitude):

    hourly = response.Hourly()

    hourly_data = {
        "time": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )
    }

    for i, variable in enumerate(_WEATHER_HOURLY_VARS):
        hourly_data[variable] = hourly.Variables(i).ValuesAsNumpy()

    hourly_df = pd.DataFrame(hourly_data)

    daily = response.Daily()

    daily_data = {
        "time": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        )
    }

    for i, variable in enumerate(_WEATHER_DAILY_VARS):
        daily_data[variable] = daily.Variables(i).ValuesAsNumpy()

    daily_df = pd.DataFrame(daily_data)

    return {
        "location": {
            "latitude": latitude,
            "longitude": longitude
        },
        "timezone": response.Timezone(),
        "hourly": hourly_df.to_dict(orient="records"),
        "daily": daily_df.to_dict(orient="records")
    }


def _fetch_weather_batch(coords: List[Tuple[float, float]]):

    latitudes = [c[0] for c in coords]
    longitudes = [c[1] for c in coords]

    params = {
        "latitude": latitudes,
        "longitude": longitudes,
        "hourly": _WEATHER_HOURLY_VARS,
        "daily": _WEATHER_DAILY_VARS,
        "forecast_days": 7,
        "timezone": "auto"
    }

    # NOTE on timeouts: this call goes through openmeteo_requests, whose
    # session (retry_session above) does not have an explicit numeric
    # timeout configured here either way — there was never one to
    # "remove" on this path. retry_session only adds retry/backoff on
    # top of a plain requests session, whose default is also "wait
    # indefinitely" unless a timeout is passed per-request, which
    # openmeteo_requests.Client() does not currently expose here.
    responses = openmeteo.weather_api(WEATHER_URL, params=params)

    results = []
    for response, (lat, lon) in zip(responses, coords):
        results.append(_parse_weather_response(response, lat, lon))

    return results


def get_weather_data(latitude: float, longitude: float) -> dict:
    """
    Plain (non-tool) function used by weather_node.py to fetch weather
    for a single point. Kept undecorated so nodes can call it directly
    with keyword arguments — a @tool-wrapped function cannot be called
    like a normal Python function (it raises TypeError).
    """
    results = _fetch_weather_batch([(latitude, longitude)])
    return results[0]


def get_weather_for_grid(locations: list) -> list:
    """
    Fetch weather for all grid points in one batched Open-Meteo call.

    locations format:

    [
        {"id": "center", "latitude": 21.63, "longitude": 87.51},
        ...
    ]
    """
    coords = [
        (location["latitude"], location["longitude"])
        for location in locations
    ]

    results = _fetch_weather_batch(coords)

    final_results = []
    for location, result in zip(locations, results):
        result["id"] = location.get("id")
        final_results.append(result)

    return final_results


@tool
def weather_agent(latitude: float, longitude: float) -> dict:
    """
    Get weather forecast data for a specific latitude
    and longitude using Open-Meteo. LangChain-tool-wrapped version
    of get_weather_data, for use by LLM agents (e.g. a chatbot).
    """
    return get_weather_data(latitude=latitude, longitude=longitude)


# ============================================================
# Ocean (Open-Meteo marine API)
# ============================================================

_OCEAN_HOURLY_VARS = [
    "wave_height",
    "wave_direction",
    "wave_period",
    "swell_wave_height",
    "swell_wave_direction",
    "swell_wave_period",
    "sea_surface_temperature",
    "ocean_current_velocity",
    "ocean_current_direction",
]


def _parse_ocean_response(response, latitude, longitude):

    hourly = response.Hourly()

    hourly_data = {
        "time": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )
    }

    for i, variable in enumerate(_OCEAN_HOURLY_VARS):
        values = hourly.Variables(i).ValuesAsNumpy()
        hourly_data[variable] = values.round(2)

    hourly_df = pd.DataFrame(hourly_data)

    return {
        "location": {
            "latitude": latitude,
            "longitude": longitude
        },
        "timezone": response.Timezone(),
        "hourly": hourly_df.to_dict(orient="records")
    }


def _fetch_ocean_batch(coords: List[Tuple[float, float]]):

    latitudes = [c[0] for c in coords]
    longitudes = [c[1] for c in coords]

    params = {
        "latitude": latitudes,
        "longitude": longitudes,
        "hourly": _OCEAN_HOURLY_VARS,
        "forecast_days": 7,
        "timezone": "auto"
    }

    # Same note as _fetch_weather_batch: no explicit timeout was ever
    # set on this path, so there is nothing to remove here.
    responses = openmeteo.weather_api(OCEAN_URL, params=params)

    results = []
    for response, (lat, lon) in zip(responses, coords):
        results.append(_parse_ocean_response(response, lat, lon))

    return results


def get_ocean_data_plain(latitude: float, longitude: float) -> dict:
    """
    Plain (non-tool) function for a single-point ocean lookup, safe to
    call directly with keyword arguments.
    """
    results = _fetch_ocean_batch([(latitude, longitude)])
    return results[0]


def get_ocean_data_for_grid(locations: list) -> list:
    """
    Fetch ocean data for all grid points in one batched Open-Meteo call.

    locations format:

    [
        {"id": "center", "latitude": 21.63, "longitude": 87.51},
        ...
    ]
    """
    coords = [
        (location["latitude"], location["longitude"])
        for location in locations
    ]

    results = _fetch_ocean_batch(coords)

    final_results = []
    for location, result in zip(locations, results):
        result["id"] = location.get("id")
        final_results.append(result)

    return final_results


@tool
def get_ocean_data(latitude: float, longitude: float) -> dict:
    """
    Get ocean conditions for a specific latitude
    and longitude using Open-Meteo Marine API. LangChain-tool-wrapped
    version of get_ocean_data_plain, for use by LLM agents.
    """
    return get_ocean_data_plain(latitude=latitude, longitude=longitude)


# ============================================================
# Cyclone (ORCA backend cyclone API)
# ============================================================
#
# Deterministic tool for fetching active-cyclone data from the ORCA
# backend Cyclone API. No LLM reasoning happens here — it's a pure
# HTTP fetch + response-validation utility.
#
# Two entry points:
#   1. fetch_cyclone_data(lat, lon, radius_km) — plain function, always
#      returns a dict with a "success" flag, never raises for expected
#      failure modes. Used directly by cyclone_node.py.
#   2. cyclone_tool(latitude, longitude, radius_km) — LangChain @tool
#      wrapped version of the same call, for an LLM agent.
#
# IMPORTANT — "no active cyclone" vs. an actual error:
# The upstream API's "cyclone" field can legitimately be null/absent
# when there is simply no active tropical cyclone within radius_km —
# that is a normal, successful result, not a failure. It is only
# treated as an error (MISSING_FIELD) when the response as a whole
# does not indicate success (payload.get("success") is falsy) or is
# otherwise malformed. If your ORCA backend's actual contract differs
# (e.g. it never returns null and instead always includes a
# "no_active_cyclone" style object), adjust the branch below to match.

CYCLONE_API_BASE_URL = os.getenv(
    "CYCLONE_API_BASE_URL",
    "https://orca-backend-gkvo.onrender.com/api/cyclone",
)

DEFAULT_RADIUS_KM = 500

# timeout intentionally removed (per request) — REQUEST_TIMEOUT_SECONDS
# is kept only so the (now effectively unreachable) requests.Timeout
# except branch below still has a value to format into its message if
# a timeout is ever reintroduced.
REQUEST_TIMEOUT_SECONDS = 15


def fetch_cyclone_data(latitude: float, longitude: float, radius_km: int = DEFAULT_RADIUS_KM) -> dict:
    """
    Calls GET {CYCLONE_API_BASE_URL}?lat=..&lon=..&radius_km=..

    Returns:
        {
          "success": bool,
          "error_code": str | None,
          "error_message": str | None,
          "active": bool | None,   # True/False once success, None on failure
          "cyclone": dict | None,  # populated only when active is True
          "queried_at": str,       # ISO8601 UTC
          "query": {"lat": ..., "lon": ..., "radius_km": ...}
        }
    """
    params = {"lat": latitude, "lon": longitude, "radius_km": radius_km}

    try:
        # timeout=None: this call now waits indefinitely instead of
        # failing after REQUEST_TIMEOUT_SECONDS. The requests.Timeout
        # branch below can no longer actually fire — requests only
        # raises Timeout when a numeric timeout is set — but it is left
        # in place in case a timeout is reintroduced later.
        response = requests.get(CYCLONE_API_BASE_URL, params=params, timeout=None)
    except requests.exceptions.Timeout:
        return _cyclone_failure("TIMEOUT", f"Cyclone API did not respond within {REQUEST_TIMEOUT_SECONDS}s")
    except requests.exceptions.ConnectionError as exc:
        return _cyclone_failure("CONNECTION_ERROR", f"Could not reach Cyclone API: {exc}")
    except requests.exceptions.RequestException as exc:
        return _cyclone_failure("REQUEST_ERROR", str(exc))

    if response.status_code != 200:
        return _cyclone_failure(
            "HTTP_ERROR",
            f"Cyclone API returned HTTP {response.status_code}",
            raw_text=response.text[:500],
        )

    try:
        payload = response.json()
    except ValueError:
        return _cyclone_failure("INVALID_JSON", "Cyclone API response was not valid JSON")

    if not isinstance(payload, dict) or not payload.get("success", True):
        # Upstream explicitly reported failure (or gave us something
        # that isn't even a dict) — this is a real error, not "no
        # active cyclone".
        return _cyclone_failure(
            "UPSTREAM_FAILURE",
            "Cyclone API reported failure",
            raw_text=str(payload)[:500],
        )

    cyclone = payload.get("cyclone")

    return {
        "success": True,
        "error_code": None,
        "error_message": None,
        "active": cyclone is not None,
        "cyclone": cyclone,
        "queried_at": datetime.now(timezone.utc).isoformat(),
        "query": {"lat": latitude, "lon": longitude, "radius_km": radius_km},
    }


def _cyclone_failure(error_code: str, message: str, raw_text: str = None) -> dict:
    result = {
        "success": False,
        "error_code": error_code,
        "error_message": message,
        "active": None,
        "cyclone": None,
        "queried_at": datetime.now(timezone.utc).isoformat(),
    }
    if raw_text:
        result["raw_response"] = raw_text
    return result


@tool
def cyclone_tool(latitude: float, longitude: float, radius_km: int = DEFAULT_RADIUS_KM) -> str:
    """
    Check for active tropical cyclones within a given radius of a coastal
    location. Use this when the user asks about cyclones, tropical storms,
    or storm tracking near a location.

    Args:
        latitude: Latitude of the location to check, in decimal degrees.
        longitude: Longitude of the location to check, in decimal degrees.
        radius_km: Search radius in kilometers (default 500km).

    Returns:
        A JSON string with a consistent shape on every path:
            {
              "active": bool | None,   # True/False once the check
                                        # succeeded, None if the check
                                        # itself failed
              "status": "OK" | "NO_ACTIVE_CYCLONE" | "UNAVAILABLE",
              "cyclone": dict | None,  # name, distance, category, wind
                                       # speeds, pressure, movement,
                                       # warning level, forecast track —
                                       # populated only when active
              "error": str | None,    # populated only when unavailable
            }
    """
    result = fetch_cyclone_data(latitude, longitude, radius_km)
    if not result["success"]:
        return json.dumps({
            "active": None,
            "status": "UNAVAILABLE",
            "cyclone": None,
            "error": result["error_message"],
        })
    if not result["active"]:
        return json.dumps({
            "active": False,
            "status": "NO_ACTIVE_CYCLONE",
            "cyclone": None,
            "error": None,
        })
    return json.dumps({
        "active": True,
        "status": "OK",
        "cyclone": result["cyclone"],
        "error": None,
    })
