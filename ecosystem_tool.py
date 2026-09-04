from datetime import datetime, timezone

import requests
from langchain_core.tools import tool


ECOSYSTEM_API_URL = (
    "https://orca-backend-ecosystem.onrender.com/api/ecosystem"
)


def fetch_ecosystem_data(latitude: float, longitude: float, date: str) -> dict:
    """
    Plain (non-tool) fetch, safe to call directly with keyword arguments
    and safe to call in a loop (get_ecosystem_data_for_grid below) —
    never raises for expected failure modes (network error, bad HTTP
    status, non-JSON body). Every such case is converted into a
    structured failure dict instead, matching the pattern used by
    fetch_cyclone_data / fetch_gis_data / get_pfz_data:

        {
          "success": bool,
          "error_code": str | None,
          "error_message": str | None,
          "ecosystem": dict | None,
          "queried_at": str,   # ISO8601 UTC
          "query": {"latitude": ..., "longitude": ..., "date": ...}
        }
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "date": date,
    }

    try:
        # timeout intentionally left as None (per request) — see
        # pfz_tool.py for the trade-off: this call now waits
        # indefinitely instead of failing after 60s if the ecosystem
        # backend hangs.
        response = requests.get(
            ECOSYSTEM_API_URL,
            params=params,
            timeout=None,
        )
    except requests.exceptions.ConnectionError as exc:
        return _ecosystem_failure(latitude, longitude, date, "CONNECTION_ERROR",
                                  f"Could not reach Ecosystem API: {exc}")
    except requests.exceptions.RequestException as exc:
        return _ecosystem_failure(latitude, longitude, date, "REQUEST_ERROR", str(exc))

    if response.status_code != 200:
        return _ecosystem_failure(
            latitude, longitude, date, "HTTP_ERROR",
            f"Ecosystem API returned HTTP {response.status_code}",
            raw_text=response.text[:500],
        )

    try:
        data = response.json()
    except ValueError:
        return _ecosystem_failure(latitude, longitude, date, "INVALID_JSON",
                                  "Ecosystem API response was not valid JSON")

    return {
        "success": True,
        "error_code": None,
        "error_message": None,
        "ecosystem": data,
        "queried_at": datetime.now(timezone.utc).isoformat(),
        "query": {"latitude": latitude, "longitude": longitude, "date": date},
    }


def _ecosystem_failure(latitude: float, longitude: float, date: str,
                       error_code: str, message: str, raw_text: str = None) -> dict:
    result = {
        "success": False,
        "error_code": error_code,
        "error_message": message,
        "ecosystem": None,
        "queried_at": datetime.now(timezone.utc).isoformat(),
        "query": {"latitude": latitude, "longitude": longitude, "date": date},
    }
    if raw_text:
        result["raw_response"] = raw_text
    return result


@tool
def get_ecosystem_data(
    latitude: float,
    longitude: float,
    date: str
) -> dict:
    """
    Fetch marine ecosystem data for a latitude,
    longitude and date.

    Returns a structured dict with a "success" flag rather than raising
    or fabricating data on failure. See fetch_ecosystem_data for the
    exact shape.
    """
    return fetch_ecosystem_data(latitude, longitude, date)


def get_ecosystem_data_for_grid(
    locations: list,
    date: str
) -> list:
    """
    Fetch ecosystem data for all planner grid points.

    locations format:

    [
        {
            "id": "P1",
            "latitude": 19.1661,
            "longitude": 72.8777,
            "bearing_deg": 0,
            "distance_km": 10
        },
        ...
    ]

    Each point is fetched independently via fetch_ecosystem_data, which
    never raises — so one failing/slow grid point produces a
    "success": False entry for that point instead of aborting the
    whole grid or crashing the caller. The caller should check each
    result's "success" flag rather than assume every point returned
    live data.

    NOTE: this still issues one HTTP request per grid point (unlike
    get_weather_for_grid / get_ocean_data_for_grid in tools.py, which
    batch every point into a single Open-Meteo call) — the ecosystem
    backend has no batch endpoint to call instead.
    """

    results = []

    for location in locations:

        latitude = location["latitude"]
        longitude = location["longitude"]

        data = fetch_ecosystem_data(latitude, longitude, date)

        # Keep planner grid-point information regardless of success/failure
        data["id"] = location.get("id")
        data["bearing_deg"] = location.get("bearing_deg")
        data["distance_km"] = location.get("distance_km")

        results.append(data)

    return results
