# gis_tool.py
#
# Deterministic tool for fetching coastal/marine GIS context from the
# ORCA backend GIS API. No LLM reasoning happens here — it's a pure
# HTTP fetch + response-validation utility, following the same pattern
# as fetch_cyclone_data / cyclone_tool in tools.py.
#
# Two entry points:
#   1. fetch_gis_data(latitude, longitude) — plain function, always
#      returns a dict with a "success" flag, never raises for expected
#      failure modes. Used directly by gis_node.py.
#   2. gis_tool(latitude, longitude) — LangChain @tool wrapped version
#      of the same call, for an LLM agent (e.g. chatbot follow-ups).

import json
import os
from datetime import datetime, timezone

import requests
from langchain_core.tools import tool


GIS_API_URL = os.getenv(
    "GIS_API_URL",
    "https://orca-backend-gis.onrender.com/api/gis",
)

# timeout intentionally removed (per request) — fetch_gis_data() below
# now waits indefinitely instead of failing after REQUEST_TIMEOUT_SECONDS.
REQUEST_TIMEOUT_SECONDS = 30


def fetch_gis_data(latitude: float, longitude: float) -> dict:
    """
    Calls GET {GIS_API_URL}?latitude=..&longitude=..

    Returns:
        {
          "success": bool,
          "error_code": str | None,
          "error_message": str | None,
          "gis": dict | None,
          "location": {"latitude": ..., "longitude": ...} | None,
          "queried_at": str,   # ISO8601 UTC
          "query": {"latitude": ..., "longitude": ...}
        }

    Never raises — every failure mode (timeout, connection error, bad
    HTTP status, invalid JSON, upstream success=false, missing "gis"
    field) is converted into a structured failure dict instead, so a
    calling node can just check result["success"].
    """
    params = {"latitude": latitude, "longitude": longitude}

    try:
        response = requests.get(GIS_API_URL, params=params, timeout=None)
    except requests.exceptions.Timeout:
        return _gis_failure(
            "TIMEOUT", f"GIS API did not respond within {REQUEST_TIMEOUT_SECONDS}s"
        )
    except requests.exceptions.ConnectionError as exc:
        return _gis_failure("CONNECTION_ERROR", f"Could not reach GIS API: {exc}")
    except requests.exceptions.RequestException as exc:
        return _gis_failure("REQUEST_ERROR", str(exc))

    if response.status_code != 200:
        return _gis_failure(
            "HTTP_ERROR",
            f"GIS API returned HTTP {response.status_code}",
            raw_text=response.text[:500],
        )

    try:
        payload = response.json()
    except ValueError:
        return _gis_failure("INVALID_JSON", "GIS API response was not valid JSON")

    if not payload.get("success", False):
        return _gis_failure(
            "UPSTREAM_FAILURE",
            "GIS API reported failure",
            raw_text=str(payload)[:500],
        )

    gis = payload.get("gis")
    if gis is None:
        return _gis_failure(
            "MISSING_FIELD",
            "Response did not contain a 'gis' object",
            raw_text=str(payload)[:500],
        )

    return {
        "success": True,
        "error_code": None,
        "error_message": None,
        "gis": gis,
        "location": payload.get("location", {"latitude": latitude, "longitude": longitude}),
        "queried_at": datetime.now(timezone.utc).isoformat(),
        "query": {"latitude": latitude, "longitude": longitude},
    }


def _gis_failure(error_code: str, message: str, raw_text: str = None) -> dict:
    result = {
        "success": False,
        "error_code": error_code,
        "error_message": message,
        "gis": None,
        "location": None,
        "queried_at": datetime.now(timezone.utc).isoformat(),
    }
    if raw_text:
        result["raw_response"] = raw_text
    return result


@tool
def gis_tool(latitude: float, longitude: float) -> str:
    """
    Look up coastal/marine GIS context for a location: distance to
    coast, water depth, nearest restricted zone, whether the point is
    inside a marine protected area (and which one), nearest maritime
    boundary, nearest port, and which country's EEZ / fishing-zone
    jurisdiction the point falls in.

    Use this for questions like "how far offshore is this point",
    "am I inside a restricted or protected zone", "whose waters is
    this", or "what's the nearest port".

    Args:
        latitude: Latitude of the point to check, in decimal degrees.
        longitude: Longitude of the point to check, in decimal degrees.

    Returns a JSON string with the GIS fields. If the API call fails,
    returns a JSON string with "success": false and an error message
    instead of raising or fabricating data.
    """
    result = fetch_gis_data(latitude, longitude)
    if not result["success"]:
        return json.dumps({
            "success": False,
            "error": result["error_message"],
        })
    return json.dumps(result["gis"], default=str)
