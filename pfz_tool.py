# pfz_tool.py

from datetime import datetime, timezone

import requests


PFZ_API_URL = "https://orca-backend-fz.onrender.com/api/pfz"


def get_pfz_data(latitude: float, longitude: float) -> dict:
    """
    Fetch PFZ (Potential Fishing Zone) data from the
    ORCA PFZ backend.

    No LLM is used.

    Never raises for expected failure modes (network error, bad HTTP
    status, non-JSON body). Every such case is converted into a
    structured failure dict instead, matching the pattern used by
    fetch_cyclone_data / fetch_gis_data:

        {
          "success": bool,
          "error_code": str | None,
          "error_message": str | None,
          "pfz": dict | None,
          "queried_at": str,   # ISO8601 UTC
          "query": {"latitude": ..., "longitude": ...}
        }

    On success, "pfz" holds the raw parsed JSON body from the PFZ
    backend (its shape is defined upstream, not by this file).
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
    }

    try:
        # timeout intentionally left as None (per request) — this call
        # will wait indefinitely for a response instead of failing after
        # 30s. NOTE: the PFZ backend drives a headless-Chrome/Puppeteer
        # scrape of INCOIS on every request and runs on a cold-starting
        # Render free tier, so it can legitimately take a long time —
        # but with no timeout at all, a genuinely stuck upstream request
        # will hang this call (and the whole graph run behind it)
        # forever. If that ever happens in practice, set an explicit
        # generous timeout (e.g. timeout=120) rather than leaving it
        # unbounded.
        response = requests.get(
            PFZ_API_URL,
            params=params,
            timeout=None,
        )
    except requests.exceptions.ConnectionError as exc:
        return _pfz_failure(latitude, longitude, "CONNECTION_ERROR",
                            f"Could not reach PFZ API: {exc}")
    except requests.exceptions.RequestException as exc:
        return _pfz_failure(latitude, longitude, "REQUEST_ERROR", str(exc))

    if response.status_code != 200:
        return _pfz_failure(
            latitude, longitude, "HTTP_ERROR",
            f"PFZ API returned HTTP {response.status_code}",
            raw_text=response.text[:500],
        )

    try:
        data = response.json()
    except ValueError:
        return _pfz_failure(latitude, longitude, "INVALID_JSON",
                            "PFZ API response was not valid JSON")

    return {
        "success": True,
        "error_code": None,
        "error_message": None,
        "pfz": data,
        "queried_at": datetime.now(timezone.utc).isoformat(),
        "query": {"latitude": latitude, "longitude": longitude},
    }


def _pfz_failure(latitude: float, longitude: float, error_code: str,
                 message: str, raw_text: str = None) -> dict:
    result = {
        "success": False,
        "error_code": error_code,
        "error_message": message,
        "pfz": None,
        "queried_at": datetime.now(timezone.utc).isoformat(),
        "query": {"latitude": latitude, "longitude": longitude},
    }
    if raw_text:
        result["raw_response"] = raw_text
    return result
