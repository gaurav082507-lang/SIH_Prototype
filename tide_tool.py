"""
ORCA — Tide Tool
================
Path: ai-service/tools/tide_tool.py

SIH26176 — Marine EcOsystem Reasoning with Collaborative Agents

WHAT THIS FILE IS
-----------------
The *tool* layer for tide data. It is the only place in ORCA that knows:

  1. the tide service URL and its query contract,
  2. how INCOIS PAT timestamps are formatted ("DD-MM-YYYY HH:mm", IST),
  3. how to turn a list of high/low tide EVENTS into a usable tide STATE
     (height now, phase, next high, next low, tidal range).

It has NO LangGraph imports and NO ORCA state knowledge. That belongs in
`graph/nodes/tide_node.py`. This file is safe to import from:

  - tide_node.py            (pipeline: Planner -> Ocean/Tide -> Risk -> Decision)
  - the Chatbot Agent       (follow-up questions, via `tide_tool`)
  - any test script

UPSTREAM SERVICE
----------------
    GET https://orca-backend-tide.onrender.com/tide
        ?lat=19.076&lon=72.8777&fromDate=2026-08-29&toDate=2026-11-30

    -> { success, date_range, locations_count, data: [ { latitude, longitude,
         station{name,latitude,longitude,distance_km}, date_range,
         tides{ current_tide_height_m, tide_phase, high_tide[], low_tide[],
                tidal_current_velocity_ms, tidal_current_direction_deg },
         source } ] }

IMPORTANT — WHAT THE SERVICE DOES *NOT* GIVE US
-----------------------------------------------
The upstream API returns null for:

    current_tide_height_m
    tide_phase
    tidal_current_velocity_ms
    tidal_current_direction_deg

`current_tide_height_m` and `tide_phase` are DERIVED here from the predicted
extrema using the standard "rule of twelfths" cosine interpolation, and are
labelled with source "INCOIS PAT (derived by ORCA)" so nobody downstream
mistakes them for a measurement.

Tidal CURRENTS cannot be derived from tide heights. They stay NOT_MAPPED,
matching the known-gaps list in the I/O contract. NULL != ZERO.

TIMEZONE
--------
INCOIS PAT publishes tide times in Indian Standard Time. Every datetime this
module produces is timezone-aware IST, and every emitted string carries the
offset (e.g. "2026-09-03T15:11:00+05:30"), so no consumer has to guess.

A naive (tzinfo-less) datetime passed in anywhere in this module (e.g. via
`reference_times`) is always treated as "this clock time, in IST" — it is
never passed through Python's `.astimezone()`, which would instead
reinterpret it as local server time and silently convert it. See
`_coerce_ist()` below; every caller of a naive/aware datetime goes through
it so behaviour can't diverge between call sites.

Dependencies: requests (stdlib otherwise).
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from datetime import date as _date
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TIDE_API_URL = os.getenv("TIDE_API_URL", "https://orca-backend-tide.onrender.com/tide")

# timeout intentionally removed (per request) — this call now waits
# indefinitely for a response instead of failing after TIDE_TIMEOUT_S.
# TIDE_TIMEOUT_S / TIDE_RETRIES / TIDE_BACKOFF_S are kept (and the retry
# loop below still runs) in case a timeout is reintroduced later, but
# with no timeout set, requests.Timeout can never actually be raised, so
# every attempt effectively has unlimited time to respond before the
# retry loop would move on.
TIDE_TIMEOUT_S = float(os.getenv("TIDE_TIMEOUT_S", "50"))
TIDE_RETRIES = int(os.getenv("TIDE_RETRIES", "2"))          # attempts after the first
TIDE_BACKOFF_S = float(os.getenv("TIDE_BACKOFF_S", "2.0"))

# Tide predictions are astronomical — they do not change hour to hour.
# A long cache is safe and saves the grid from hammering INCOIS.
TIDE_CACHE_TTL_S = int(os.getenv("TIDE_CACHE_TTL_S", str(6 * 3600)))

SOURCE_NAME = "INCOIS PAT"
SOURCE_DERIVED = "INCOIS PAT (derived by ORCA)"

IST = timezone(timedelta(hours=5, minutes=30))

# INCOIS timestamp: "29-08-2026 00:21"
_TS_RE = re.compile(r"^\s*(\d{2})-(\d{2})-(\d{4})\s+(\d{2}):(\d{2})\s*$")

# A tide is treated as "slack" (turning) within this many minutes of an extremum.
SLACK_WINDOW_MIN = 30

# Station further than this from the requested point gets a warning attached.
STATION_DISTANCE_WARN_KM = float(os.getenv("TIDE_STATION_WARN_KM", "60"))


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class TideError(Exception):
    """Categorised tide failure, so the node can build a proper errors[] entry."""

    def __init__(self, category: str, message: str, detail: Optional[str] = None):
        super().__init__(message)
        self.category = category      # see categories below
        self.message = message
        self.detail = detail

    # Categories:
    #   INVALID_INPUT      bad lat/lon/date before we ever call out
    #   TIMEOUT            upstream did not answer in time
    #   NETWORK            DNS/connection/TLS failure
    #   UPSTREAM_HTTP      non-2xx from the tide service
    #   UPSTREAM_PAYLOAD   2xx but success=false or unusable body
    #   NO_DATA            valid response, zero tide events
    #   PARSE              timestamps/heights could not be read

    def as_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "message": self.message,
            "detail": self.detail,
            "source": SOURCE_NAME,
        }


# ---------------------------------------------------------------------------
# Shared ORCA shapes
# ---------------------------------------------------------------------------

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def data_field(
    value: Any,
    unit: Optional[str],
    *,
    source: str = SOURCE_NAME,
    status: str = "OK",
    freshness: str = "LIVE",
    retrieved_at: Optional[str] = None,
) -> Dict[str, Any]:
    """The ORCA DataField wrapper.

    status:    OK | UNAVAILABLE | STALE | NOT_MAPPED
    freshness: LIVE | CACHED | NRT | NOT_APPLICABLE
    """
    if value is None and status == "OK":
        status = "UNAVAILABLE"
    return {
        "value": value,
        "unit": unit,
        "source": source,
        "retrieved_at": retrieved_at or now_utc_iso(),
        "status": status,
        "freshness": freshness,
    }


def not_mapped_field(unit: Optional[str], reason: str) -> Dict[str, Any]:
    f = data_field(None, unit, status="NOT_MAPPED", freshness="NOT_APPLICABLE")
    f["reason"] = reason
    return f


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_coords(lat: Any, lon: Any) -> Tuple[float, float]:
    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except (TypeError, ValueError):
        raise TideError("INVALID_INPUT", "Latitude/longitude are not numeric",
                        f"lat={lat!r} lon={lon!r}")
    if not (-90.0 <= lat_f <= 90.0):
        raise TideError("INVALID_INPUT", "Latitude out of range", f"lat={lat_f}")
    if not (-180.0 <= lon_f <= 180.0):
        raise TideError("INVALID_INPUT", "Longitude out of range", f"lon={lon_f}")
    return lat_f, lon_f


def _validate_date(value: Any, label: str) -> str:
    if isinstance(value, _date) and not isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.date().isoformat()
    if not isinstance(value, str):
        raise TideError("INVALID_INPUT", f"{label} must be YYYY-MM-DD", repr(value))
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d").date().isoformat()
    except ValueError:
        raise TideError("INVALID_INPUT", f"{label} must be YYYY-MM-DD", value)


def parse_ist(ts: str) -> Optional[datetime]:
    """'29-08-2026 00:21' -> aware datetime in IST. None if unparseable."""
    m = _TS_RE.match(ts or "")
    if not m:
        return None
    dd, mm, yyyy, hh, mi = (int(g) for g in m.groups())
    try:
        return datetime(yyyy, mm, dd, hh, mi, tzinfo=IST)
    except ValueError:
        return None


def to_iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat(timespec="seconds") if dt else None


def _coerce_ist(dt: datetime) -> datetime:
    """Normalize any datetime (naive or aware) to aware IST, consistently.

    A naive datetime is treated as "this clock time, already in IST" — it
    is labelled with `.replace(tzinfo=IST)`, never converted with
    `.astimezone()` (which would instead reinterpret it as local server
    time and shift the clock value). An aware datetime in some other zone
    is properly converted to IST with `.astimezone(IST)`.

    This is the single place that decides how naive datetimes are
    interpreted, so every caller (timeline construction, the anchor
    state, etc.) agrees on the same instant for the same input.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=IST)
    return dt.astimezone(IST)


# ---------------------------------------------------------------------------
# Cache (process-local, thread-safe)
# ---------------------------------------------------------------------------

_cache: Dict[Tuple, Tuple[float, Dict[str, Any]]] = {}
_cache_lock = threading.Lock()


def _cache_key(lat: float, lon: float, from_date: str, to_date: str) -> Tuple:
    # ~1 km rounding. Grid points 5 km apart still resolve to the same PAT
    # station in almost every case, so this collapses most of the 9-point fan-out.
    return (round(lat, 2), round(lon, 2), from_date, to_date)


def _cache_get(key: Tuple) -> Optional[Dict[str, Any]]:
    with _cache_lock:
        hit = _cache.get(key)
        if not hit:
            return None
        stored_at, payload = hit
        if time.time() - stored_at > TIDE_CACHE_TTL_S:
            _cache.pop(key, None)
            return None
        return payload


def _cache_put(key: Tuple, payload: Dict[str, Any]) -> None:
    with _cache_lock:
        _cache[key] = (time.time(), payload)


def clear_tide_cache() -> None:
    with _cache_lock:
        _cache.clear()


# ---------------------------------------------------------------------------
# Layer 1 — raw HTTP
# ---------------------------------------------------------------------------

def fetch_tide_raw(
    lat: float,
    lon: float,
    from_date: str,
    to_date: str,
    *,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Call the ORCA tide service. Returns the raw JSON body plus `_orca_cached`.

    Raises TideError on every failure path.
    """
    lat, lon = _validate_coords(lat, lon)
    from_date = _validate_date(from_date, "fromDate")
    to_date = _validate_date(to_date, "toDate")
    if from_date > to_date:
        raise TideError("INVALID_INPUT", "fromDate is after toDate",
                        f"{from_date} > {to_date}")

    key = _cache_key(lat, lon, from_date, to_date)
    if use_cache:
        cached = _cache_get(key)
        if cached is not None:
            out = dict(cached)
            out["_orca_cached"] = True
            return out

    params = {"lat": lat, "lon": lon, "fromDate": from_date, "toDate": to_date}
    last_exc: Optional[TideError] = None

    for attempt in range(TIDE_RETRIES + 1):
        try:
            # timeout=None: see the module-level note near TIDE_TIMEOUT_S —
            # this call now waits indefinitely instead of giving up after
            # TIDE_TIMEOUT_S seconds.
            resp = requests.get(TIDE_API_URL, params=params, timeout=None)
        except requests.Timeout as e:
            last_exc = TideError("TIMEOUT",
                                 f"Tide service timed out after {TIDE_TIMEOUT_S}s",
                                 str(e))
        except requests.RequestException as e:
            last_exc = TideError("NETWORK", "Could not reach the tide service", str(e))
        else:
            if resp.status_code >= 500:
                last_exc = TideError("UPSTREAM_HTTP",
                                     f"Tide service returned {resp.status_code}",
                                     resp.text[:300])
            elif resp.status_code >= 400:
                # 4xx is our fault — retrying will not help.
                raise TideError("UPSTREAM_HTTP",
                                f"Tide service rejected the request ({resp.status_code})",
                                resp.text[:300])
            else:
                try:
                    body = resp.json()
                except ValueError as e:
                    raise TideError("UPSTREAM_PAYLOAD",
                                    "Tide service did not return JSON", str(e))
                if not isinstance(body, dict) or not body.get("success"):
                    raise TideError("UPSTREAM_PAYLOAD",
                                    "Tide service reported failure",
                                    json.dumps(body)[:300])
                if use_cache:
                    _cache_put(key, body)
                out = dict(body)
                out["_orca_cached"] = False
                return out

        if attempt < TIDE_RETRIES:
            time.sleep(TIDE_BACKOFF_S * (2 ** attempt))

    raise last_exc or TideError("NETWORK", "Tide service unreachable")


# ---------------------------------------------------------------------------
# Layer 2 — events
# ---------------------------------------------------------------------------

def build_events(tides: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Merge high_tide[] + low_tide[] into one chronological event list.

    Event: {"type": "HIGH"|"LOW", "dt": aware datetime IST,
            "time": ISO string, "height_m": float}
    Returns (events, warnings).
    """
    events: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for kind, key in (("HIGH", "high_tide"), ("LOW", "low_tide")):
        for raw in (tides or {}).get(key) or []:
            dt = parse_ist(raw.get("time", ""))
            height = raw.get("height_m")
            if dt is None:
                warnings.append(f"Unparseable {kind} tide timestamp: {raw.get('time')!r}")
                continue
            try:
                height = float(height)
            except (TypeError, ValueError):
                warnings.append(f"Unparseable {kind} tide height at {raw.get('time')}")
                continue
            if not math.isfinite(height):
                warnings.append(f"Non-finite {kind} tide height at {raw.get('time')}")
                continue
            events.append({"type": kind, "dt": dt, "time": to_iso(dt), "height_m": height})

    events.sort(key=lambda e: e["dt"])

    # Drop exact duplicates (the upstream parser already does this, but the
    # merge of two lists can still collide on identical rows).
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for e in events:
        sig = (e["type"], e["time"], e["height_m"])
        if sig in seen:
            continue
        seen.add(sig)
        deduped.append(e)

    return deduped, warnings


def _public_event(e: Dict[str, Any]) -> Dict[str, Any]:
    """Event without the internal datetime object, safe to put in JSON."""
    return {"type": e["type"], "time": e["time"], "height_m": round(e["height_m"], 3)}


# ---------------------------------------------------------------------------
# Layer 3 — derived tide state
# ---------------------------------------------------------------------------

def interpolate_height(events: Sequence[Dict[str, Any]], at: datetime) -> Optional[float]:
    """Water level at `at`, via cosine ("rule of twelfths") interpolation.

    Returns None outside the bracketing extrema — ORCA never extrapolates tide.
    """
    if len(events) < 2:
        return None
    if at < events[0]["dt"] or at > events[-1]["dt"]:
        return None

    for i in range(len(events) - 1):
        a, b = events[i], events[i + 1]
        if a["dt"] <= at <= b["dt"]:
            span = (b["dt"] - a["dt"]).total_seconds()
            if span <= 0:
                return a["height_m"]
            frac = (at - a["dt"]).total_seconds() / span
            h = a["height_m"] + (b["height_m"] - a["height_m"]) * (1 - math.cos(math.pi * frac)) / 2
            return round(h, 3)
    return None


def tide_phase_at(events: Sequence[Dict[str, Any]], at: datetime) -> Optional[str]:
    """RISING | FALLING | HIGH_SLACK | LOW_SLACK, or None if out of range."""
    if len(events) < 2:
        return None
    if at < events[0]["dt"] or at > events[-1]["dt"]:
        return None

    for i in range(len(events) - 1):
        a, b = events[i], events[i + 1]
        if a["dt"] <= at <= b["dt"]:
            if abs((at - a["dt"]).total_seconds()) <= SLACK_WINDOW_MIN * 60:
                return "HIGH_SLACK" if a["type"] == "HIGH" else "LOW_SLACK"
            if abs((b["dt"] - at).total_seconds()) <= SLACK_WINDOW_MIN * 60:
                return "HIGH_SLACK" if b["type"] == "HIGH" else "LOW_SLACK"
            return "RISING" if b["height_m"] > a["height_m"] else "FALLING"
    return None


def next_event(events: Sequence[Dict[str, Any]], at: datetime,
               kind: Optional[str] = None) -> Optional[Dict[str, Any]]:
    for e in events:
        if e["dt"] >= at and (kind is None or e["type"] == kind):
            return e
    return None


def previous_event(events: Sequence[Dict[str, Any]], at: datetime,
                   kind: Optional[str] = None) -> Optional[Dict[str, Any]]:
    found = None
    for e in events:
        if e["dt"] <= at and (kind is None or e["type"] == kind):
            found = e
        elif e["dt"] > at:
            break
    return found


def tide_state_at(events: Sequence[Dict[str, Any]], at: datetime) -> Dict[str, Any]:
    """Everything ORCA wants to know about the tide at one instant.

    All values here are DERIVED from predicted extrema, never measured.
    """
    height = interpolate_height(events, at)
    phase = tide_phase_at(events, at)
    nh = next_event(events, at, "HIGH")
    nl = next_event(events, at, "LOW")
    ph = previous_event(events, at, "HIGH")
    pl = previous_event(events, at, "LOW")

    def _mins(e):
        return round((e["dt"] - at).total_seconds() / 60) if e else None

    return {
        "time": to_iso(at),
        "tide_height_m": height,
        "tide_phase": phase,
        "next_high_tide": _public_event(nh) if nh else None,
        "minutes_to_next_high": _mins(nh),
        "next_low_tide": _public_event(nl) if nl else None,
        "minutes_to_next_low": _mins(nl),
        "previous_high_tide": _public_event(ph) if ph else None,
        "previous_low_tide": _public_event(pl) if pl else None,
        "in_prediction_range": height is not None,
    }


def daily_range(events: Sequence[Dict[str, Any]], day: str) -> Dict[str, Any]:
    """Tidal range for one calendar day (IST). `day` is 'YYYY-MM-DD'."""
    same_day = [e for e in events if e["time"][:10] == day]
    if not same_day:
        return {"date": day, "max_height_m": None, "min_height_m": None,
                "tidal_range_m": None, "event_count": 0}
    highs = [e["height_m"] for e in same_day if e["type"] == "HIGH"]
    lows = [e["height_m"] for e in same_day if e["type"] == "LOW"]
    hi = max(highs) if highs else None
    lo = min(lows) if lows else None
    return {
        "date": day,
        "max_height_m": round(hi, 3) if hi is not None else None,
        "min_height_m": round(lo, 3) if lo is not None else None,
        "tidal_range_m": round(hi - lo, 3) if (hi is not None and lo is not None) else None,
        "event_count": len(same_day),
    }


def spring_neap_hint(events: Sequence[Dict[str, Any]], day: str) -> Optional[str]:
    """Heuristic label only — SPRING_LIKE / NEAP_LIKE / AVERAGE.

    Compares the day's range against the ranges across the whole fetched window.
    Not an astronomical calculation; label it as advisory wherever it is shown.
    """
    days = sorted({e["time"][:10] for e in events})
    ranges = []
    for d in days:
        r = daily_range(events, d)["tidal_range_m"]
        if r is not None:
            ranges.append(r)
    if len(ranges) < 4:
        return None
    today = daily_range(events, day)["tidal_range_m"]
    if today is None:
        return None
    lo, hi = min(ranges), max(ranges)
    if hi - lo < 1e-6:
        return "AVERAGE"
    pos = (today - lo) / (hi - lo)
    if pos >= 0.75:
        return "SPRING_LIKE"
    if pos <= 0.25:
        return "NEAP_LIKE"
    return "AVERAGE"


# ---------------------------------------------------------------------------
# Layer 4 — the ORCA-shaped tide product
# ---------------------------------------------------------------------------

def get_tide(
    lat: float,
    lon: float,
    from_date: str,
    to_date: str,
    *,
    reference_times: Optional[Sequence[datetime]] = None,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Fetch + normalize + derive. This is what tide_node.py calls.

    `reference_times` are datetimes (aware or naive) at which a full tide
    state is computed — typically the hourly timestamps of the requested
    day-part. Naive datetimes are treated as IST clock times (see
    `_coerce_ist`); aware datetimes in another zone are converted to IST.
    This interpretation is applied consistently to every reference time,
    including the one used to build the anchor state.

    Returns a dict with `station`, `events`, `timeline`, `fields`, `warnings`.
    Raises TideError on unrecoverable failure.
    """
    body = fetch_tide_raw(lat, lon, from_date, to_date, use_cache=use_cache)
    cached = bool(body.get("_orca_cached"))
    retrieved_at = now_utc_iso()
    freshness = "CACHED" if cached else "LIVE"

    locations = body.get("data") or []
    if not locations:
        raise TideError("NO_DATA", "Tide service returned no locations")

    loc = locations[0]
    tides = loc.get("tides") or {}
    station = loc.get("station") or {}
    source = loc.get("source") or SOURCE_NAME

    events, warnings = build_events(tides)
    if not events:
        raise TideError("NO_DATA", "No tide events for the requested range",
                        f"{from_date}..{to_date}")

    dist = station.get("distance_km")
    if isinstance(dist, (int, float)) and dist > STATION_DISTANCE_WARN_KM:
        warnings.append(
            f"Nearest PAT station '{station.get('name')}' is {dist} km away; "
            "tide times at the requested point may differ noticeably."
        )

    # Normalize every reference time through the same rule (naive -> IST
    # label, aware -> converted to IST) so the timeline and the anchor
    # state can never disagree about what instant a given input means.
    ref_ist_list = [_coerce_ist(ref) for ref in (reference_times or [])]

    timeline: List[Dict[str, Any]] = [tide_state_at(events, ref_ist) for ref_ist in ref_ist_list]

    # Anchor state: first reference time (already normalized above), else
    # the start of the requested range.
    anchor = (ref_ist_list[0] if ref_ist_list
              else _coerce_ist(datetime.fromisoformat(from_date)))
    anchor_state = tide_state_at(events, anchor)
    day_stats = daily_range(events, anchor.date().isoformat())

    derived_kw = dict(source=SOURCE_DERIVED, freshness=freshness, retrieved_at=retrieved_at)
    observed_kw = dict(source=source, freshness=freshness, retrieved_at=retrieved_at)

    fields = {
        # From the service
        "high_tide_events": data_field(
            [_public_event(e) for e in events if e["type"] == "HIGH"], "list", **observed_kw),
        "low_tide_events": data_field(
            [_public_event(e) for e in events if e["type"] == "LOW"], "list", **observed_kw),

        # Derived by ORCA from the predicted extrema
        "tide_height_m": data_field(anchor_state["tide_height_m"], "m", **derived_kw),
        "tide_phase": data_field(anchor_state["tide_phase"], "category", **derived_kw),
        "next_high_tide": data_field(anchor_state["next_high_tide"], "event", **derived_kw),
        "next_low_tide": data_field(anchor_state["next_low_tide"], "event", **derived_kw),
        "tidal_range_m": data_field(day_stats["tidal_range_m"], "m", **derived_kw),
        "spring_neap_hint": data_field(
            spring_neap_hint(events, anchor.date().isoformat()), "category", **derived_kw),

        # Known gaps — see the I/O contract. NULL != ZERO.
        "tidal_current_velocity_ms": not_mapped_field(
            "m/s", "No tidal-current source is wired into ORCA yet; "
                   "it cannot be derived from tide heights."),
        "tidal_current_direction_deg": not_mapped_field(
            "deg", "No tidal-current source is wired into ORCA yet; "
                   "it cannot be derived from tide heights."),
    }

    return {
        "requested": {"lat": lat, "lon": lon, "from_date": from_date, "to_date": to_date},
        "station": {
            "name": station.get("name"),
            "latitude": station.get("latitude"),
            "longitude": station.get("longitude"),
            "distance_km": station.get("distance_km"),
        },
        "source": source,
        "cached": cached,
        "retrieved_at": retrieved_at,
        "timezone": "Asia/Kolkata (+05:30)",
        "event_count": len(events),
        "events": [_public_event(e) for e in events],
        "_events_internal": events,          # datetimes kept for the node's own maths
        "anchor_state": anchor_state,
        "day_stats": day_stats,
        "timeline": timeline,
        "fields": fields,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Layer 5 — Chatbot Agent tool
# ---------------------------------------------------------------------------

def tide_lookup(
    lat: float,
    lon: float,
    date: Optional[str] = None,
    time_hhmm: Optional[str] = None,
) -> Dict[str, Any]:
    """Compact, LLM-friendly tide answer for one point and one moment.

    date:      'YYYY-MM-DD' (default: today, IST)
    time_hhmm: 'HH:MM' IST  (default: now if date is today, else 06:00)
    """
    try:
        lat, lon = _validate_coords(lat, lon)
        target_day = _validate_date(date, "date") if date else datetime.now(IST).date().isoformat()

        if time_hhmm:
            try:
                hh, mm = (int(x) for x in str(time_hhmm).strip().split(":"))
                ref = datetime.fromisoformat(target_day).replace(hour=hh, minute=mm, tzinfo=IST)
            except (ValueError, TypeError):
                raise TideError("INVALID_INPUT", "time must be HH:MM", str(time_hhmm))
        elif target_day == datetime.now(IST).date().isoformat():
            ref = datetime.now(IST).replace(second=0, microsecond=0)
        else:
            ref = datetime.fromisoformat(target_day).replace(hour=6, tzinfo=IST)

        # One day either side so interpolation always has bracketing extrema.
        d = datetime.fromisoformat(target_day).date()
        result = get_tide(
            lat, lon,
            (d - timedelta(days=1)).isoformat(),
            (d + timedelta(days=1)).isoformat(),
            reference_times=[ref],
        )
        state = result["timeline"][0]
        day = result["day_stats"]

        return {
            "ok": True,
            "location": {"lat": lat, "lon": lon},
            "station": result["station"],
            "at": state["time"],
            "tide_height_m": state["tide_height_m"],
            "tide_phase": state["tide_phase"],
            "next_high_tide": state["next_high_tide"],
            "next_low_tide": state["next_low_tide"],
            "tidal_range_today_m": day["tidal_range_m"],
            "high_tides_today": [e for e in result["events"]
                                 if e["type"] == "HIGH" and e["time"][:10] == target_day],
            "low_tides_today": [e for e in result["events"]
                                if e["type"] == "LOW" and e["time"][:10] == target_day],
            "tidal_currents": "NOT_MAPPED — ORCA has no tidal-current data source yet",
            "source": result["source"],
            "note": "Heights and phase are interpolated from INCOIS PAT predicted "
                    "high/low tides, not measured water levels. Times are IST.",
            "warnings": result["warnings"],
        }
    except TideError as e:
        return {"ok": False, "error": e.as_dict()}
    except Exception as e:  # never let the chatbot crash on a tool call
        return {"ok": False, "error": {"category": "UNEXPECTED", "message": str(e)}}


def tide_tool(lat: float, lon: float, date: str = "", time: str = "") -> str:
    """Get tide conditions at a coastal point in India.

    Use for questions about tide height, high/low tide timings, whether the
    tide is rising or falling, or tidal range. Times and dates are IST.

    Args:
        lat: latitude, decimal degrees
        lon: longitude, decimal degrees
        date: 'YYYY-MM-DD'; empty means today
        time: 'HH:MM' IST; empty means now (or 06:00 for a future date)

    Returns a JSON string. Tidal currents are not available in ORCA.
    """
    return json.dumps(
        tide_lookup(lat, lon, date or None, time or None),
        ensure_ascii=False,
        default=str,
    )


# Register with LangChain only if it is installed, so this file stays importable
# from plain scripts and from the Flask layer.
try:  # pragma: no cover
    from langchain_core.tools import tool as _lc_tool

    tide_tool_lc = _lc_tool(tide_tool)
except Exception:  # pragma: no cover
    tide_tool_lc = None


# ---------------------------------------------------------------------------
# Smoke test:  python tide_tool.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Tide service: {TIDE_API_URL}\n")

    # Mumbai, the same point as the sample response.
    out = tide_lookup(19.076, 72.8777, "2026-09-03", "15:00")
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))

    if out.get("ok"):
        print("\nSecond call (should be a cache hit, near-instant):")
        t0 = time.time()
        tide_lookup(19.076, 72.8777, "2026-09-03", "18:00")
        print(f"  {time.time() - t0:.3f}s")
