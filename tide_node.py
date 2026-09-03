"""
ORCA — Tide Node
================
Path: ai-service/graph/nodes/tide_node.py

SIH26176 — Marine EcOsystem Reasoning with Collaborative Agents

WHAT THIS FILE IS
-----------------
The LangGraph node that turns the Planner Agent's search grid + date + day-part
into a tide `AgentEnvelope` for the Risk Agent.

    Planner Agent  ──▶  [ Weather ]  [ Ocean ]  [ Ecosystem ]  ──▶  Risk  ──▶  Decision
                                        │
                                    tide_node   ◀── you are here
                                        │
                                   tide_tool.py
                                        │
                        https://orca-backend-tide.onrender.com/tide

It reads state, calls `tools/tide_tool.py`, computes per-point and per-hour
tide state, raises safety flags, and writes ONE envelope back into state under
`state["tide"]`.

WHY IT DOESN'T CALL THE API NINE TIMES
--------------------------------------
The tide service resolves the nearest INCOIS PAT station by Haversine distance.
The 9 grid points sit 2–10 km apart, and PAT stations are tens to hundreds of km
apart — so all 9 points almost always resolve to the SAME station and therefore
the SAME tide curve. Default mode is `center`: one call, result shared across all
points with `station_shared: true` so nothing downstream pretends the points
have independent tide data.

Set `TIDE_GRID_MODE=per_point` to fan out anyway (the tool's cache still
collapses duplicate station lookups).

TIMEZONE
--------
INCOIS PAT tide times are IST. The day-part windows here are interpreted as
IST clock hours, because that is what a fisherman means by "morning". If your
Planner emits UTC-labelled windows, flip DAY_PART_TZ to UTC below — it is the
one place this assumption lives.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

# Import path assumes `ai-service/` is the package root.
try:
    from tide_tool import (
        IST,
        TideError,
        data_field,
        get_tide,
        not_mapped_field,
        now_utc_iso,
        tide_state_at,
    )
except ImportError:  # running the file directly, tool sitting alongside it
    from tide_tool import (  # type: ignore
        IST,
        TideError,
        data_field,
        get_tide,
        not_mapped_field,
        now_utc_iso,
        tide_state_at,
    )

AGENT_NAME = "tide"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TIDE_GRID_MODE = os.getenv("TIDE_GRID_MODE", "center")     # center | per_point
DAY_PART_TZ = IST                                          # see module docstring

# Same windows the Planner and Weather agents use.
DAY_PART_WINDOWS: Dict[str, Tuple[int, int]] = {
    "morning": (6, 11),
    "afternoon": (12, 17),
    "evening": (17, 20),
    "night": (20, 23),
    "full_day": (0, 23),
}

# Thresholds for tide flags. Placeholders — tune with domain input, they are
# deliberately in one block so the Risk Agent's story stays auditable.
LARGE_RANGE_M = float(os.getenv("TIDE_LARGE_RANGE_M", "3.0"))     # strong tidal streams likely
VERY_LOW_WATER_M = float(os.getenv("TIDE_VERY_LOW_WATER_M", "0.5"))  # grounding / bar risk
SHALLOW_DRAFT_ACTIVITIES = {"fishing", "boating"}
TIDE_SENSITIVE_ACTIVITIES = {"fishing", "boating", "diving", "surfing", "marine_research"}


# ---------------------------------------------------------------------------
# State readers — tolerant of both flat and nested planner output
# ---------------------------------------------------------------------------

def _dig(state: Dict[str, Any], *paths: str) -> Any:
    """Return the first non-empty value found at any dotted path."""
    for path in paths:
        cur: Any = state
        ok = True
        for part in path.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok and cur not in (None, "", [], {}):
            return cur
    return None


def _extract_grid(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    pts = _dig(
        state,
        "search_grid.points",
        "planner.data.search_grid.points",
        "planner_agent.data.search_grid.points",
        "data.search_grid.points",
        "grid_points",
        # This graph stores the Planner's output under state["plan"],
        # with a flat "grid_points" key (see planner_node.py) — add
        # that path so tide_node actually uses the Planner's grid
        # instead of always falling back to a single center point.
        "plan.grid_points",
    )
    if isinstance(pts, list) and pts:
        return pts

    lat = _dig(state, "latitude", "lat", "location.latitude",
               "planner.data.validation.normalized_lat")
    lon = _dig(state, "longitude", "lon", "location.longitude",
               "planner.data.validation.normalized_lon")
    if lat is not None and lon is not None:
        return [{"point_id": "center", "lat": lat, "lon": lon,
                 "bearing_deg": None, "distance_from_center_km": 0}]
    return []


def _extract_date(state: Dict[str, Any]) -> str:
    d = _dig(state, "planned_date", "date", "user_input.planned_date",
             "planner.data.time_range.date", "time_range.date",
             "plan.date")
    if isinstance(d, str) and len(d) >= 10:
        return d[:10]
    return datetime.now(IST).date().isoformat()


def _extract_day_part(state: Dict[str, Any]) -> str:
    dp = _dig(state, "day_part", "user_input.day_part",
              "planner.data.time_range.day_parts_covered")
    if isinstance(dp, list) and dp:
        dp = dp[0]
    dp = str(dp).lower() if dp else "full_day"
    return dp if dp in DAY_PART_WINDOWS else "full_day"


def _extract_activity(state: Dict[str, Any]) -> str:
    a = _dig(state, "activity", "user_input.activity", "planner.data.activity",
             "plan.activity")
    return str(a).lower() if a else "fishing"


def _extract_request_id(state: Dict[str, Any]) -> Optional[str]:
    return _dig(state, "request_id", "planner.request_id", "user_input.request_id")


def _hourly_timestamps(day: str, day_part: str) -> List[datetime]:
    start_h, end_h = DAY_PART_WINDOWS.get(day_part, DAY_PART_WINDOWS["full_day"])
    base = datetime.fromisoformat(day).replace(tzinfo=DAY_PART_TZ)
    return [base.replace(hour=h) for h in range(start_h, end_h + 1)]


# ---------------------------------------------------------------------------
# Flags — the "why" the Risk Agent will quote back to the user
# ---------------------------------------------------------------------------

def _build_flags(day_stats: Dict[str, Any],
                 timeline: List[Dict[str, Any]],
                 activity: str) -> List[Dict[str, Any]]:
    flags: List[Dict[str, Any]] = []

    rng = day_stats.get("tidal_range_m")
    if rng is not None and rng >= LARGE_RANGE_M:
        flags.append({
            "code": "LARGE_TIDAL_RANGE",
            "severity": "CAUTION",
            "message": f"Tidal range today is {rng} m (>= {LARGE_RANGE_M} m). "
                       "Expect strong tidal streams around mid-tide, especially "
                       "in channels, creeks and river mouths.",
            "evidence": {"tidal_range_m": rng, "threshold_m": LARGE_RANGE_M},
        })

    in_range = [t for t in timeline if t.get("tide_height_m") is not None]
    if in_range and activity in SHALLOW_DRAFT_ACTIVITIES:
        lowest = min(in_range, key=lambda t: t["tide_height_m"])
        if lowest["tide_height_m"] <= VERY_LOW_WATER_M:
            flags.append({
                "code": "LOW_WATER_DURING_ACTIVITY",
                "severity": "CAUTION",
                "message": f"Water level drops to about {lowest['tide_height_m']} m at "
                           f"{lowest['time'][11:16]} IST. Sandbars, harbour bars and "
                           "shallow approaches may not be passable near that time.",
                "evidence": {"lowest_height_m": lowest["tide_height_m"],
                             "at": lowest["time"], "threshold_m": VERY_LOW_WATER_M},
            })

    if not in_range and timeline:
        flags.append({
            "code": "OUTSIDE_PREDICTION_RANGE",
            "severity": "INFO",
            "message": "The requested times fall outside the tide predictions returned "
                       "for this station; tide height and phase were not derived.",
            "evidence": {"timestamps": len(timeline)},
        })

    flags.append({
        "code": "TIDAL_CURRENTS_NOT_MAPPED",
        "severity": "INFO",
        "message": "Tidal current speed and direction are not available in ORCA. "
                   "Absence of a current warning is not evidence of weak currents.",
        "evidence": {"fields": ["tidal_current_velocity_ms", "tidal_current_direction_deg"]},
    })

    return flags


def _point_payload(point: Dict[str, Any],
                   result: Dict[str, Any],
                   timeline: List[Dict[str, Any]],
                   shared: bool) -> Dict[str, Any]:
    anchor = timeline[0] if timeline else result["anchor_state"]
    day = result["day_stats"]
    src = result["source"]
    fresh = "CACHED" if result["cached"] else "LIVE"
    derived = dict(source=f"{src} (derived by ORCA)", freshness=fresh,
                   retrieved_at=result["retrieved_at"])

    return {
        # Tolerant of both this module's own point shape
        # (point_id/lat/lon/distance_from_center_km) and the Planner's
        # tools.generate_grid shape (id/latitude/longitude/distance_km).
        "point_id": point.get("point_id", point.get("id")),
        "lat": point.get("lat", point.get("latitude")),
        "lon": point.get("lon", point.get("longitude")),
        "bearing_deg": point.get("bearing_deg"),
        "distance_from_center_km": point.get(
            "distance_from_center_km", point.get("distance_km")),
        "station": result["station"],
        "station_shared": shared,
        "tides": {
            "tide_height_m": data_field(anchor.get("tide_height_m"), "m", **derived),
            "tide_phase": data_field(anchor.get("tide_phase"), "category", **derived),
            "next_high_tide": data_field(anchor.get("next_high_tide"), "event", **derived),
            "next_low_tide": data_field(anchor.get("next_low_tide"), "event", **derived),
            "tidal_range_m": data_field(day.get("tidal_range_m"), "m", **derived),
            "max_height_m": data_field(day.get("max_height_m"), "m",
                                       source=src, freshness=fresh,
                                       retrieved_at=result["retrieved_at"]),
            "min_height_m": data_field(day.get("min_height_m"), "m",
                                       source=src, freshness=fresh,
                                       retrieved_at=result["retrieved_at"]),
            "tidal_current_velocity_ms": not_mapped_field(
                "m/s", "No tidal-current source wired into ORCA."),
            "tidal_current_direction_deg": not_mapped_field(
                "deg", "No tidal-current source wired into ORCA."),
        },
        "timeline": timeline,
    }


def _envelope(request_id: Optional[str], status: str,
              data: Optional[Dict[str, Any]], errors: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "request_id": request_id,
        "agent": AGENT_NAME,
        "status": status,                 # SUCCESS | PARTIAL | FAILED
        "generated_at": now_utc_iso(),
        "data": data,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# The node
# ---------------------------------------------------------------------------

def tide_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node. Returns a partial state update: {"tide_data": <AgentEnvelope>}."""
    request_id = _extract_request_id(state)
    errors: List[Dict[str, Any]] = []

    grid = _extract_grid(state)
    if not grid:
        return {"tide_data": _envelope(request_id, "FAILED", None, [{
            "category": "INVALID_INPUT",
            "message": "No search grid or coordinates found in state",
            "detail": None, "source": "tide_node",
        }])}

    day = _extract_date(state)
    day_part = _extract_day_part(state)
    activity = _extract_activity(state)
    timestamps = _hourly_timestamps(day, day_part)

    # One day either side so interpolation always has bracketing extrema.
    d = datetime.fromisoformat(day).date()
    from_date = (d - timedelta(days=1)).isoformat()
    to_date = (d + timedelta(days=1)).isoformat()

    per_point = TIDE_GRID_MODE == "per_point"
    lookup_points = grid if per_point else [grid[0]]

    results: Dict[str, Dict[str, Any]] = {}
    for pt in lookup_points:
        pid = pt.get("point_id", pt.get("id", "center"))
        try:
            results[pid] = get_tide(
                pt.get("lat", pt.get("latitude")),
                pt.get("lon", pt.get("longitude")),
                from_date, to_date,
                reference_times=timestamps,
            )
        except TideError as e:
            errors.append({**e.as_dict(), "point_id": pid})
        except Exception as e:  # defensive — a node must never kill the graph
            errors.append({"category": "UNEXPECTED", "message": str(e),
                           "detail": None, "source": "tide_node", "point_id": pid})

    if not results:
        return {"tide_data": _envelope(request_id, "FAILED", None, errors)}

    base_result = results.get(
        grid[0].get("point_id", grid[0].get("id", "center"))
    ) or next(iter(results.values()))
    for w in base_result.get("warnings", []):
        errors.append({"category": "WARNING", "message": w,
                       "detail": None, "source": base_result["source"]})

    points_out: List[Dict[str, Any]] = []
    for pt in grid:
        pid = pt.get("point_id", pt.get("id", "center"))
        if per_point:
            res = results.get(pid)
            if res is None:
                continue          # its error is already in errors[]
            tl = res["timeline"]
            shared = False
        else:
            res = base_result
            tl = base_result["timeline"]
            shared = True
        points_out.append(_point_payload(pt, res, tl, shared))

    timeline = base_result["timeline"]
    day_stats = base_result["day_stats"]
    events_in_window = [
        e for e in base_result["events"]
        if timestamps and timestamps[0].isoformat(timespec="seconds") <= e["time"]
        <= timestamps[-1].isoformat(timespec="seconds")
    ]

    data = {
        "date": day,
        "day_part": day_part,
        "activity": activity,
        "timezone": "Asia/Kolkata (+05:30)",
        "window": {
            "from": timestamps[0].isoformat(timespec="seconds") if timestamps else None,
            "to": timestamps[-1].isoformat(timespec="seconds") if timestamps else None,
            "resolution": "hourly",
        },
        "resolution_mode": TIDE_GRID_MODE,
        "station": base_result["station"],
        "station_note": (
            "All grid points share one INCOIS PAT station: PAT stations are far "
            "more widely spaced than the 2–10 km search grid."
            if not per_point else
            "Each grid point resolved its own nearest PAT station."
        ),
        "summary": {
            "tidal_range_m": day_stats.get("tidal_range_m"),
            "max_height_m": day_stats.get("max_height_m"),
            "min_height_m": day_stats.get("min_height_m"),
            "spring_neap_hint": base_result["fields"]["spring_neap_hint"]["value"],
            "phase_at_window_start": timeline[0]["tide_phase"] if timeline else None,
            "height_at_window_start_m": timeline[0]["tide_height_m"] if timeline else None,
            "next_high_tide": timeline[0]["next_high_tide"] if timeline else None,
            "next_low_tide": timeline[0]["next_low_tide"] if timeline else None,
            "tide_events_in_window": events_in_window,
        },
        "timeline": timeline,
        "events": base_result["events"],
        "points": points_out,
        "flags": _build_flags(day_stats, timeline, activity),
        "not_mapped": ["tidal_current_velocity_ms", "tidal_current_direction_deg"],
        "provenance": {
            "source": base_result["source"],
            "service_url": os.getenv("TIDE_API_URL",
                                     "https://orca-backend-tide.onrender.com/tide"),
            "cached": base_result["cached"],
            "retrieved_at": base_result["retrieved_at"],
            "derivation": "Tide height and phase are cosine-interpolated between "
                          "predicted high/low extrema; they are not measured water "
                          "levels. Predictions are astronomical and exclude storm "
                          "surge, wind setup and river discharge.",
        },
    }

    status = "PARTIAL" if errors else "SUCCESS"
    if per_point and len(points_out) < len(grid):
        status = "PARTIAL"

    return {"tide_data": _envelope(request_id, status, data, errors)}


# Alias, in case your graph registers nodes by a `run` convention.
run = tide_node


# ---------------------------------------------------------------------------
# Wiring reminder (graph/workflow.py)
# ---------------------------------------------------------------------------
#
#   from graph.nodes.tide_node import tide_node
#
#   graph.add_node("tide", tide_node)
#   graph.add_edge("planner", "tide")      # or run it inside the Ocean Agent
#   graph.add_edge("tide", "risk")
#
# Risk Agent reads:  state["tide"]["data"]["flags"]
#                    state["tide"]["data"]["summary"]
#                    state["tide"]["data"]["points"][i]["tides"]
#
# ---------------------------------------------------------------------------
# Smoke test:  python tide_node.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    demo_state = {
        "request_id": "req_tide_demo",
        "planned_date": "2026-09-04",
        "day_part": "morning",
        "activity": "fishing",
        "search_grid": {
            "points": [
                {"point_id": "center", "lat": 19.076, "lon": 72.8777,
                 "bearing_deg": None, "distance_from_center_km": 0},
                {"point_id": "N_5km", "lat": 19.121, "lon": 72.8777,
                 "bearing_deg": 0, "distance_from_center_km": 5},
                {"point_id": "W_5km", "lat": 19.076, "lon": 72.8301,
                 "bearing_deg": 270, "distance_from_center_km": 5},
            ]
        },
    }

    out = tide_node(demo_state)
    env = out["tide"]

    print(f"status: {env['status']}   errors: {len(env['errors'])}")
    if env["data"]:
        print(f"station: {env['data']['station']}")
        print(f"summary: {json.dumps(env['data']['summary'], indent=2, default=str)}")
        print("\ntimeline:")
        for t in env["data"]["timeline"]:
            print(f"  {t['time'][11:16]} IST  h={t['tide_height_m']}  {t['tide_phase']}")
        print("\nflags:")
        for f in env["data"]["flags"]:
            print(f"  [{f['severity']}] {f['code']}: {f['message']}")
    for e in env["errors"]:
        print(f"  ERROR {e['category']}: {e['message']}")
