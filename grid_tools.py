"""Deterministic search-grid tool used by planner_agent.py (the legacy,
standalone agent module — see the note at the top of planner_agent.py).

NOT used by the graph that actually runs (graph.py): planner_node.py,
the planner node graph.py wires in, uses tools.generate_grid instead,
which returns a different point shape (id/latitude/longitude/
bearing_deg/distance_km) matching what weather_node.py, ocean_node.py,
and ecosystem_tool.py expect. Keep that in mind if you re-enable
planner_agent.py — the two generate_grid implementations are not
interchangeable (this one returns lat/lon/point_id/
distance_from_center_km).
"""

from __future__ import annotations

import math

from langchain.tools import tool


@tool
def generate_grid(
    center_lat: float,
    center_lon: float,
    radius_km: float = 5.0,
) -> dict[str, object]:
    """Return the center and eight compass points around a location.

    The local equirectangular approximation is appropriate for small marine
    search areas. The LLM decides when to call this tool, but never performs
    the coordinate arithmetic itself.
    """
    if not -90 <= center_lat <= 90:
        raise ValueError("center_lat must be between -90 and 90")
    if not -180 <= center_lon <= 180:
        raise ValueError("center_lon must be between -180 and 180")
    if radius_km <= 0:
        raise ValueError("radius_km must be greater than zero")
    if abs(center_lat) >= 90:
        raise ValueError("center_lat must be below the poles")

    latitude_delta = radius_km / 111.0
    longitude_delta = radius_km / (
        111.0 * math.cos(math.radians(center_lat))
    )
    diagonal_latitude_delta = radius_km / 111.0 / math.sqrt(2)
    diagonal_longitude_delta = radius_km / (
        111.0 * math.cos(math.radians(center_lat)) * math.sqrt(2)
    )
    compass_points = (
        ("N", 0, latitude_delta, 0),
        ("NE", 45, diagonal_latitude_delta, diagonal_longitude_delta),
        ("E", 90, 0, longitude_delta),
        ("SE", 135, -diagonal_latitude_delta, diagonal_longitude_delta),
        ("S", 180, -latitude_delta, 0),
        ("SW", 225, -diagonal_latitude_delta, -diagonal_longitude_delta),
        ("W", 270, 0, -longitude_delta),
        ("NW", 315, diagonal_latitude_delta, -diagonal_longitude_delta),
    )
    points: list[dict[str, object]] = [{
        "point_id": "center",
        "lat": round(center_lat, 4),
        "lon": round(center_lon, 4),
        "bearing_deg": None,
        "distance_from_center_km": 0,
    }]
    points.extend({
        "point_id": f"{name}_{radius_km:g}km",
        "lat": round(center_lat + lat_offset, 4),
        "lon": round(center_lon + lon_offset, 4),
        "bearing_deg": bearing,
        "distance_from_center_km": radius_km,
    } for name, bearing, lat_offset, lon_offset in compass_points)
    return {
        "radius_km": radius_km,
        "pattern": "center_plus_compass_ring_8",
        "points": points,
    }


# Backward-compatible name for callers using the first prototype.
calculate_grid_points = generate_grid