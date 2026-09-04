# ============================================================
# Add this to tools.py (alongside generate_grid)
# ============================================================
#
# Requires: pip install global-land-mask
# (pure-offline land/ocean raster lookup, no API key, no network calls)

import math

from global_land_mask import globe
from langchain_core.tools import tool


def _offset_latlon(lat, lon, distance_km, bearing_deg):
    """Return a (lat, lon) point `distance_km` away from (lat, lon) at bearing_deg."""
    R = 6371.0  # Earth radius km
    bearing = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)

    lat2 = math.asin(
        math.sin(lat1) * math.cos(distance_km / R)
        + math.cos(lat1) * math.sin(distance_km / R) * math.cos(bearing)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(distance_km / R) * math.cos(lat1),
        math.cos(distance_km / R) - math.sin(lat1) * math.sin(lat2),
    )

    return math.degrees(lat2), math.degrees(lon2)


@tool
def check_coastal_proximity(
    latitude: float,
    longitude: float,
    max_radius_km: float = 100,
    ring_samples: int = 16,
) -> dict:
    """
    Deterministically check whether a lat/lon point is on/near a coastline.

    Returns:
        {
            "is_over_water": bool,       # point itself is on the sea
            "is_coastal": bool,          # point is on water OR within max_radius_km of a coast
            "approx_distance_km": float | None,  # nearest ring radius where land/water boundary found
            "checked_radii_km": [list of radii sampled],
        }

    Method: uses an offline land/ocean raster (global-land-mask). If the
    point itself is over water, it's coastal by definition. Otherwise,
    samples points in rings of increasing radius around the coordinate;
    if any ring mixes land and water samples, a coastline crosses that
    ring, so the point is considered coastal at ~that distance.
    """
    lat = float(latitude)
    lon = float(longitude)

    is_over_water = bool(globe.is_land(lat, lon)) is False

    if is_over_water:
        return {
            "is_over_water": True,
            "is_coastal": True,
            "approx_distance_km": 0.0,
            "checked_radii_km": [],
        }

    # Point is on land -> check rings of increasing radius for a coastline.
    radii_to_check = [r for r in (10, 25, 50, max_radius_km) if r <= max_radius_km]
    if max_radius_km not in radii_to_check:
        radii_to_check.append(max_radius_km)

    for radius in radii_to_check:
        found_water = False
        for i in range(ring_samples):
            bearing = (360.0 / ring_samples) * i
            sample_lat, sample_lon = _offset_latlon(lat, lon, radius, bearing)
            if not globe.is_land(sample_lat, sample_lon):
                found_water = True
                break

        if found_water:
            return {
                "is_over_water": False,
                "is_coastal": True,
                "approx_distance_km": radius,
                "checked_radii_km": radii_to_check,
            }

    return {
        "is_over_water": False,
        "is_coastal": False,
        "approx_distance_km": None,
        "checked_radii_km": radii_to_check,
    }
