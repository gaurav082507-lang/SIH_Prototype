# coastline.py
#
# A small coastal-proximity check, built to replace global_land_mask
# inside tools.check_coastal_proximity.
#
# WHY THIS EXISTS
#
# global_land_mask answers "is this point on land" by decompressing a
# global boolean raster into memory. Measured on the deployed service:
# resident memory went from 184 MB to 525 MB on the first call, which
# is an out-of-memory kill on a 512 MB instance. That is 341 MB spent
# on one geographic lookup.
#
# The pipeline only ever reads `is_coastal` (planner_node.py), so the
# raster was buying one boolean. This module answers the same question
# from a list of coastline waypoints and a haversine distance —
# roughly 100 KB of Python floats, no numpy, no data files.
#
# COVERAGE AND ITS LIMITS
#
# The waypoints trace the Indian mainland coast plus the Andaman &
# Nicobar and Lakshadweep groups, at roughly 30-80 km spacing, then
# densified to ~15 km at import. They are approximate — good to a few
# kilometres, which is immaterial against a 10-100 km proximity test,
# but they are NOT a substitute for a real coastline dataset.
#
# Outside that region the module does not guess. Beyond
# COVERAGE_RADIUS_KM from every waypoint it reports coverage="outside"
# and is_coastal=True, so a location it cannot speak for is never
# rejected — the planner's own prompt remains the gate there. It only
# returns is_coastal=False when a point is confidently inland WITHIN
# the covered region (Delhi, Bengaluru, Nagpur), which is exactly the
# case the deterministic check existed to catch.
#
# To cover another country, append its coastline to COAST_SEGMENTS in
# the same north-to-south / along-shore order. Nothing else changes.

from __future__ import annotations

import math
import os

# ============================================================
# TUNABLES
# ============================================================

# Spacing used to fill in between the waypoints below.
DENSIFY_SPACING_KM = float(os.getenv("COASTLINE_SPACING_KM", "15"))

# Past this distance from every waypoint, treat the location as
# outside what this dataset can speak for.
COVERAGE_RADIUS_KM = float(os.getenv("COASTLINE_COVERAGE_KM", "1200"))

EARTH_RADIUS_KM = 6371.0088


# ============================================================
# WAYPOINTS
# ============================================================
#
# Each list is one continuous stretch of coast, in order along the
# shore. Interpolation happens between consecutive points within a
# list, never across lists.

COAST_SEGMENTS: list[list[tuple[float, float]]] = [
    # ---- Mainland: Gujarat south to Kanyakumari (west coast) ----
    [
        (23.60, 68.55),   # Kori Creek, Kachchh
        (22.83, 69.35),   # Mandvi
        (22.47, 69.07),   # Okha
        (22.24, 68.97),   # Dwarka
        (21.64, 69.61),   # Porbandar
        (20.90, 70.37),   # Veraval
        (20.71, 70.98),   # Diu
        (20.87, 71.37),   # Jafrabad
        (21.76, 72.15),   # Bhavnagar
        (21.10, 72.62),   # Hazira / Surat
        (20.40, 72.83),   # Daman
        (19.98, 72.72),   # Dahanu
        (19.30, 72.79),   # Vasai
        (19.08, 72.83),   # Mumbai
        (18.64, 72.87),   # Alibag
        (18.33, 72.96),   # Murud
        (17.99, 73.05),   # Shrivardhan
        (16.99, 73.30),   # Ratnagiri
        (16.35, 73.44),   # Vijaydurg
        (15.86, 73.63),   # Vengurla
        (15.49, 73.83),   # Panaji, Goa
        (15.01, 74.05),   # Canacona
        (14.81, 74.13),   # Karwar
        (14.30, 74.40),   # Honnavar
        (13.97, 74.55),   # Bhatkal
        (13.35, 74.70),   # Malpe / Udupi
        (12.87, 74.84),   # Mangaluru
        (12.50, 74.99),   # Kasaragod
        (11.87, 75.37),   # Kannur
        (11.25, 75.78),   # Kozhikode
        (10.77, 75.92),   # Ponnani
        (10.02, 76.22),   # Kodungallur
        (9.93, 76.27),    # Kochi
        (9.50, 76.34),    # Alappuzha
        (8.89, 76.59),    # Kollam
        (8.48, 76.92),    # Thiruvananthapuram
        (8.08, 77.55),    # Kanyakumari
    ],
    # ---- Mainland: Kanyakumari north to the Sundarbans (east) ----
    [
        (8.08, 77.55),    # Kanyakumari
        (8.50, 78.12),    # Tiruchendur
        (8.76, 78.13),    # Thoothukudi
        (9.37, 78.83),    # Ramanathapuram
        (9.29, 79.31),    # Rameswaram
        (9.85, 79.42),    # Devipattinam coast
        (10.29, 79.85),   # Point Calimere
        (10.77, 79.84),   # Nagapattinam
        (10.92, 79.84),   # Karaikal
        (11.40, 79.79),   # Chidambaram coast
        (11.75, 79.77),   # Cuddalore
        (11.93, 79.83),   # Puducherry
        (12.62, 80.19),   # Mamallapuram
        (13.08, 80.29),   # Chennai
        (13.66, 80.32),   # Pulicat
        (14.28, 80.12),   # Krishnapatnam
        (14.91, 80.06),   # Kavali
        (15.55, 80.15),   # Ongole
        (15.90, 80.55),   # Nizampatnam
        (16.17, 81.14),   # Machilipatnam
        (16.60, 81.75),   # Narsapur coast
        (16.99, 82.25),   # Kakinada
        (17.35, 82.75),   # Uppada coast
        (17.69, 83.30),   # Visakhapatnam
        (17.89, 83.45),   # Bheemunipatnam
        (18.33, 84.12),   # Kalingapatnam
        (19.26, 84.90),   # Gopalpur
        (19.80, 85.83),   # Puri
        (20.32, 86.61),   # Paradip
        (20.79, 86.98),   # Dhamra
        (21.45, 87.05),   # Chandipur
        (21.63, 87.51),   # Digha
        (21.85, 87.95),   # Junput coast
        (22.03, 88.09),   # Haldia
        (21.90, 88.90),   # Sundarbans seafront
    ],
    # ---- Andaman & Nicobar ----
    [
        (13.25, 92.98),   # Diglipur
        (12.50, 92.90),   # Mayabunder
        (11.98, 92.78),   # Rangat
        (11.62, 92.73),   # Port Blair
        (10.60, 92.55),   # Little Andaman
        (9.17, 92.77),    # Car Nicobar
        (8.05, 93.55),    # Nancowry
        (7.00, 93.85),    # Great Nicobar
    ],
    # ---- Lakshadweep ----
    [
        (11.12, 72.73),   # Amini
        (10.57, 72.64),   # Kavaratti
        (10.06, 73.63),   # Kalpeni
        (8.28, 73.05),    # Minicoy
    ],
]


# ============================================================
# GEOMETRY
# ============================================================


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )

    return 2.0 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def _densify(segment: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """
    Fill in intermediate points so the gap between neighbours is at
    most DENSIFY_SPACING_KM.

    Linear interpolation in lat/lon rather than great-circle: over the
    30-80 km hops here the difference is metres, and this keeps the
    module free of any dependency.
    """

    if len(segment) < 2:
        return list(segment)

    points: list[tuple[float, float]] = [segment[0]]

    for (lat1, lon1), (lat2, lon2) in zip(segment, segment[1:]):
        gap = haversine_km(lat1, lon1, lat2, lon2)
        steps = max(1, int(math.ceil(gap / DENSIFY_SPACING_KM)))

        for step in range(1, steps + 1):
            fraction = step / steps
            points.append(
                (
                    lat1 + (lat2 - lat1) * fraction,
                    lon1 + (lon2 - lon1) * fraction,
                )
            )

    return points


COAST_POINTS: list[tuple[float, float]] = [
    point for segment in COAST_SEGMENTS for point in _densify(segment)
]


# ============================================================
# LANDMASS OUTLINE
# ============================================================
#
# Distance to the coast alone is not enough. A vessel 300 km into the
# Arabian Sea is 300 km from shore, and the planner's radius is 100 km
# — so a distance-only test would reject a perfectly legitimate marine
# question. The original land-mask check got this right by answering
# "is this point at sea" first.
#
# This polygon restores that: the two mainland coastal segments closed
# along India's northern land border. Inside means land, outside means
# sea. The northern border points are deliberately coarse — an error of
# 100 km along the Himalaya changes nothing for a marine app; what
# matters is that Delhi falls inside and the Arabian Sea falls outside.

_NORTHERN_BORDER: list[tuple[float, float]] = [
    (21.90, 88.90),   # Sundarbans, where the east coast ends
    (25.20, 89.60),   # north along the eastern border
    (26.70, 88.40),   # Siliguri corridor
    (27.00, 84.00),   # Nepal border
    (28.50, 80.10),
    (30.00, 80.20),   # Uttarakhand
    (32.50, 78.50),   # Himachal
    (34.50, 76.00),   # Kashmir
    (34.00, 74.00),
    (32.00, 74.50),   # Punjab
    (30.00, 73.90),
    (28.00, 70.00),   # Rajasthan
    (25.00, 69.50),
    (23.90, 68.50),   # Kachchh, closing on the west coast start
]

# West coast north-to-south, then east coast south-to-north, then the
# border east-to-west: one closed ring.
INDIA_LANDMASS: list[tuple[float, float]] = (
    COAST_SEGMENTS[0]
    + COAST_SEGMENTS[1][1:]
    + _NORTHERN_BORDER
)


def _point_in_polygon(lat: float, lon: float, polygon) -> bool:
    """Ray casting. Longitude is x, latitude is y."""

    inside = False
    count = len(polygon)

    j = count - 1
    for i in range(count):
        lat_i, lon_i = polygon[i]
        lat_j, lon_j = polygon[j]

        if (lon_i > lon) != (lon_j > lon):
            crossing_lat = (
                (lat_j - lat_i) * (lon - lon_i) / (lon_j - lon_i) + lat_i
            )
            if lat < crossing_lat:
                inside = not inside

        j = i

    return inside


def is_on_land(latitude: float, longitude: float) -> bool:
    """
    True if the point falls inside the Indian landmass outline.

    Only meaningful inside the covered region; callers check coverage
    first.
    """

    return _point_in_polygon(float(latitude), float(longitude), INDIA_LANDMASS)


# ============================================================
# PUBLIC API
# ============================================================


def distance_to_coast_km(latitude: float, longitude: float) -> float:
    """Kilometres from this point to the nearest coastline waypoint."""

    lat = float(latitude)
    lon = float(longitude)

    # A cheap bounding filter first: one degree of latitude is ~111 km,
    # so anything further than that in latitude alone cannot win.
    best = float("inf")

    for point_lat, point_lon in COAST_POINTS:
        if abs(point_lat - lat) * 111.0 >= best:
            continue

        distance = haversine_km(lat, lon, point_lat, point_lon)

        if distance < best:
            best = distance

    return best


def coastal_proximity(
    latitude: float,
    longitude: float,
    max_radius_km: float = 100.0,
) -> dict:
    """
    Same question global_land_mask was answering, at ~0.1% of the memory.

    Returns:
        distance_km     kilometres to the nearest coastline waypoint
        is_over_water   True at sea, False on land, None when unknown
        is_coastal      the field the pipeline actually reads
        coverage        "covered" - inside the dataset's region
                        "outside" - too far away to judge; not rejected

    A point at sea is always coastal, however far offshore — that is
    what a marine assessment is for. Only land is measured against
    max_radius_km.
    """

    distance = distance_to_coast_km(latitude, longitude)

    if distance > COVERAGE_RADIUS_KM:
        # Somewhere this dataset says nothing about. Refuse to reject.
        return {
            "distance_km": round(distance, 1),
            "is_over_water": None,
            "is_coastal": True,
            "coverage": "outside",
        }

    if not is_on_land(latitude, longitude):
        return {
            "distance_km": round(distance, 1),
            "is_over_water": True,
            "is_coastal": True,
            "coverage": "covered",
        }

    return {
        "distance_km": round(distance, 1),
        "is_over_water": False,
        "is_coastal": distance <= max_radius_km,
        "coverage": "covered",
    }
