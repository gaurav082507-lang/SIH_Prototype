# weather_node.py

from tools import get_weather_for_grid


def weather_node(state):
    """
    Weather Agent

    Gets weather data for the grid points selected by the Planner.
    No LLM is used here.
    """

    plan = state.get("plan", {})

    if isinstance(plan, str):
        import json
        try:
            plan = json.loads(plan)
        except Exception:
            return {
                "weather_data": {
                    "status": "FAILED",
                    "error": "Invalid planner output"
                }
            }

    grid_points = plan.get("grid_points", [])

    if not grid_points:
        return {
            "weather_data": {
                "status": "FAILED",
                "error": "No grid points provided by Planner"
            }
        }

    # Normalize each point to have "id"/"latitude"/"longitude" keys,
    # since the Planner's grid uses those but some callers historically
    # used "point_id"/"lat"/"lon".
    normalized_points = []
    for point in grid_points:
        lat = point.get("latitude", point.get("lat"))
        lon = point.get("longitude", point.get("lon"))

        if lat is None or lon is None:
            continue

        normalized_points.append({
            "id": point.get("id", point.get("point_id")),
            "latitude": lat,
            "longitude": lon,
        })

    if not normalized_points:
        return {
            "weather_data": {
                "status": "FAILED",
                "error": "No valid grid points found"
            }
        }

    try:
        # Fetch all grid points in a single batched Open-Meteo call
        # instead of looping one HTTP request per point.
        raw_results = get_weather_for_grid(normalized_points)

        weather_results = [
            {
                "point_id": result.get("id"),
                "latitude": result["location"]["latitude"],
                "longitude": result["location"]["longitude"],
                "data": result
            }
            for result in raw_results
        ]

        return {
            "weather_data": {
                "status": "SUCCESS",
                "source": "Open-Meteo",
                "grid_results": weather_results
            }
        }

    except Exception as e:
        return {
            "weather_data": {
                "status": "FAILED",
                "error": str(e)
            }
        }