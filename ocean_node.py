# ocean_node.py

from tools import get_ocean_data_for_grid


def ocean_node(state):
    """
    Ocean Agent

    Retrieves ocean/sea-condition data for the grid points
    selected by the Planner.

    No LLM is used here.
    """

    plan = state.get("plan", {})

    # Handle planner output if it is returned as a JSON string
    if isinstance(plan, str):
        import json

        try:
            plan = json.loads(plan)
        except Exception:
            return {
                "ocean_data": {
                    "status": "FAILED",
                    "error": "Invalid planner output"
                }
            }

    grid_points = plan.get("grid_points", [])

    if not grid_points:
        return {
            "ocean_data": {
                "status": "FAILED",
                "error": "No grid points provided by Planner"
            }
        }

    # Normalize each point to "id"/"latitude"/"longitude" keys.
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
            "ocean_data": {
                "status": "FAILED",
                "error": "No valid grid points found"
            }
        }

    try:
        # get_ocean_data is a @tool-wrapped function and cannot be
        # called directly with keyword arguments (that raises
        # "TypeError: 'StructuredTool' object is not callable").
        # get_ocean_data_for_grid is the plain batched helper meant
        # for node use, and issues one HTTP call for all points.
        raw_results = get_ocean_data_for_grid(normalized_points)

        ocean_results = [
            {
                "point_id": result.get("id"),
                "latitude": result["location"]["latitude"],
                "longitude": result["location"]["longitude"],
                "data": result
            }
            for result in raw_results
        ]

        return {
            "ocean_data": {
                "status": "SUCCESS",
                "source": "Open-Meteo",
                "grid_results": ocean_results
            }
        }

    except Exception as e:

        return {
            "ocean_data": {
                "status": "FAILED",
                "error": str(e)
            }
        }