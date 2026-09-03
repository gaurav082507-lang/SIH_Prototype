# gis_node.py

from gis_tool import fetch_gis_data


def gis_node(state):
    """
    GIS Agent

    Retrieves coastal/marine GIS context for the user's location:
    distance to coast, water depth, nearest restricted zone, marine
    protected area status, nearest maritime boundary, nearest port,
    and EEZ / fishing-zone jurisdiction.

    No LLM is used here.
    """

    try:

        # -------------------------------------------------
        # Get coordinates from state
        # -------------------------------------------------

        latitude = state.get("latitude")
        longitude = state.get("longitude")

        # -------------------------------------------------
        # If coordinates are not directly in state,
        # get them from planner output
        # -------------------------------------------------

        if latitude is None or longitude is None:

            plan = state.get("plan", {})

            if isinstance(plan, str):
                import json

                try:
                    plan = json.loads(plan)
                except Exception:
                    return {
                        "gis_data": {
                            "status": "FAILED",
                            "error": "Invalid planner output"
                        }
                    }

            latitude = plan.get("latitude")
            longitude = plan.get("longitude")

        # -------------------------------------------------
        # Validate coordinates
        # -------------------------------------------------

        if latitude is None or longitude is None:
            return {
                "gis_data": {
                    "status": "FAILED",
                    "error": "Latitude and longitude are required"
                }
            }

        # -------------------------------------------------
        # Call GIS Tool
        # -------------------------------------------------

        result = fetch_gis_data(
            latitude=latitude,
            longitude=longitude
        )

        # -------------------------------------------------
        # fetch_gis_data() never raises — it reports failure via
        # result["success"] instead (timeout, HTTP error, upstream
        # success=false, etc.). Reflect that here so downstream
        # consumers (recommendation_node's "if an agent failed,
        # mention that its data was unavailable" instruction) can
        # actually tell success from failure, instead of always
        # reporting SUCCESS regardless of what happened.
        # -------------------------------------------------

        if not result.get("success"):
            return {
                "gis_data": {
                    "status": "FAILED",
                    "latitude": latitude,
                    "longitude": longitude,
                    "error": result.get("error_message", "GIS lookup failed")
                }
            }

        # -------------------------------------------------
        # Return GIS data
        # -------------------------------------------------

        return {
            "gis_data": {
                "status": "SUCCESS",
                "latitude": latitude,
                "longitude": longitude,
                "data": result["gis"]
            }
        }

    except Exception as e:

        return {
            "gis_data": {
                "status": "FAILED",
                "error": str(e)
            }
        }
