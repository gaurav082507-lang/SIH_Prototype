# pfz_node.py

from pfz_tool import get_pfz_data


def pfz_node(state):
    """
    PFZ Agent

    Retrieves Potential Fishing Zone information
    for the user's location.

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
                        "pfz_data": {
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
                "pfz_data": {
                    "status": "FAILED",
                    "error": "Latitude and longitude are required"
                }
            }

        # -------------------------------------------------
        # Call PFZ Tool
        # -------------------------------------------------

        data = get_pfz_data(
            latitude=latitude,
            longitude=longitude
        )

        # -------------------------------------------------
        # Return PFZ data
        # -------------------------------------------------

        return {
            "pfz_data": {
                "status": "SUCCESS",
                "latitude": latitude,
                "longitude": longitude,
                "data": data
            }
        }

    except Exception as e:

        return {
            "pfz_data": {
                "status": "FAILED",
                "error": str(e)
            }
        }