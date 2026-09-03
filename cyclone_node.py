# cyclone_node.py

from tools import fetch_cyclone_data


def cyclone_node(state):
    """
    Cyclone Agent

    Retrieves cyclone information around the user's location.

    No LLM is used here.
    """

    try:
        latitude = state.get("latitude")
        longitude = state.get("longitude")

        # If coordinates are not directly in state,
        # try to get them from the planner output.
        if latitude is None or longitude is None:

            plan = state.get("plan", {})

            if isinstance(plan, str):
                import json

                try:
                    plan = json.loads(plan)
                except Exception:
                    return {
                        "cyclone_data": {
                            "status": "FAILED",
                            "error": "Invalid planner output"
                        }
                    }

            latitude = plan.get("latitude")
            longitude = plan.get("longitude")

        if latitude is None or longitude is None:
            return {
                "cyclone_data": {
                    "status": "FAILED",
                    "error": "Latitude and longitude are required"
                }
            }

        # Fetch cyclone information
        data = fetch_cyclone_data(
            latitude=latitude,
            longitude=longitude
        )

        return {
            "cyclone_data": {
                "status": "SUCCESS",
                "latitude": latitude,
                "longitude": longitude,
                "data": data
            }
        }

    except Exception as e:

        return {
            "cyclone_data": {
                "status": "FAILED",
                "error": str(e)
            }
        }