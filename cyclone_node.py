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

        # Fetch cyclone information.
        #
        # fetch_cyclone_data() never raises — every failure mode
        # (timeout, connection error, bad HTTP status, invalid JSON,
        # missing "cyclone" field) is converted into a structured
        # {"success": False, "error_message": ...} dict instead.
        #
        # BUG FIX: this node previously always returned
        # "status": "SUCCESS" regardless of what fetch_cyclone_data
        # actually returned, so a cyclone-API outage was completely
        # invisible to the rest of the pipeline (the recommendation
        # LLM would receive a "successful" envelope wrapping error
        # text as if it were real cyclone data). This now checks
        # result["success"] first, mirroring gis_node.py's pattern.
        result = fetch_cyclone_data(
            latitude=latitude,
            longitude=longitude
        )

        if not result.get("success"):
            return {
                "cyclone_data": {
                    "status": "FAILED",
                    "latitude": latitude,
                    "longitude": longitude,
                    "error": result.get("error_message", "Cyclone lookup failed"),
                    "error_code": result.get("error_code"),
                }
            }

        return {
            "cyclone_data": {
                "status": "SUCCESS",
                "latitude": latitude,
                "longitude": longitude,
                "data": result.get("cyclone")
            }
        }

    except Exception as e:

        return {
            "cyclone_data": {
                "status": "FAILED",
                "error": str(e)
            }
        }
