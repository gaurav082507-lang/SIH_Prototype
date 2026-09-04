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
        # get_pfz_data() returns a structured dict, never raises:
        #   {
        #     "success": bool,
        #     "error_code": str | None,
        #     "error_message": str | None,
        #     "pfz": dict | None,   # the actual upstream payload
        #     "queried_at": str,
        #     "query": {...},
        #     "raw_response": str,  # only present on some failures
        #   }
        # This must be checked explicitly — "success": False is a
        # normal, expected outcome (upstream down/timeout/bad body),
        # not something to paper over as SUCCESS.
        result = get_pfz_data(
            latitude=latitude,
            longitude=longitude
        )

        if not result.get("success"):
            return {
                "pfz_data": {
                    "status": "FAILED",
                    "latitude": latitude,
                    "longitude": longitude,
                    "error": result.get("error_message") or "PFZ data unavailable",
                    "error_code": result.get("error_code"),
                }
            }

        # -------------------------------------------------
        # Return PFZ data
        # -------------------------------------------------
        # Unwrap "pfz" so downstream consumers keep seeing the same
        # shape they always have (pfz_data.data == the raw upstream
        # PFZ payload), even though get_pfz_data() itself now wraps
        # that payload in a success/error envelope.
        return {
            "pfz_data": {
                "status": "SUCCESS",
                "latitude": latitude,
                "longitude": longitude,
                "data": result.get("pfz"),
            }
        }
    except Exception as e:
        return {
            "pfz_data": {
                "status": "FAILED",
                "error": str(e)
            }
        }
