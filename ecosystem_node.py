# ecosystem_node.py

from ecosystem_tool import get_ecosystem_data_for_grid


def ecosystem_node(state):
    """
    Ecosystem Agent

    Retrieves marine ecosystem information for the
    grid points selected by the Planner.

    No LLM is used here.
    """

    plan = state.get("plan", {})

    # Handle planner output as JSON string
    if isinstance(plan, str):
        import json

        try:
            plan = json.loads(plan)
        except Exception:
            return {
                "ecosystem_data": {
                    "status": "FAILED",
                    "error": "Invalid planner output"
                }
            }

    grid_points = plan.get("grid_points", [])
    date = plan.get("date")

    if not grid_points:
        return {
            "ecosystem_data": {
                "status": "FAILED",
                "error": "No grid points provided by Planner"
            }
        }

    if not date:
        return {
            "ecosystem_data": {
                "status": "FAILED",
                "error": "No date provided by Planner"
            }
        }

    try:

        # get_ecosystem_data_for_grid's parameter is named "locations",
        # not "grid_points" — the mismatched keyword raised a TypeError
        # on every call.
        ecosystem_results = get_ecosystem_data_for_grid(
            locations=grid_points,
            date=date
        )

        return {
            "ecosystem_data": {
                "status": "SUCCESS",
                "source": "ORCA Ecosystem API",
                "date": date,
                "grid_results": ecosystem_results
            }
        }

    except Exception as e:

        return {
            "ecosystem_data": {
            "status": "FAILED",
            "error": str(e)
            }
        }