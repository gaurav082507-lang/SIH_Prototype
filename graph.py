from langgraph.graph import StateGraph, START, END

from state import MarineState
from node_timeout import with_timeout

from planner_node import planner_node
from weather_node import weather_node
from ocean_node import ocean_node
from tide_node import tide_node
from cyclone_node import cyclone_node
from ecosystem_node import ecosystem_node
from pfz_node import pfz_node
from gis_node import gis_node
from recommendation_node import recommendation_node


# ============================================================
# ROUTING AFTER PLANNER
# ============================================================

def route_after_planner(state: MarineState):
    """
    Decide which specialist agents should run after the planner.

    GIS always runs.
    Other agents run only when selected by the planner.
    """

    plan = state.get("plan", {})

    # --------------------------------------------------------
    # Planner may return JSON as a string
    # --------------------------------------------------------
    if isinstance(plan, str):
        import json

        try:
            plan = json.loads(plan)
        except Exception:
            plan = {}

    # --------------------------------------------------------
    # Safety fallback
    # --------------------------------------------------------
    if not isinstance(plan, dict):
        return ["gis"]

    # --------------------------------------------------------
    # If planner rejected the request
    # --------------------------------------------------------
    if plan.get("rejected", False):
        return ["gis"]

    # --------------------------------------------------------
    # Get required agents
    # --------------------------------------------------------
    required_agents = plan.get("required_agents", [])

    if not isinstance(required_agents, list):
        required_agents = []

    required_agents = {
        str(agent).strip().lower()
        for agent in required_agents
        if agent is not None
    }

    # --------------------------------------------------------
    # GIS always runs
    # --------------------------------------------------------
    routes = ["gis"]

    # --------------------------------------------------------
    # Specialist agents
    # --------------------------------------------------------
    if "weather" in required_agents:
        routes.append("weather")

    if "ocean" in required_agents:
        routes.append("ocean")

    if "tide" in required_agents:
        routes.append("tide")

    if "cyclone" in required_agents:
        routes.append("cyclone")

    if "ecosystem" in required_agents:
        routes.append("ecosystem")

    if "pfz" in required_agents:
        routes.append("pfz")

    # --------------------------------------------------------
    # Remove duplicates while preserving order
    # --------------------------------------------------------
    routes = list(dict.fromkeys(routes))

    print("\n" + "=" * 70)
    print("PLANNER ROUTING")
    print("=" * 70)
    print(f"Required agents : {sorted(required_agents)}")
    print(f"Graph routes    : {routes}")
    print("=" * 70 + "\n")

    return routes


# ============================================================
# BUILD GRAPH
# ============================================================

builder = StateGraph(MarineState)


# ============================================================
# PLANNER
# ============================================================

builder.add_node(
    "planner",
    with_timeout(
        "planner",
        planner_node,
        "plan",
    ),
)


# ============================================================
# SPECIALIST AGENTS
# ============================================================

builder.add_node(
    "weather",
    with_timeout(
        "weather",
        weather_node,
    ),
)

builder.add_node(
    "ocean",
    with_timeout(
        "ocean",
        ocean_node,
    ),
)

builder.add_node(
    "tide",
    with_timeout(
        "tide",
        tide_node,
    ),
)

builder.add_node(
    "cyclone",
    with_timeout(
        "cyclone",
        cyclone_node,
    ),
)

builder.add_node(
    "ecosystem",
    with_timeout(
        "ecosystem",
        ecosystem_node,
    ),
)

builder.add_node(
    "pfz",
    with_timeout(
        "pfz",
        pfz_node,
    ),
)

builder.add_node(
    "gis",
    with_timeout(
        "gis",
        gis_node,
    ),
)


# ============================================================
# RECOMMENDATION AGENT
# ============================================================
#
# defer=True makes recommendation execute after the pending
# specialist branches have completed.
#
# This is important because the planner dynamically selects
# which specialist agents should run.
# ============================================================

builder.add_node(
    "recommendation",
    with_timeout(
        "recommendation",
        recommendation_node,
        "recommendation",
    ),
    defer=True,
)


# ============================================================
# START -> PLANNER
# ============================================================

builder.add_edge(
    START,
    "planner",
)


# ============================================================
# PLANNER -> SPECIALIST AGENTS
# ============================================================

builder.add_conditional_edges(
    "planner",
    route_after_planner,
    {
        "weather": "weather",
        "ocean": "ocean",
        "tide": "tide",
        "cyclone": "cyclone",
        "ecosystem": "ecosystem",
        "pfz": "pfz",
        "gis": "gis",
    },
)


# ============================================================
# SPECIALIST AGENTS -> RECOMMENDATION
# ============================================================

builder.add_edge(
    "weather",
    "recommendation",
)

builder.add_edge(
    "ocean",
    "recommendation",
)

builder.add_edge(
    "tide",
    "recommendation",
)

builder.add_edge(
    "cyclone",
    "recommendation",
)

builder.add_edge(
    "ecosystem",
    "recommendation",
)

builder.add_edge(
    "pfz",
    "recommendation",
)

builder.add_edge(
    "gis",
    "recommendation",
)


# ============================================================
# RECOMMENDATION -> END
# ============================================================

builder.add_edge(
    "recommendation",
    END,
)


# ============================================================
# COMPILE GRAPH
# ============================================================

marine_graph = builder.compile()


# ============================================================
# STARTUP LOG
# ============================================================

print("\n" + "=" * 70)
print("MARINE GRAPH COMPILED SUCCESSFULLY")
print("=" * 70)
print("Graph flow:")
print("  START")
print("    ↓")
print("  PLANNER")
print("    ↓")
print("  GIS + selected specialist agents")
print("    ↓")
print("  RECOMMENDATION")
print("    ↓")
print("  END")
print("=" * 70 + "\n")
