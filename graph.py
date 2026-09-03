from langgraph.graph import StateGraph, START, END

from state import MarineState

from planner_node import planner_node
from weather_node import weather_node
from ocean_node import ocean_node
from tide_node import tide_node
from cyclone_node import cyclone_node
from ecosystem_node import ecosystem_node
from pfz_node import pfz_node
from recommendation_node import recommendation_node


# ---------------------------------------------------------
# ROUTING AFTER PLANNER
# ---------------------------------------------------------

def route_after_planner(state: MarineState):

    plan = state.get("plan", {})

    # If planner rejected the request
    if plan.get("rejected", False):
        return ["recommendation"]

    required_agents = plan.get("required_agents", [])

    if not isinstance(required_agents, list):
        required_agents = []

    routes = []

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

    # If nothing was selected, still generate a response
    if not routes:
        routes.append("recommendation")

    return routes


# ---------------------------------------------------------
# BUILD GRAPH
# ---------------------------------------------------------

builder = StateGraph(MarineState)


# --------------------
# NODES
# --------------------

builder.add_node("planner", planner_node)

builder.add_node("weather", weather_node)
builder.add_node("ocean", ocean_node)
builder.add_node("tide", tide_node)
builder.add_node("cyclone", cyclone_node)
builder.add_node("ecosystem", ecosystem_node)
builder.add_node("pfz", pfz_node)

builder.add_node("recommendation", recommendation_node)


# --------------------
# START → PLANNER
# --------------------

builder.add_edge(START, "planner")


# ---------------------------------------------------------
# PLANNER → SPECIALIST AGENTS
# ---------------------------------------------------------

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
        "recommendation": "recommendation",
    }
)


# ---------------------------------------------------------
# SPECIALIST AGENTS → RECOMMENDATION
# ---------------------------------------------------------

builder.add_edge("weather", "recommendation")
builder.add_edge("ocean", "recommendation")
builder.add_edge("tide", "recommendation")
builder.add_edge("cyclone", "recommendation")
builder.add_edge("ecosystem", "recommendation")
builder.add_edge("pfz", "recommendation")


# ---------------------------------------------------------
# RECOMMENDATION → END
# ---------------------------------------------------------

builder.add_edge("recommendation", END)


# ---------------------------------------------------------
# COMPILE
# ---------------------------------------------------------

marine_graph = builder.compile()