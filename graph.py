# graph.py
#
# ORCA Marine Intelligence LangGraph pipeline.
#
# Planner selects the required specialists.
# GIS runs for every accepted query.
# The final recommendation waits for the selected specialist branches.

from langgraph.graph import StateGraph, START, END

from state import MarineState
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
# ROUTING
# ============================================================

def route_after_planner(state: MarineState):
    plan = state.get("plan", {})

    if isinstance(plan, str):
        import json

        try:
            plan = json.loads(plan)
        except Exception:
            plan = {}

    if not isinstance(plan, dict):
        return ["recommendation"]

    if plan.get("rejected", False):
        return ["recommendation"]

    required_agents = plan.get("required_agents", [])

    if not isinstance(required_agents, list):
        required_agents = []

    required_agents = {
        str(agent).strip().lower()
        for agent in required_agents
    }

    routes = []

    # GIS is the application baseline and runs for every accepted query.
    routes.append("gis")

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

    return routes


# ============================================================
# GRAPH
# ============================================================

builder = StateGraph(MarineState)

builder.add_node("planner", planner_node)
builder.add_node("weather", weather_node)
builder.add_node("ocean", ocean_node)
builder.add_node("tide", tide_node)
builder.add_node("cyclone", cyclone_node)
builder.add_node("ecosystem", ecosystem_node)
builder.add_node("pfz", pfz_node)
builder.add_node("gis", gis_node)
builder.add_node("recommendation", recommendation_node)

builder.add_edge(START, "planner")

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
        "recommendation": "recommendation",
    },
)

# Every specialist branch feeds the final synthesis node.
# LangGraph synchronizes multiple incoming edges before executing
# the downstream node.
builder.add_edge("weather", "recommendation")
builder.add_edge("ocean", "recommendation")
builder.add_edge("tide", "recommendation")
builder.add_edge("cyclone", "recommendation")
builder.add_edge("ecosystem", "recommendation")
builder.add_edge("pfz", "recommendation")
builder.add_edge("gis", "recommendation")

builder.add_edge("recommendation", END)

marine_graph = builder.compile()
