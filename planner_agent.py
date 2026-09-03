"""Mistral-backed planner agent for marine routing and area scans.

NOTE: this module is a standalone/experimental LangChain "create_agent"
implementation and is NOT wired into graph.py. The graph's actual
planner node lives in planner_node.py, which owns its own LLM client
and produces a different (flatter) plan schema that the rest of the
graph's nodes expect: rejected/required_agents/grid_points/date/
latitude/longitude, rather than this file's geocoding/validation/
search_grid/route schema.

Keep that in mind before wiring this module into the graph — its
output would need to be translated into the flat schema first, and
its route[] values (e.g. "weather_agent") use different names than
graph.py's routing keys (e.g. "weather").
"""

from __future__ import annotations

import json
import os
from datetime import UTC, date, datetime
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mistralai import ChatMistralAI

from grid_tools import generate_grid

load_dotenv()

SYSTEM_PROMPT = """You are the Planner Agent of a Marine Intelligence Platform.
Return only valid JSON. Do not answer the user or invent marine data.

First resolve the location from raw_location_text when possible, then validate
whether it is coastal. For a clearly inland marine request, reject it and ask
for a coastal location. Do not invent a precise coast distance. The only tool
available is generate_grid. Use it for a fishing search, with radius_km=5,
and copy its result exactly into search_grid.points. Do not calculate points
yourself. For a single location, still include the center-plus-eight grid
when the request is a fishing activity.

Route selection according to the user query:
- Read the complete user query before creating the route.
- Add every downstream agent required by the query, in execution order.
- Weather requests route to weather_agent; ocean requests route to ocean_agent.
- Tide requests route to tide_agent; spatial or restriction requests route to gis_agent.
- Cyclone, storm, lightning, or hazard requests route to cyclone_agent.
- Fishing-zone requests route to pfz_agent, weather_agent, ocean_agent, and
    gis_agent; add tide_agent when tides affect suitability.
- Marine or fishing safety requests route to weather_agent, ocean_agent,
    tide_agent, and gis_agent when those conditions are requested.
- Keep route limited to agents actually required. Never leave route empty when
    the user query requires a downstream agent.

Use this exact output schema:
{
    "geocoding": {"query": "", "resolved_lat": 0.0, "resolved_lon": 0.0,
        "matched_name": "", "source": "", "confidence": "high|medium|low",
        "status": "OK|FAILED"},
    "validation": {"is_coastal": true, "normalized_lat": 0.0,
        "normalized_lon": 0.0, "coast_distance_km": 0.0, "reason": ""},
    "search_grid": {"radius_km": 5, "pattern": "center_plus_compass_ring_8",
        "points": []},
    "time_range": {"mode": "day_part", "day_parts_covered": [],
        "resolution": "hourly", "decision_method": "llm", "rationale": "",
        "timestamps": []},
    "route": []
}

For a rejected request, return the same geocoding and validation objects,
set search_grid and time_range to empty objects, set route to [], and do not
add fields inside data. The wrapper will add errors and message_to_user.
"""


def build_planner_agent() -> Any:
    """Create the planner agent with the grid calculation tool."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is not set")

    model = ChatMistralAI(
        model=os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
        temperature=0,
        api_key=api_key,
    )
    return create_agent(
        model=model,
        tools=[generate_grid],
        system_prompt=SYSTEM_PROMPT,
    )


def run_planner(request: dict[str, Any]) -> dict[str, Any]:
    """Run the planner and return a JSON-serializable agent response."""
    agent = build_planner_agent()
    user_input = request.get("user_input", {})
    user_query = (
        user_input.get("user_query")
        or user_input.get("user_question")
        or user_input.get("raw_location_text")
        or "No user query supplied"
    )
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"USER QUERY:\n{user_query}\n\n"
                        "Select every required downstream agent for this query "
                        "and place them in the route field.\n\n"
                        f"Request JSON:\n{json.dumps(request, indent=2)}\n\n"
                        "Use coordinates from application state when supplied.\n"
                        f"Today's date: {date.today().isoformat()}\n"
                        f"Resolved latitude: {user_input.get('resolved_lat')}\n"
                        f"Resolved longitude: {user_input.get('resolved_lon')}"
                    ),
                }
            ]
        }
    )
    last_message = result["messages"][-1]
    content = last_message.content
    if isinstance(content, list):
        content = "".join(
            item.get("text", "") for item in content if isinstance(item, dict)
        )
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    plan = json.loads(content.strip())
    rejected = bool(plan.get("validation", {}).get("is_coastal") is False)
    errors = []
    message_to_user = None
    if rejected:
        plan.pop("search_grid", None)
        plan.pop("time_range", None)
        plan["route"] = []
        errors = [{
            "field": "location",
            "reason": "NOT_COASTAL - reject and ask user to re-enter a location",
        }]
        message_to_user = (
            "The location you gave doesn't look like a coastal or marine spot. "
            "Could you name a place closer to the coast - a beach, harbour, "
            "or nearby coastal town?"
        )
    return {
        "request_id": request.get("request_id"),
        "agent": "planner_agent",
        "status": "FAILED" if rejected else "SUCCESS",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "data": plan,
        "errors": errors,
        **({"message_to_user": message_to_user} if message_to_user else {}),
    }


def build_agent_response_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph-compatible node using the existing state input fields.

    Renamed from "planner_node" — that name collided with the unrelated
    (and actually-used) planner_node() in planner_node.py, and code that
    imported from this module expecting the other one's schema was the
    root cause of a crash on every request. This function is currently
    unused by graph.py.
    """
    request = {
        "user_input": {
            "raw_location_text": state.get("location_name"),
            "planned_date": state.get("date"),
            "day_part": state.get("time_range"),
            "activity": state.get("activity"),
            "resolved_lat": state.get("latitude"),
            "resolved_lon": state.get("longitude"),
            "user_question": state.get("user_question"),
        }
    }
    return {"plan": run_planner(request)["data"]}


if __name__ == "__main__":
    with open("1_input.json", encoding="utf-8") as input_file:
        print(json.dumps(run_planner(json.load(input_file)), indent=2))