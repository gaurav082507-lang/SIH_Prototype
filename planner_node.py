# planner_node.py
#
# NOTE: this file previously did
#     from planner_agent import planner_llm, PLANNER_PROMPT
# but planner_agent.py never defined either name (it defines
# build_planner_agent()/run_planner() with a completely different
# output schema: geocoding/validation/search_grid/route, instead of
# the flat rejected/required_agents/grid_points/date/latitude/longitude
# shape that graph.py's routing and every specialist node
# (weather_node, ocean_node, ecosystem_node, cyclone_node, pfz_node)
# actually read). That import crashed on the very first request.
#
# This node is now self-contained: it owns its own Mistral client and
# prompt, and always builds the search grid deterministically via
# tools.generate_grid rather than trusting the LLM to compute
# coordinates.

import json
import os
from datetime import date as _date

from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

from tools import generate_grid

load_dotenv()


# ---------------------------------------------------------
# LLM
# ---------------------------------------------------------

planner_llm = ChatMistralAI(
    model=os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
    temperature=0,
    api_key=os.getenv("MISTRAL_API_KEY"),
    # See the matching comment in recommendation_node.py: no timeout
    # was set here to begin with, so it's left alone rather than
    # risking an unverified None crashing this at import time.
    # max_retries is raised because a 429 (rate limit) needs retries
    # with backoff to actually recover, which a timeout change cannot
    # provide.
    max_retries=6,
)


# ---------------------------------------------------------
# PROMPT
# ---------------------------------------------------------

ALLOWED_AGENTS = {
    "weather",
    "ocean",
    "tide",
    "cyclone",
    "ecosystem",
    "pfz",
    "gis",
}

PLANNER_PROMPT = ChatPromptTemplate.from_template(
"""
You are the Planner Agent of a Marine Intelligence Platform.

Read the user's marine-related question and decide which specialist
agents are required to answer it, using ONLY these agent names:

- weather   : temperature, rainfall, wind, visibility, atmospheric conditions
- ocean     : waves, swell, sea state, ocean conditions
- tide      : high/low tide, tidal range, tidal conditions
- cyclone   : cyclones, tropical storms, storm warnings
- ecosystem : marine ecosystem, biodiversity, marine species
- pfz       : potential fishing zones / where to fish
- gis       : distance to coast, water depth, restricted/protected zones,
              maritime/EEZ boundary, nearest port (geofencing context)

Rules:
- Include every agent whose data is actually needed to answer the question.
- If the question is about fishing or a fishing trip, include "pfz",
  "weather", "ocean" and "tide".
- If the question is about general safety at sea, include "weather",
  "ocean" and "tide".
- "gis" always runs automatically for every accepted request regardless
  of what you choose here, so you do not need to include it — but it is
  harmless if you do.
- Never leave required_agents empty unless the request has nothing to
  do with marine conditions at all.
- Do not invent data yourself — you are only selecting which agents run.

Return ONLY valid JSON, no markdown fences, in this exact schema:

{{
    "rejected": false,
    "rejection_reason": null,
    "required_agents": ["weather", "ocean"],
    "activity": "fishing",
    "date": "YYYY-MM-DD"
}}

"rejected" should be true only if the user_question has nothing to do
with marine/ocean conditions at all — in that case set required_agents
to [] and explain why in rejection_reason.

"date" should be today's date ({today}) unless the user asked about a
specific different date.

User question:
{user_question}

Latitude: {latitude}
Longitude: {longitude}
"""
)


def planner_node(state):
    """
    Planner Agent

    Uses Mistral to:
    1. Understand the user's marine query
    2. Determine the required specialist agents
    3. Determine the date / activity
    4. Generate grid points deterministically (tools.generate_grid)
    5. Return a structured plan
    """

    user_question = state.get("user_question", "")

    latitude = state.get("latitude")
    longitude = state.get("longitude")

    if not user_question:
        return {
            "plan": {
                "rejected": True,
                "rejection_reason": "No user question provided",
                "required_agents": []
            }
        }

    if latitude is None or longitude is None:
        return {
            "plan": {
                "rejected": True,
                "rejection_reason": "Latitude and longitude are required",
                "required_agents": []
            }
        }

    try:
        response = planner_llm.invoke(
            PLANNER_PROMPT.format(
                user_question=user_question,
                latitude=latitude,
                longitude=longitude,
                today=_date.today().isoformat(),
            )
        )

        content = response.content

        # Remove markdown code fences if the LLM adds them
        if "```json" in content:
            content = content.replace("```json", "").replace("```", "")
        elif "```" in content:
            content = content.replace("```", "")

        plan = json.loads(content.strip())

        # -----------------------------------------------------
        # Validate / normalize required_agents
        # -----------------------------------------------------

        required_agents = plan.get("required_agents", [])

        if not isinstance(required_agents, list):
            required_agents = []

        required_agents = [
            agent for agent in required_agents
            if agent in ALLOWED_AGENTS
        ]

        plan["required_agents"] = required_agents

        # -----------------------------------------------------
        # Coordinates and date always come from application state,
        # not the LLM, so they can't drift or be hallucinated.
        # -----------------------------------------------------

        plan["latitude"] = latitude
        plan["longitude"] = longitude

        if not plan.get("date"):
            plan["date"] = _date.today().isoformat()

        if not plan.get("activity"):
            plan["activity"] = "fishing"

        # -----------------------------------------------------
        # Deterministic search grid — never computed by the LLM.
        # -----------------------------------------------------

        if plan.get("rejected"):
            plan["grid_points"] = []
        else:
            grid_result = generate_grid.invoke({
                "center_lat": latitude,
                "center_lon": longitude,
                "radius_km": 40,
                "num_points": 10,
            })
            plan["grid_points"] = grid_result

        return {
            "plan": plan
        }

    except json.JSONDecodeError as e:

        return {
            "plan": {
                "rejected": True,
                "rejection_reason": "Planner returned invalid JSON",
                "required_agents": [],
                "error": str(e)
            }
        }

    except Exception as e:

        return {
            "plan": {
                "rejected": True,
                "rejection_reason": "Planner execution failed",
                "required_agents": [],
                "error": str(e)
            }
        }
