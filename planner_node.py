# planner_node.py
#
# Planner node using ChatGroq with a deliberately compact prompt.
# The planner does NOT use Groq JSON response_format because that can
# trigger json_validate_failed with some model/provider combinations.
# JSON is parsed and validated locally with json.loads / Pydantic.

import json
import os
from datetime import date as _date

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from tools import generate_grid
from schemas import Plan

load_dotenv()


# ---------------------------------------------------------
# LLM
# ---------------------------------------------------------

planner_llm = ChatGroq(
    model=os.getenv("GROQ_MODEL_PLANNER", "openai/gpt-oss-120b"),
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
    max_retries=2,
    # Keep the planner response small because the free Groq tier
    # has an 8,000 TPM limit.
    max_completion_tokens=200,
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

# Keep history out of the planner by default. The planner only needs
# the current question to select agents. This prevents old summaries
# from consuming the free-tier TPM budget.
MAX_HISTORY_TURNS_IN_PROMPT = 0
MAX_HISTORY_CHARS = 0
MAX_QUESTION_CHARS = 2000

PLANNER_PROMPT = ChatPromptTemplate.from_template(
    """
You are the Planner Agent of a Marine Intelligence Platform.

Select specialist agents needed for the user's marine question.

Allowed agents:
- weather: temperature, rainfall, wind, visibility, atmospheric conditions
- ocean: waves, swell, sea state, ocean conditions
- tide: high/low tide and tidal conditions
- cyclone: cyclones, tropical storms, storm warnings
- ecosystem: marine ecosystem, biodiversity, marine species, fish-related data
- pfz: potential fishing zones / where to fish
- gis: distance to coast, depth, restricted/protected zones, EEZ, nearest port

Rules:
- Include every agent whose data is needed.

- FISH / FISHING RULE:
  If the user's question is about fish, fishing, target species,
  fish availability, fish abundance, fish distribution, fish habitat,
  fish suitability, or fishing areas, ALWAYS include "ecosystem".

- FISHING TRIP RULE:
  For a fishing trip, include:
  "ecosystem", "pfz", "weather", "ocean", "tide".

- PFZ RULE:
  If the question asks about PFZ, potential fishing zones,
  fishing hotspots, best fishing locations, or where to fish,
  include BOTH "ecosystem" and "pfz".

- General safety at sea -> weather, ocean, tide.
- GIS runs automatically, so it is optional here.
- Reject only non-marine questions.
- Do not invent data.
- Use today's date unless the user specifies another date.

Return ONLY ONE valid JSON object. No markdown. No explanation.
Use exactly these keys:
rejected, rejection_reason, required_agents, activity, date

Example:
{{"rejected":false,"rejection_reason":null,"required_agents":["ecosystem","pfz","weather","ocean","tide"],"activity":"fishing","date":"{today}"}}

User question:
{user_question}

Latitude: {latitude}
Longitude: {longitude}
"""
)


def _format_history(conversation_history):
    # Intentionally disabled to keep the planner request well below
    # the free-tier TPM limit. Follow-up handling remains in the graph/state.
    return "(no prior turns)"


def planner_node(state):
    """
    Planner Agent.

    Uses the LLM to:
    1. Select specialist agents.
    2. Determine date/activity.
    3. Generate deterministic grid points.
    4. Return a schema-validated plan.
    """

    user_question = (state.get("user_question") or "").strip()
    latitude = state.get("latitude")
    longitude = state.get("longitude")
    conversation_history = state.get("conversation_history", [])

    if not user_question:
        return {
            "plan": {
                "rejected": True,
                "rejection_reason": "No user question provided",
                "required_agents": [],
            }
        }

    if latitude is None or longitude is None:
        return {
            "plan": {
                "rejected": True,
                "rejection_reason": "Latitude and longitude are required",
                "required_agents": [],
            }
        }

    # Hard cap the current question as an additional safety measure.
    user_question = user_question[:MAX_QUESTION_CHARS]

    try:
        prompt_text = PLANNER_PROMPT.format(
            user_question=user_question,
            latitude=latitude,
            longitude=longitude,
            today=_date.today().isoformat(),
        )

        # Debugging: this should be a small prompt. If this prints a huge
        # value, something has been changed in this file/prompt.
        print(
            f"[planner] prompt chars={len(prompt_text)} "
            f"words={len(prompt_text.split())}"
        )

        response = planner_llm.invoke(prompt_text)
        content = response.content

        # LangChain can sometimes expose content as a list of blocks.
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            content = "".join(parts)

        content = str(content).strip()

        # Remove accidental markdown fences.
        if content.startswith("```json"):
            content = content[len("```json"):].strip()
        elif content.startswith("```"):
            content = content[3:].strip()

        if content.endswith("```"):
            content = content[:-3].strip()

        plan = json.loads(content)

        # -----------------------------------------------------
        # Normalize required_agents
        # -----------------------------------------------------

        required_agents = plan.get("required_agents", [])

        if not isinstance(required_agents, list):
            required_agents = []

        plan["required_agents"] = [
            agent for agent in required_agents
            if agent in ALLOWED_AGENTS
        ]

        # -----------------------------------------------------
        # Application-owned values
        # -----------------------------------------------------

        plan["latitude"] = latitude
        plan["longitude"] = longitude

        if not plan.get("date"):
            plan["date"] = _date.today().isoformat()

        if not plan.get("activity"):
            plan["activity"] = "fishing"

        # -----------------------------------------------------
        # Deterministic search grid
        # -----------------------------------------------------

        if plan.get("rejected"):
            plan["grid_points"] = []
        elif not required_agents and conversation_history:
            plan["grid_points"] = []
        else:
            grid_result = generate_grid.invoke({
                "center_lat": latitude,
                "center_lon": longitude,
                "radius_km": 40,
                "num_points": 10,
            })
            plan["grid_points"] = grid_result

        # -----------------------------------------------------
        # Local schema validation
        # -----------------------------------------------------

        try:
            Plan.model_validate(plan)
        except ValidationError as ve:
            plan["schema_warning"] = str(ve)

        return {"plan": plan}

    except json.JSONDecodeError as e:
        return {
            "plan": {
                "rejected": True,
                "rejection_reason": "Planner returned invalid JSON",
                "required_agents": [],
                "error": str(e),
            }
        }

    except Exception as e:
        # Keep the real provider error visible in the terminal.
        import traceback
        traceback.print_exc()

        return {
            "plan": {
                "rejected": True,
                "rejection_reason": "Planner execution failed",
                "required_agents": [],
                "error": repr(e),
            }
        }
