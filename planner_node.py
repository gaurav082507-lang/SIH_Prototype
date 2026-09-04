# planner_node.py
#
# UPDATED: swapped ChatMistralAI -> ChatGroq. Everything else (chat
# memory / follow-up handling, schema validation) is unchanged.
#
# UPDATED for chat memory: the planner now sees conversation_history
# (if the caller supplied any) and can decide a question is a pure
# follow-up on the previous analysis — in which case it's fine to
# return an empty required_agents list, since the caller may have
# carried the previous turn's agent data forward into this same state
# (see the app.py / api.py patches). GIS still always runs regardless
# (see graph.py's route_after_planner), so that part is unchanged.
#
# Also UPDATED to validate the assembled plan with schemas.Plan before
# returning it, so a malformed shape is caught here rather than
# surfacing as a confusing error three nodes later.
#
# (Original note, unchanged: this node is self-contained — it owns its
# own LLM client/prompt rather than importing from planner_agent.py,
# which defines a different, incompatible output schema and is not
# wired into graph.py. See AI_EXISTING_CODE_AUDIT.md.)

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
    max_retries=6,
    max_completion_tokens=500,
    model_kwargs={"response_format": {"type": "json_object"}},
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

MAX_HISTORY_TURNS_IN_PROMPT = 3

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
  do with marine conditions at all, OR (see below) this is a pure
  follow-up question that can be answered from data already collected
  in a previous turn.
- Do not invent data yourself — you are only selecting which agents run.

CONVERSATION HISTORY (most recent last; may be empty if this is the
first message):

{conversation_history}

If PRIOR_TURNS above is non-empty and the user's new question is
clearly a follow-up about the SAME location/analysis already discussed
(e.g. "why is that risky", "what about the northeast point", "explain
that further") rather than a request for new/different conditions, you
may return an empty required_agents list — the caller will still have
the previous turn's data available downstream. If the question asks
about a genuinely new date, activity, or location, treat it as a fresh
request and select agents normally.

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


def _format_history(conversation_history):
    """Compact the last few turns into a short text block for the
    prompt — full agent_data is deliberately NOT included here, only
    the question and the prior recommendation's summary/risk_level, to
    keep this prompt small regardless of how much raw data previous
    turns collected."""
    if not conversation_history:
        return "(no prior turns)"

    recent = conversation_history[-MAX_HISTORY_TURNS_IN_PROMPT:]
    lines = []
    for turn in recent:
        question = turn.get("question", "")
        rec = turn.get("recommendation") or {}
        summary = rec.get("summary", "") if isinstance(rec, dict) else ""
        risk_level = rec.get("risk_level", "") if isinstance(rec, dict) else ""
        lines.append(f"- User asked: {question!r} -> risk_level={risk_level!r}, summary={summary!r}")
    return "\n".join(lines)


def planner_node(state):
    """
    Planner Agent

    Uses the LLM to:
    1. Understand the user's marine query (in light of any conversation
       history the caller supplied)
    2. Determine the required specialist agents (possibly none, for a
       pure follow-up question)
    3. Determine the date / activity
    4. Generate grid points deterministically (tools.generate_grid)
    5. Return a structured, schema-validated plan
    """

    user_question = state.get("user_question", "")

    latitude = state.get("latitude")
    longitude = state.get("longitude")

    conversation_history = state.get("conversation_history", [])

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
                conversation_history=_format_history(conversation_history),
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
        # Skipped for a pure follow-up (no agents required) at the
        # same location, since the caller already has grid points
        # from the previous turn's carried-forward data.
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
        # Schema validation — catches a malformed shape here rather
        # than three nodes downstream. Does not block the pipeline on
        # failure (a demo shouldn't hard-crash on a validation edge
        # case); it just records the warning.
        # -----------------------------------------------------

        try:
            Plan.model_validate(plan)
        except ValidationError as ve:
            plan["schema_warning"] = str(ve)

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
