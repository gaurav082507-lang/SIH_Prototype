# planner_node.py
#
# Planner node using ChatGroq.
#
# Responsibilities:
#   1. Select specialist agents.
#   2. Detect fish/fishing queries and always include ecosystem.
#   3. Determine date/activity.
#   4. Generate deterministic grid points.
#   5. Parse GPT-OSS output robustly.
#   6. Validate the final plan locally with Pydantic.
#
# Important:
#   - No Groq response_format is used.
#   - GPT-OSS reasoning is set to low.
#   - Reasoning is excluded from the returned content.
#   - JSON is parsed locally.

import json
import os
import traceback
from datetime import date as _date

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from tools import generate_grid
from schemas import Plan

load_dotenv()


# =========================================================
# LLM
# =========================================================

planner_llm = ChatGroq(
    model=os.getenv(
        "GROQ_MODEL_PLANNER",
        "openai/gpt-oss-120b",
    ),
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
    max_retries=2,

    # GPT-OSS is a reasoning model.
    # Keep reasoning low so that enough completion budget
    # remains for the actual JSON response.
    reasoning_effort="low",

    # Do not return the reasoning in response.content.
    include_reasoning=False,

    # Planner only needs a very small JSON object.
    max_completion_tokens=300,
)


# =========================================================
# ALLOWED AGENTS
# =========================================================

ALLOWED_AGENTS = {
    "weather",
    "ocean",
    "tide",
    "cyclone",
    "ecosystem",
    "pfz",
    "gis",
}


# =========================================================
# PROMPT LIMITS
# =========================================================

# History is intentionally disabled.
# The planner only needs the current question.
MAX_HISTORY_TURNS_IN_PROMPT = 0
MAX_HISTORY_CHARS = 0

# Prevent extremely large user questions from increasing
# the Groq TPM usage.
MAX_QUESTION_CHARS = 2000


# =========================================================
# PLANNER PROMPT
# =========================================================

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

Return ONLY ONE valid JSON object.
No markdown.
No explanation.
No text before or after the JSON.

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


# =========================================================
# HISTORY
# =========================================================

def _format_history(conversation_history):
    """
    History is intentionally disabled.

    Keeping previous conversation turns out of the planner
    prevents unnecessary token usage on the Groq free tier.
    """
    return "(no prior turns)"


# =========================================================
# JSON EXTRACTION
# =========================================================

def _parse_planner_json(content):
    """
    Robustly parse JSON returned by the planner.

    Handles:
      - normal JSON
      - ```json fenced JSON
      - ``` fenced JSON
      - small text surrounding the JSON
      - LangChain content blocks
      - empty responses
    """

    # -----------------------------------------------------
    # Handle LangChain content blocks
    # -----------------------------------------------------

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, dict):
                text = item.get("text", "")

                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))

        content = "".join(parts)

    content = str(content or "").strip()

    print(f"[planner] raw response chars={len(content)}")
    print(f"[planner] raw response={content!r}")

    # -----------------------------------------------------
    # Empty response
    # -----------------------------------------------------

    if not content:
        raise json.JSONDecodeError(
            "Planner returned an empty response",
            content,
            0,
        )

    # -----------------------------------------------------
    # Remove markdown fences
    # -----------------------------------------------------

    if "```json" in content:
        content = content.split("```json", 1)[1]

        if "```" in content:
            content = content.split("```", 1)[0]

        content = content.strip()

    elif "```" in content:
        content = content.replace("```", "").strip()

    # -----------------------------------------------------
    # First attempt:
    # response itself is JSON
    # -----------------------------------------------------

    try:
        return json.loads(content)

    except json.JSONDecodeError:
        pass

    # -----------------------------------------------------
    # Second attempt:
    # extract the JSON object from surrounding text
    # -----------------------------------------------------

    start = content.find("{")
    end = content.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise json.JSONDecodeError(
            "No JSON object found in planner response",
            content,
            0,
        )

    json_text = content[start:end + 1]

    return json.loads(json_text)


# =========================================================
# PLANNER NODE
# =========================================================

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

    conversation_history = state.get(
        "conversation_history",
        [],
    )

    # -----------------------------------------------------
    # Validate question
    # -----------------------------------------------------

    if not user_question:
        return {
            "plan": {
                "rejected": True,
                "rejection_reason": "No user question provided",
                "required_agents": [],
            }
        }

    # -----------------------------------------------------
    # Validate coordinates
    # -----------------------------------------------------

    if latitude is None or longitude is None:
        return {
            "plan": {
                "rejected": True,
                "rejection_reason": (
                    "Latitude and longitude are required"
                ),
                "required_agents": [],
            }
        }

    # -----------------------------------------------------
    # Limit question size
    # -----------------------------------------------------

    user_question = user_question[:MAX_QUESTION_CHARS]

    try:

        # =================================================
        # Build prompt
        # =================================================

        prompt_text = PLANNER_PROMPT.format(
            user_question=user_question,
            latitude=latitude,
            longitude=longitude,
            today=_date.today().isoformat(),
        )

        print(
            f"[planner] prompt chars={len(prompt_text)} "
            f"words={len(prompt_text.split())}"
        )

        # =================================================
        # Call Groq
        # =================================================

        response = planner_llm.invoke(prompt_text)

        print(
            f"[planner] response type="
            f"{type(response.content)}"
        )

        # =================================================
        # Parse JSON
        # =================================================

        plan = _parse_planner_json(
            response.content
        )

        # -------------------------------------------------
        # Make sure the model returned an object
        # -------------------------------------------------

        if not isinstance(plan, dict):
            raise ValueError(
                "Planner JSON must be an object"
            )

        # =================================================
        # Normalize required_agents
        # =================================================

        required_agents = plan.get(
            "required_agents",
            [],
        )

        if not isinstance(required_agents, list):
            required_agents = []

        # Keep only allowed agents.
        required_agents = [
            agent
            for agent in required_agents
            if isinstance(agent, str)
            and agent in ALLOWED_AGENTS
        ]

        # =================================================
        # IMPORTANT FISH SAFETY NET
        # =================================================
        #
        # The prompt already tells the model to include
        # ecosystem for fish/fishing questions.
        #
        # This application-level rule makes sure the model
        # cannot accidentally omit ecosystem.
        #
        # This is intentionally based on the user's question,
        # not on model output.

        question_lower = user_question.lower()

        fish_keywords = (
            "fish",
            "fishing",
            "fisheries",
            "fisherman",
            "fishermen",
            "catch",
            "tuna",
            "sardine",
            "mackerel",
            "anchovy",
            "pomfret",
            "hilsa",
            "kingfish",
            "seer fish",
            "shrimp",
            "prawn",
            "species",
            "target species",
            "where to fish",
            "fishing area",
            "fishing spot",
            "fishing zone",
        )

        is_fish_query = any(
            keyword in question_lower
            for keyword in fish_keywords
        )

        if is_fish_query:
            if "ecosystem" not in required_agents:
                required_agents.append("ecosystem")

        # -------------------------------------------------
        # PFZ safety net
        # -------------------------------------------------

        pfz_keywords = (
            "pfz",
            "potential fishing zone",
            "potential fishing zones",
            "fishing hotspot",
            "fishing hotspots",
            "best fishing location",
            "best fishing locations",
            "where to fish",
            "fishing zone",
            "fishing zones",
        )

        is_pfz_query = any(
            keyword in question_lower
            for keyword in pfz_keywords
        )

        if is_pfz_query:
            if "pfz" not in required_agents:
                required_agents.append("pfz")

            if "ecosystem" not in required_agents:
                required_agents.append("ecosystem")

        # -------------------------------------------------
        # Fishing-trip safety net
        # -------------------------------------------------

        fishing_trip_keywords = (
            "fishing trip",
            "go fishing",
            "going fishing",
            "plan a fishing",
            "fishing tomorrow",
            "fishing today",
        )

        is_fishing_trip = any(
            keyword in question_lower
            for keyword in fishing_trip_keywords
        )

        if is_fishing_trip:
            for agent in (
                "ecosystem",
                "pfz",
                "weather",
                "ocean",
                "tide",
            ):
                if agent not in required_agents:
                    required_agents.append(agent)

        plan["required_agents"] = required_agents

        # =================================================
        # Application-owned values
        # =================================================

        plan["latitude"] = latitude
        plan["longitude"] = longitude

        if not plan.get("date"):
            plan["date"] = _date.today().isoformat()

        if not plan.get("activity"):
            plan["activity"] = (
                "fishing"
                if is_fish_query
                else "marine_analysis"
            )

        # =================================================
        # Ensure rejection value is boolean
        # =================================================

        if not isinstance(
            plan.get("rejected"),
            bool,
        ):
            plan["rejected"] = False

        # =================================================
        # Deterministic search grid
        # =================================================

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

        # =================================================
        # Local schema validation
        # =================================================

        try:
            Plan.model_validate(plan)

        except ValidationError as ve:
            # Keep the plan usable but expose schema issues.
            plan["schema_warning"] = str(ve)

            print(
                "[planner] schema warning:",
                str(ve),
            )

        # =================================================
        # Debug final plan
        # =================================================

        print(
            "[planner] final required_agents=",
            plan.get("required_agents"),
        )

        print(
            "[planner] final activity=",
            plan.get("activity"),
        )

        print(
            "[planner] final date=",
            plan.get("date"),
        )

        # =================================================
        # Return state
        # =================================================

        return {
            "plan": plan
        }

    # =====================================================
    # JSON ERROR
    # =====================================================

    except json.JSONDecodeError as e:

        print(
            "[planner] JSON parsing failed:",
            repr(e),
        )

        print(
            "[planner] user question:",
            repr(user_question),
        )

        return {
            "plan": {
                "rejected": True,
                "rejection_reason": (
                    "Planner returned invalid JSON"
                ),
                "required_agents": [],
                "error": str(e),
            }
        }

    # =====================================================
    # GENERAL ERROR
    # =====================================================

    except Exception as e:

        print(
            "[planner] Planner execution failed:",
            repr(e),
        )

        traceback.print_exc()

        return {
            "plan": {
                "rejected": True,
                "rejection_reason": (
                    "Planner execution failed"
                ),
                "required_agents": [],
                "error": repr(e),
            }
        }
