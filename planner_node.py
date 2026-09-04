# planner_node.py
#
# ORCA Marine Intelligence - Planner Agent
# - Uses Groq GPT-OSS 120B by default.
# - Does not use provider-side response_format/json_schema.
# - Parses JSON locally and also handles JSON wrapped in quotes/fences.
# - Does not assume every query is about fishing.
# - Keeps the prompt small for Groq's free-tier TPM limit.

import ast
import json
import os
import re
from datetime import date as _date

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from tools import generate_grid
from schemas import Plan

load_dotenv()


# ============================================================
# LLM
# ============================================================

planner_llm = ChatGroq(
    model=os.getenv("GROQ_MODEL_PLANNER", "openai/gpt-oss-120b"),
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
    max_retries=2,
    reasoning_effort="low",
    include_reasoning=False,
    max_completion_tokens=300,
)


# ============================================================
# CONSTANTS
# ============================================================

ALLOWED_AGENTS = {
    "weather",
    "ocean",
    "tide",
    "cyclone",
    "ecosystem",
    "pfz",
    "gis",
}

MAX_QUESTION_CHARS = 2000


# ============================================================
# PROMPT
# ============================================================

PLANNER_PROMPT = ChatPromptTemplate.from_template(
    """
You are the Planner Agent for a Marine Intelligence Platform.

Determine whether the user's question is marine-related and select only the
specialist agents actually needed.

Allowed agents:
- weather: weather, wind, rainfall, visibility, temperature, lightning
- ocean: waves, swell, sea state, currents, ocean conditions
- tide: tides, high/low tide, tidal timing
- cyclone: cyclones, storms, tropical systems, storm warnings
- ecosystem: marine life, biodiversity, species, ecosystem conditions
- pfz: potential fishing zones, fishing locations, where/when to fish
- gis: location, distance, coastline, depth, restricted zones, EEZ, ports

Rules:
- Reject ONLY clearly non-marine questions.
- Do NOT assume the user is fishing unless the question says or strongly
  implies fishing/PFZ.
- General sea safety normally needs weather, ocean, and tide.
- Fishing/PFZ questions normally need pfz, weather, ocean, and tide.
- Cyclone questions need cyclone plus any supporting weather/ocean agents.
- Ecosystem questions need ecosystem and supporting marine agents when useful.
- GIS is executed automatically by the application for every accepted query;
  it does not need to be selected.
- Do not invent data.
- Use today's date unless the user explicitly gives another date.
- "activity" must describe the user's actual intent. Do not use "fishing"
  for a non-fishing question.

Return ONLY one valid JSON object with EXACTLY these keys:
rejected, rejection_reason, required_agents, activity, date

Example:
{{"rejected":false,"rejection_reason":null,"required_agents":["weather","ocean","tide"],"activity":"sea safety","date":"{today}"}}

User question:
{user_question}

Latitude: {latitude}
Longitude: {longitude}
"""
)


# ============================================================
# HELPERS
# ============================================================

def _content_to_text(content):
    """Normalize LangChain message content into plain text."""
    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))

        return "".join(parts).strip()

    return str(content or "").strip()


def _extract_json_object(text):
    """
    Extract a JSON object from model output.

    Handles:
      1. normal JSON
      2. markdown fenced JSON
      3. Python/JSON string wrappers such as:
         '{\\n "rejected": false, ... }'
      4. explanatory text before/after the JSON
    """
    text = str(text or "").strip()

    if not text:
        raise ValueError("Planner model returned an empty response.")

    # Remove common markdown fences.
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()

    # First: normal JSON.
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value

        if isinstance(value, str):
            text = value.strip()
    except json.JSONDecodeError:
        pass

    # Second: Python-style quoted string returned by the model.
    # ast.literal_eval correctly converts literal \\n into newlines.
    if len(text) >= 2 and text[0] in ("'", '"') and text[-1] == text[0]:
        try:
            value = ast.literal_eval(text)
            if isinstance(value, dict):
                return value
            if isinstance(value, str):
                text = value.strip()
        except (ValueError, SyntaxError):
            pass

        try:
            value = json.loads(text)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass

    # Third: locate the first balanced JSON object.
    starts = [m.start() for m in re.finditer(r"\{", text)]

    for start in starts:
        depth = 0
        in_string = False
        escaped = False

        for index in range(start, len(text)):
            char = text[index]

            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1

                if depth == 0:
                    candidate = text[start:index + 1]

                    try:
                        value = json.loads(candidate)
                        if isinstance(value, dict):
                            return value
                    except json.JSONDecodeError:
                        pass

                    break

    raise ValueError(
        "Planner model returned non-JSON content: "
        f"{text[:1200]!r}"
    )


def _normalize_agents(value):
    if not isinstance(value, list):
        return []

    result = []

    for agent in value:
        agent = str(agent).strip().lower()

        if agent in ALLOWED_AGENTS and agent not in result:
            result.append(agent)

    return result


# ============================================================
# NODE
# ============================================================

def planner_node(state):
    """Plan the current marine question and prepare the search grid."""

    user_question = str(state.get("user_question") or "").strip()
    latitude = state.get("latitude")
    longitude = state.get("longitude")

    if not user_question:
        return {
            "plan": {
                "rejected": True,
                "rejection_reason": "No user question provided.",
                "required_agents": [],
                "activity": "unknown",
                "date": _date.today().isoformat(),
                "grid_points": [],
            }
        }

    if latitude is None or longitude is None:
        return {
            "plan": {
                "rejected": True,
                "rejection_reason": "Latitude and longitude are required.",
                "required_agents": [],
                "activity": "unknown",
                "date": _date.today().isoformat(),
                "grid_points": [],
            }
        }

    try:
        latitude = float(latitude)
        longitude = float(longitude)
    except (TypeError, ValueError):
        return {
            "plan": {
                "rejected": True,
                "rejection_reason": "Invalid latitude or longitude.",
                "required_agents": [],
                "activity": "unknown",
                "date": _date.today().isoformat(),
                "grid_points": [],
            }
        }

    user_question = user_question[:MAX_QUESTION_CHARS]

    try:
        prompt_text = PLANNER_PROMPT.format(
            user_question=user_question,
            latitude=latitude,
            longitude=longitude,
            today=_date.today().isoformat(),
        )

        print(
            f"[planner] prompt_chars={len(prompt_text)} "
            f"question_chars={len(user_question)}"
        )

        response = planner_llm.invoke(prompt_text)
        content = _content_to_text(response.content)

        print(f"[planner] raw_response_chars={len(content)}")
        print(f"[planner] raw_response={content[:2000]!r}")

        plan = _extract_json_object(content)

        # --------------------------------------------------------
        # Normalize model-owned fields
        # --------------------------------------------------------

        plan["rejected"] = bool(plan.get("rejected", False))

        required_agents = _normalize_agents(
            plan.get("required_agents", [])
        )
        plan["required_agents"] = required_agents

        rejection_reason = plan.get("rejection_reason")
        if plan["rejected"]:
            plan["rejection_reason"] = (
                str(rejection_reason).strip()
                if rejection_reason
                else "Request is not related to marine intelligence."
            )
        else:
            plan["rejection_reason"] = None

        activity = str(plan.get("activity") or "").strip()
        plan["activity"] = activity or "marine conditions"

        plan_date = str(plan.get("date") or "").strip()
        plan["date"] = plan_date or _date.today().isoformat()

        # Application-owned location values.
        plan["latitude"] = latitude
        plan["longitude"] = longitude

        # --------------------------------------------------------
        # Deterministic grid
        # --------------------------------------------------------

        if plan["rejected"]:
            plan["grid_points"] = []
        else:
            grid_result = generate_grid.invoke(
                {
                    "center_lat": latitude,
                    "center_lon": longitude,
                    "radius_km": 40,
                    "num_points": 10,
                }
            )
            plan["grid_points"] = grid_result

        # --------------------------------------------------------
        # Local schema validation
        # --------------------------------------------------------

        try:
            Plan.model_validate(plan)
        except ValidationError as exc:
            # Keep the graph running but expose the schema mismatch.
            plan["schema_warning"] = str(exc)

        return {
            "plan": plan,
            "status": "PLANNED",
        }

    except Exception as exc:
        import traceback
        traceback.print_exc()

        return {
            "plan": {
                "rejected": True,
                "rejection_reason": "Planner execution failed.",
                "required_agents": [],
                "activity": "unknown",
                "date": _date.today().isoformat(),
                "grid_points": [],
                "error": repr(exc),
            },
            "status": "FAILED",
        }
