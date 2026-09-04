# recommendation_node.py

import os
import json

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate


# ---------------------------------------------------------
# LLM
# ---------------------------------------------------------

recommendation_llm = ChatMistralAI(
    model=os.getenv(
        "MISTRAL_MODEL",
        "mistral-medium-3-5"
    ),
    temperature=0,
    api_key=os.getenv("MISTRAL_API_KEY"),
    # No explicit timeout was ever set here, so there was nothing to
    # "remove" — it's left on ChatMistralAI's own default rather than
    # forcing an unbounded wait, since an unverified None here risks a
    # hard crash at import time if the underlying client doesn't accept
    # it. This is the actual fix for "Error response 429 ... rate limit
    # exceeded": a 429 from Mistral is an immediate rejection, not a
    # slow response, so no timeout setting affects it at all. Raising
    # max_retries makes langchain retry the call (with backoff) instead
    # of surfacing the 429 straight to recommendation_node's except
    # block on the very first hit.
    max_retries=6,
)


# ---------------------------------------------------------
# PROMPT
# ---------------------------------------------------------

RECOMMENDATION_PROMPT = ChatPromptTemplate.from_template(
"""
You are the Final Marine Intelligence and Recommendation Agent.

You receive structured data collected by specialist marine
agents.

Your job is to analyze the available data and provide a
clear, evidence-based recommendation for the user.

Do NOT invent data.

Only use information present in the agent outputs.

Available specialist agents:

1. Weather
   - temperature
   - rainfall
   - wind
   - visibility
   - atmospheric conditions

2. Ocean
   - waves
   - wave height
   - swell
   - sea state
   - ocean conditions

3. Tide
   - high tide
   - low tide
   - tidal conditions

4. Cyclone
   - cyclone information
   - storm conditions
   - cyclone warnings
   - cyclone proximity

5. Ecosystem
   - marine ecosystem
   - biodiversity
   - marine species
   - ecological conditions

6. PFZ (Potential Fishing Zone)
   - recommended fishing zones
   - fishing-zone suitability

7. GIS (coastal/marine geospatial context)
   - distance to coast, water depth
   - restricted zones and marine protected areas (geofencing)
   - maritime/EEZ boundary and jurisdiction
   - nearest port


IMPORTANT RULES:

- Consider only agents that actually returned data.
- Do not assume an agent was executed if its data is null.
- Do not invent missing values.
- If an agent failed, mention that its data was unavailable.
- Give priority to safety-related information.
- If dangerous marine conditions are present, clearly warn the user.
- Explain WHY the recommendation was made.
- Keep the final recommendation understandable.
- Do not expose internal implementation details.
- Some agent data below may be abbreviated ("...N more entries
  omitted for brevity...") to keep this prompt a reasonable size —
  treat that as "additional similar data points were collected but
  not shown", not as missing data.

User Question:

{user_question}


Agent Data:

{agent_data}


Return ONLY valid JSON in this format:

{{
    "summary": "Short overall assessment",

    "risk_level": "LOW",

    "recommendation": "Clear recommendation to the user",

    "key_findings": [
        "Important finding 1",
        "Important finding 2"
    ],

    "safety_advice": [
        "Safety advice 1",
        "Safety advice 2"
    ],

    "agent_findings": {{
        "weather": null,
        "ocean": null,
        "tide": null,
        "cyclone": null,
        "ecosystem": null,
        "pfz": null,
        "gis": null
    }}
}}

risk_level must be one of:

LOW
MODERATE
HIGH
SEVERE
"""
)


# ---------------------------------------------------------
# PROMPT-SIZE COMPACTION
# ---------------------------------------------------------
#
# Specialist agents like weather/ocean/ecosystem return dense
# per-grid-point x per-hour time series (see
# final_agent_output_format.md). Dumping that raw into the prompt
# regularly exceeded Mistral's input size limit and failed with:
#   {"type": "invalid_request_prompt_too_long", "code": "3059",
#    "raw_status_code": 400}
# which was being swallowed by the generic `except Exception` below
# and reported as a plain "Recommendation generation failed."
#
# _compact_for_llm keeps every agent's data but caps how many entries
# any list contributes, so the prompt stays bounded no matter how
# much raw data a specialist agent returns. MAX_PROMPT_CHARS is a
# belt-and-suspenders hard cap in case compaction still isn't enough
# (e.g. many agents all required at once).

MAX_LIST_LEN = 6
MAX_COMPACT_DEPTH = 6
MAX_PROMPT_CHARS = 24000


def _compact_for_llm(obj, max_list_len=MAX_LIST_LEN, max_depth=MAX_COMPACT_DEPTH, _depth=0):
    """Recursively shrink long lists so large per-grid-point/per-hour time
    series don't blow out the LLM's prompt size limit. Keeps the first
    (max_list_len - 1) entries of any oversized list and appends a short
    note describing how many were omitted. Dicts and scalars pass through
    unchanged (aside from recursing into their contents)."""

    if _depth >= max_depth:
        return obj

    if isinstance(obj, dict):
        return {
            key: _compact_for_llm(value, max_list_len, max_depth, _depth + 1)
            for key, value in obj.items()
        }

    if isinstance(obj, list):
        if len(obj) > max_list_len:
            kept = obj[: max_list_len - 1]
            compacted = [
                _compact_for_llm(item, max_list_len, max_depth, _depth + 1)
                for item in kept
            ]
            omitted = len(obj) - len(kept)
            compacted.append(f"...{omitted} more entries omitted for brevity...")
            return compacted
        return [
            _compact_for_llm(item, max_list_len, max_depth, _depth + 1)
            for item in obj
        ]

    return obj


# ---------------------------------------------------------
# RECOMMENDATION NODE
# ---------------------------------------------------------

def recommendation_node(state):
    """
    Final Recommendation Agent.

    Combines outputs from all executed specialist agents
    and uses Mistral to generate the final assessment.
    """

    user_question = state.get(
        "user_question",
        ""
    )

    # -----------------------------------------------------
    # Collect specialist outputs
    # -----------------------------------------------------

    agent_data = {
        "weather": state.get("weather_data"),
        "ocean": state.get("ocean_data"),
        "tide": state.get("tide_data"),
        "cyclone": state.get("cyclone_data"),
        "ecosystem": state.get("ecosystem_data"),
        "pfz": state.get("pfz_data"),
        "gis": state.get("gis_data"),
    }

    try:

        # -------------------------------------------------
        # Compact + convert to JSON for the LLM
        # -------------------------------------------------
        # NOTE: agent_data (the full, uncompacted version) is still
        # what we return in agent_findings on both success and
        # failure below, so the UI/caller always sees complete data —
        # only what we SEND to the model is size-capped.

        agent_data_for_prompt = _compact_for_llm(agent_data)

        agent_data_json = json.dumps(
            agent_data_for_prompt,
            ensure_ascii=False,
            indent=2,
            default=str
        )

        if len(agent_data_json) > MAX_PROMPT_CHARS:
            agent_data_json = (
                agent_data_json[:MAX_PROMPT_CHARS]
                + "\n... [truncated: agent data exceeded prompt size limit] ..."
            )

        # -------------------------------------------------
        # Call Mistral
        # -------------------------------------------------

        response = recommendation_llm.invoke(
            RECOMMENDATION_PROMPT.format(
                user_question=user_question,
                agent_data=agent_data_json
            )
        )

        content = response.content

        # -------------------------------------------------
        # Remove markdown code fences
        # -------------------------------------------------

        if "```json" in content:
            content = content.replace(
                "```json",
                ""
            ).replace(
                "```",
                ""
            )

        elif "```" in content:
            content = content.replace(
                "```",
                ""
            )

        # -------------------------------------------------
        # Parse JSON
        # -------------------------------------------------

        recommendation = json.loads(
            content.strip()
        )

        return {
            "recommendation": recommendation,
            "status": "SUCCESS"
        }

    except json.JSONDecodeError as e:

        return {
            "recommendation": {
                "summary": "Unable to parse the final recommendation.",
                "risk_level": "MODERATE",
                "recommendation": "Please review the available marine data.",
                "key_findings": [],
                "safety_advice": [],
                "agent_findings": agent_data,
                "error": str(e)
            },
            "status": "FAILED"
        }

    except Exception as e:

        return {
            "recommendation": {
                "summary": "Recommendation generation failed.",
                "risk_level": "MODERATE",
                "recommendation": "Please review the available marine data.",
                "key_findings": [],
                "safety_advice": [],
                "agent_findings": agent_data,
                "error": str(e)
            },
            "status": "FAILED"
        }
