# recommendation_node.py
#
# Gemini-based Final Marine Intelligence Recommendation Agent.
#
# Responsibilities:
# - Combine all specialist agent outputs.
# - Interpret null values correctly.
# - Treat cyclone=null as "no cyclone threat indicated".
# - Display requested factual values when the user asks for them.
# - Generate the final recommendation.
# - Robustly parse Gemini JSON responses.
# - Preserve complete specialist data in agent_findings.
#
# Gemini is used instead of Groq.
# No Groq 8K/token-budget logic is used.

import json
import os
import re

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from schemas import Recommendation


print("🔥 GEMINI RECOMMENDATION NODE LOADED")
print("FILE:", __file__)


# =========================================================
# GEMINI LLM
# =========================================================

recommendation_llm = ChatGoogleGenerativeAI(
    model=os.getenv(
        "GEMINI_MODEL_RECOMMENDATION",
        "gemini-3.5-flash",
    ),
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    max_retries=2,
)


# =========================================================
# RECOMMENDATION PROMPT
# =========================================================

RECOMMENDATION_PROMPT = ChatPromptTemplate.from_template(
    """
You are the Final Marine Intelligence and Recommendation Agent.

Answer the user's current question using ONLY the supplied marine agent data
and rule-based risk assessment.

=========================================================
IMPORTANT DATA INTERPRETATION
=========================================================

- NULL values are VALID.
- NULL does NOT automatically mean insufficient data.
- A null field means that particular field has no usable finding.
- Ignore individual null fields and continue using all other available data.
- NEVER say "insufficient data" merely because one or more fields are null.
- NEVER invent a value for a null field.
- Do not treat null as an error.

=========================================================
TOTAL NO-DATA RULE
=========================================================

Apply this rule ONLY when ALL marine agents have no usable information.

The agents are:

weather
ocean
tide
cyclone
ecosystem
pfz
gis

If EVERY agent is null, missing, empty, or contains no meaningful finding,
then and ONLY then return:

{
  "summary": "Marine data could not be retrieved for this location right now.",
  "risk_level": "LOW",
  "recommendation": "Try again shortly, or verify the coordinates and retry.",
  "key_findings": [],
  "safety_advice": []
}

IMPORTANT:

- Do NOT call partial data "insufficient data".
- Do NOT say the location is invalid.
- Do NOT assume the location is non-coastal.
- Coastal validation has already been performed earlier in the pipeline.
- If even ONE agent contains meaningful information, perform a normal
  marine assessment.

=========================================================
CYCLONE NULL RULE
=========================================================

If cyclone data is null:

Interpret it as:

"No cyclone threat is indicated by the available cyclone assessment."

Do NOT say:

- "insufficient cyclone data"
- "cyclone data unavailable"
- "unable to assess cyclone conditions"

Do NOT invent cyclone information.

Cyclone null does NOT mean the entire marine assessment has insufficient data.

=========================================================
OTHER NULL VALUES
=========================================================

weather = null
→ Ignore that weather finding and use other available evidence.

ocean = null
→ Ignore that ocean finding and use other available evidence.

tide = null
→ Ignore that tide finding and use other available evidence.

cyclone = null
→ No cyclone threat is indicated by the available cyclone assessment.

ecosystem = null
→ Ignore that ecosystem finding and use other available evidence.

pfz = null
→ Ignore that PFZ finding and use other available evidence.

gis = null
→ Ignore that GIS finding and use other available evidence.

=========================================================
USER-REQUESTED VALUE RULE
=========================================================

If the user explicitly asks for values, measurements, coordinates, times,
conditions, distances, or other factual information, DISPLAY the actual
values available in the supplied agent data.

Examples:

- wind speed
- wind gust
- temperature
- visibility
- precipitation probability
- precipitation
- wave height
- wave period
- wave direction
- sea-surface temperature
- tide height
- tide time
- tide status
- cyclone distance
- cyclone position
- cyclone wind speed
- PFZ latitude
- PFZ longitude
- distance from shore
- nearest port distance
- GIS information
- restricted-zone information
- any other numerical or factual value supplied by an agent

Rules:

1. If the requested value exists, display the ACTUAL value.

2. Do NOT replace an available value with vague language.

Bad:
"Wind conditions are moderate."

Good:
"Wind speed is 5.2 m/s with gusts up to 12 m/s."

3. NEVER invent values.

4. If a specifically requested value is null, omit that value.

5. If one requested value is available and another is null, display the
   available value and continue the assessment.

6. Preserve the units supplied by the agent.

7. If no unit is supplied, do not invent a unit.

8. Keep the answer concise.

=========================================================
GENERAL ASSESSMENT RULES
=========================================================

- Do not invent data.
- Use all meaningful data that is actually supplied.
- Partial agent data is acceptable.
- Prioritize safety.
- Explain the main reasons for the recommendation.
- If dangerous conditions are present, clearly warn the user.
- Do not increase risk merely because some fields are null.
- risk_level MUST match overall_rule_based_severity when it is not
  "insufficient_data".
- If overall_rule_based_severity is "insufficient_data", determine the
  best-supported risk level from the actual available marine evidence.
- Only mention insufficient data when essentially NO meaningful marine
  intelligence is available.
- Partial agent data is NOT a failure.
- Do not describe null cyclone data as insufficient data.

=========================================================
USER QUESTION
=========================================================

{user_question}

=========================================================
RULE-BASED RISK
=========================================================

{risk_signals}

=========================================================
MARINE AGENT DATA
=========================================================

{agent_data}

=========================================================
OUTPUT REQUIREMENTS
=========================================================

Return ONLY one valid JSON object.

Do NOT:

- use markdown
- use code fences
- return a JSON-encoded string
- return explanatory text before or after JSON
- include agent_findings in the generated JSON

The JSON must contain exactly these main fields:

{
  "summary": "One short overall assessment",
  "risk_level": "LOW",
  "recommendation": "One clear recommendation",
  "key_findings": ["finding 1", "finding 2"],
  "safety_advice": ["advice 1", "advice 2"]
}

Rules:

- summary = one sentence
- recommendation = one sentence
- maximum 2 key_findings
- maximum 2 safety_advice items
- risk_level must be exactly one of:

LOW
MODERATE
HIGH
SEVERE

IMPORTANT:

If the user explicitly requested factual values, include those actual values
inside summary and/or key_findings.

Do not mention null values unless directly relevant.

Do not say "insufficient data" because one agent is null.

Do not describe partial agent data as a failure.

=========================================================
FINAL CHECK BEFORE RESPONDING
=========================================================

Before returning the JSON:

1. Confirm that the response is a normal JSON object.
2. Confirm that risk_level is LOW, MODERATE, HIGH, or SEVERE.
3. Confirm that no invented values are present.
4. Confirm that requested factual values are included when available.
5. Confirm that cyclone=null is NOT described as insufficient data.
6. Confirm that partial data is NOT described as a retrieval failure.
7. Return ONLY the JSON object.
"""
)


# =========================================================
# HELPERS
# =========================================================

def _clean_model_content(content):
    """
    Convert Gemini response content into plain text.

    Handles:
    - normal strings
    - Gemini/LangChain content lists
    - accidental markdown code fences
    """

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))

        content = "".join(parts)

    if content is None:
        return ""

    content = str(content).strip()

    # Remove markdown fences if Gemini adds them.
    if content.startswith("```json"):
        content = content[7:].strip()
    elif content.startswith("```"):
        content = content[3:].strip()

    if content.endswith("```"):
        content = content[:-3].strip()

    return content.strip()


def _try_json_load(content):
    """
    Try several safe ways to parse Gemini's response.
    """

    if not content:
        return None

    candidates = [content]

    # Sometimes a model returns:
    #
    # {\n "summary": "..."\n}
    #
    # as literal escaped characters.
    normalized = (
        content
        .replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\r", "\r")
        .replace("\\t", "\t")
    )

    if normalized != content:
        candidates.append(normalized)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)

            # Handle a JSON string containing another JSON object.
            if isinstance(parsed, str):
                try:
                    parsed = json.loads(parsed)
                except json.JSONDecodeError:
                    pass

            if isinstance(parsed, dict):
                return parsed

        except (json.JSONDecodeError, TypeError):
            continue

    return None


def _extract_json_object(content):
    """
    Extract the first balanced JSON object if Gemini adds
    surrounding text around the JSON.
    """

    if not content:
        return None

    # First try exact parsing.
    parsed = _try_json_load(content)

    if parsed is not None:
        return parsed

    normalized = (
        content
        .replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\r", "\r")
        .replace("\\t", "\t")
    )

    candidates = [content]

    if normalized != content:
        candidates.append(normalized)

    for candidate in candidates:

        for match in re.finditer(r"\{", candidate):

            start = match.start()

            depth = 0
            in_string = False
            escaped = False

            for index in range(start, len(candidate)):

                char = candidate[index]

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

                        possible_json = candidate[
                            start:index + 1
                        ]

                        try:
                            parsed = json.loads(
                                possible_json
                            )

                            if isinstance(parsed, dict):
                                return parsed

                        except json.JSONDecodeError:
                            pass

                        break

    return None


def _has_meaningful_data(agent_data):
    """
    Determine whether at least one marine agent has usable data.

    A single non-null/non-empty agent is enough to perform
    a normal assessment.
    """

    for value in agent_data.values():

        if value is None:
            continue

        if value == "":
            continue

        if value == {}:
            continue

        if value == []:
            continue

        # A non-empty dictionary/list/string is meaningful enough
        # to allow the recommendation agent to assess it.
        return True

    return False


# =========================================================
# RECOMMENDATION NODE
# =========================================================

def recommendation_node(state):
    """
    Final Marine Intelligence Recommendation Agent.

    Combines:
    - Weather
    - Ocean
    - Tide
    - Cyclone
    - Ecosystem
    - PFZ
    - GIS
    - Rule-based risk

    Complete specialist data is preserved in agent_findings.
    """

    user_question = str(
        state.get("user_question", "")
    ).strip()

    risk_signals = state.get("risk_signals") or {}

    # -----------------------------------------------------
    # Collect ALL specialist outputs
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

    # -----------------------------------------------------
    # Debug information
    # -----------------------------------------------------

    print(
        "[recommendation_node] agent_data_presence="
        + json.dumps(
            {
                key: value is not None
                for key, value in agent_data.items()
            }
        )
    )

    print(
        "[recommendation_node] meaningful_data="
        f"{_has_meaningful_data(agent_data)}"
    )

    plan_debug = state.get("plan")

    if isinstance(plan_debug, dict):

        print(
            "[recommendation_node] plan_rejected="
            f"{plan_debug.get('rejected')} "
            f"required_agents="
            f"{plan_debug.get('required_agents')}"
        )

    try:

        # -------------------------------------------------
        # Convert data directly to JSON for Gemini
        # -------------------------------------------------

        agent_data_json = json.dumps(
            agent_data,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

        risk_json = json.dumps(
            risk_signals,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

        # -------------------------------------------------
        # Build prompt
        # -------------------------------------------------

        prompt_text = RECOMMENDATION_PROMPT.format(
            user_question=user_question,
            agent_data=agent_data_json,
            risk_signals=risk_json,
        )

        print(
            "[recommendation_node] "
            f"prompt_chars={len(prompt_text)}"
        )

        # -------------------------------------------------
        # Gemini call
        # -------------------------------------------------

        response = recommendation_llm.invoke(
            prompt_text
        )

        content = _clean_model_content(
            getattr(response, "content", "")
        )

        print(
            "[recommendation_node] "
            f"raw_response_chars={len(content)}"
        )

        print(
            "[recommendation_node] "
            f"raw_response={repr(content[:5000])}"
        )

        # -------------------------------------------------
        # Empty response
        # -------------------------------------------------

        if not content:

            raise ValueError(
                "Gemini recommendation model returned "
                "an empty response."
            )

        # -------------------------------------------------
        # Parse Gemini JSON
        # -------------------------------------------------

        recommendation = _extract_json_object(
            content
        )

        if recommendation is None:

            raise ValueError(
                "Gemini recommendation model returned "
                "non-JSON content: "
                f"{content[:2000]!r}"
            )

        # -------------------------------------------------
        # Required fields
        # -------------------------------------------------

        recommendation.setdefault(
            "summary",
            "Marine assessment completed.",
        )

        recommendation.setdefault(
            "risk_level",
            "MODERATE",
        )

        recommendation.setdefault(
            "recommendation",
            "Review the available marine data before proceeding.",
        )

        recommendation.setdefault(
            "key_findings",
            [],
        )

        recommendation.setdefault(
            "safety_advice",
            [],
        )

        # -------------------------------------------------
        # Normalize risk level
        # -------------------------------------------------

        risk_level = str(
            recommendation.get(
                "risk_level",
                "MODERATE",
            )
        ).upper().strip()

        if risk_level not in {
            "LOW",
            "MODERATE",
            "HIGH",
            "SEVERE",
        }:

            risk_level = "MODERATE"

        recommendation["risk_level"] = risk_level

        # -------------------------------------------------
        # Normalize arrays
        # -------------------------------------------------

        if not isinstance(
            recommendation.get("key_findings"),
            list,
        ):
            recommendation["key_findings"] = []

        if not isinstance(
            recommendation.get("safety_advice"),
            list,
        ):
            recommendation["safety_advice"] = []

        # Maximum two items as requested.
        recommendation["key_findings"] = (
            recommendation["key_findings"][:2]
        )

        recommendation["safety_advice"] = (
            recommendation["safety_advice"][:2]
        )

        # -------------------------------------------------
        # Preserve complete specialist data
        # -------------------------------------------------

        recommendation["agent_findings"] = agent_data

        # -------------------------------------------------
        # Pydantic validation
        # -------------------------------------------------

        try:

            Recommendation.model_validate(
                recommendation
            )

        except ValidationError as ve:

            print(
                "[recommendation_node] "
                f"Schema warning: {ve}"
            )

            # Do not destroy a valid generated assessment
            # because of an optional schema mismatch.
            recommendation["schema_warning"] = str(ve)

        # -------------------------------------------------
        # SUCCESS
        # -------------------------------------------------

        print(
            "[recommendation_node] "
            "Recommendation generated successfully."
        )

        return {
            "recommendation": recommendation,
            "status": "SUCCESS",
        }

    # =====================================================
    # FAILURE
    # =====================================================

    except Exception as e:

        print(
            "[recommendation_node] "
            f"Recommendation generation failed: {e}"
        )

        return {
            "recommendation": {
                "summary": (
                    "The final assessment could not be fully generated."
                ),
                "risk_level": "MODERATE",
                "recommendation": (
                    "Please review the available marine data "
                    "before proceeding."
                ),
                "key_findings": [],
                "safety_advice": [],
                "agent_findings": agent_data,
                "error": str(e),
            },
            "status": "FAILED",
        }
