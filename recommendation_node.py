# recommendation_node.py
#
# Groq 8K-friendly final recommendation node.
# - Uses openai/gpt-oss-120b by default.
# - Keeps the prompt deliberately small.
# - Does NOT use response_format/json_schema.
# - Parses normal JSON and JSON containing literal escaped newlines.
# - Safely extracts JSON if the model adds surrounding text.
# - Keeps complete agent data in the returned recommendation.

import json
import os
import re

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from schemas import Recommendation

print("🔥 NEW RECOMMENDATION NODE LOADED")
print("FILE:", __file__)


# =========================================================
# LLM
# =========================================================

recommendation_llm = ChatGroq(
    model=os.getenv(
        "GROQ_MODEL_RECOMENDATION",
        os.getenv(
            "GROQ_MODEL_RECOMMENDATION",
            "openai/gpt-oss-120b",
        ),
    ),
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
    max_retries=2,
    reasoning_effort="low",
    include_reasoning=False,
    max_completion_tokens=400,
)


# =========================================================
# PROMPT
# =========================================================

RECOMMENDATION_PROMPT = ChatPromptTemplate.from_template(
    """
You are the Final Marine Intelligence and Recommendation Agent.

Answer the user's current question using ONLY the supplied marine agent data
and rule-based risk assessment.

IMPORTANT DATA INTERPRETATION:
- NULL values are VALID and are NOT automatically "insufficient data".
- A null field means there is no usable finding for that particular field.
- Ignore individual null fields and continue the assessment using all other
  available evidence.
- NEVER say "insufficient data" merely because one or more fields are null.
- NEVER invent a value for a null field.
- Do not treat a null value as an error.

CYCLONE-SPECIFIC RULE:
- If cyclone data is null, interpret it as:
  "No cyclone threat is indicated by the available cyclone assessment."
- Do NOT say:
  "insufficient cyclone data",
  "cyclone data unavailable",
  or "unable to assess cyclone conditions"
  simply because cyclone data is null.
- Do not invent cyclone information.

OTHER AGENT NULL VALUES:
- weather = null → ignore that weather finding and use other available data.
- ocean = null → ignore that ocean finding and use other available data.
- tide = null → ignore that tide finding and use other available data.
- cyclone = null → no cyclone threat is indicated by the available assessment.
- ecosystem = null → ignore that ecosystem finding and use other available data.
- pfz = null → ignore that PFZ finding and use other available data.
- gis = null → ignore that GIS finding and use other available data.

VALUE-DISPLAY RULE:
- If the user explicitly asks for specific values, measurements, coordinates,
  times, conditions, or other factual data, include the requested values from
  the supplied marine agent data in the response.
- Examples include:
  - wind speed
  - wind gusts
  - temperature
  - visibility
  - precipitation probability
  - wave height
  - wave period
  - sea-surface temperature
  - tide height
  - tide time
  - tide status
  - cyclone distance
  - cyclone position
  - cyclone wind speed
  - PFZ latitude/longitude
  - distance from shore
  - port distance
  - GIS/restricted-zone information
  - any other numerical or factual value present in the supplied agent data.
- If the requested value exists in the supplied data, display the ACTUAL value.
- Do NOT replace an available value with a vague statement such as
  "conditions are moderate".
- Do NOT invent a value that is not present in the supplied data.
- If a specifically requested value is null, omit that value rather than
  inventing it or calling the entire assessment "insufficient data".
- If multiple values are requested and some are available while others are
  null, display the available values and continue the assessment normally.
- Preserve the units provided by the agent data.
- If the data has no unit, do not invent one.
- Keep the response concise while still including explicitly requested values.

GENERAL RULES:
- Do not invent data.
- Use all meaningful data that is actually supplied.
- Partial agent data is acceptable.
- Prioritize safety.
- Explain the main reasons for the recommendation.
- If dangerous conditions are present, clearly warn the user.
- Do not increase the risk level merely because some data fields are null.
- risk_level MUST match overall_rule_based_severity when it is not
  "insufficient_data".
- If overall_rule_based_severity is "insufficient_data", use the available
  marine evidence to determine the best supported risk_level.
- Only mention insufficient data when essentially NO meaningful marine
  intelligence is available to make a reasonable assessment.
- Keep the response concise.

USER QUESTION:
{user_question}

RULE-BASED RISK:
{risk_signals}

MARINE AGENT DATA:
{agent_data}

OUTPUT REQUIREMENTS:
- Return ONLY one valid JSON object.
- Do not use markdown.
- Do not wrap the JSON in code fences.
- Do not return a JSON-encoded string.
- Return a normal JSON object, not a quoted JSON string.
- Do NOT include agent_findings in your JSON.
- Keep summary to one sentence.
- Keep recommendation to one sentence.
- Return at most 2 key_findings.
- Return at most 2 safety_advice items.

JSON FORMAT:
{{
  "summary": "One short overall assessment",
  "risk_level": "LOW",
  "recommendation": "One clear recommendation",
  "key_findings": ["finding 1", "finding 2"],
  "safety_advice": ["advice 1", "advice 2"]
}}

IMPORTANT:
- The summary may contain explicitly requested values when necessary.
- key_findings should contain requested factual values when appropriate.
- recommendation should remain actionable.
- Do not mention null values unless directly relevant.
- Do not describe null cyclone data as insufficient data.
- Do not describe partial agent data as a failure.

risk_level must be exactly one of:
LOW, MODERATE, HIGH, SEVERE
"""
)


# =========================================================
# PROMPT-SIZE LIMITS
# =========================================================

MAX_LIST_LEN = 3
MAX_COMPACT_DEPTH = 4
MAX_AGENT_DATA_CHARS = 10000
MAX_QUESTION_CHARS = 1800
MAX_RISK_CHARS = 2500


# =========================================================
# HELPERS
# =========================================================

def _compact_for_llm(
    obj,
    max_list_len=MAX_LIST_LEN,
    max_depth=MAX_COMPACT_DEPTH,
    _depth=0,
):
    """
    Recursively shrink long lists and deeply nested structures.
    """
    if _depth >= max_depth:
        if isinstance(obj, (dict, list)):
            return str(obj)
        return obj

    if isinstance(obj, dict):
        return {
            key: _compact_for_llm(
                value,
                max_list_len,
                max_depth,
                _depth + 1,
            )
            for key, value in obj.items()
        }

    if isinstance(obj, list):
        if len(obj) > max_list_len:
            kept = obj[: max_list_len - 1]

            compacted = [
                _compact_for_llm(
                    item,
                    max_list_len,
                    max_depth,
                    _depth + 1,
                )
                for item in kept
            ]

            omitted = len(obj) - len(kept)

            compacted.append(
                f"...{omitted} more entries omitted for brevity..."
            )

            return compacted

        return [
            _compact_for_llm(
                item,
                max_list_len,
                max_depth,
                _depth + 1,
            )
            for item in obj
        ]

    return obj


def _safe_json_text(obj, max_chars):
    """
    Serialize to compact JSON.

    If the serialized JSON is too large, return a valid small JSON object
    rather than cutting JSON in the middle and creating invalid syntax.
    """
    text = json.dumps(
        obj,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )

    if len(text) <= max_chars:
        return text

    return json.dumps(
        {
            "truncated": True,
            "message": (
                "Agent data was truncated to protect "
                "the LLM token budget."
            ),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _clean_model_content(content):
    """
    Convert LangChain/Groq content to plain text and remove
    accidental markdown code fences.
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

    if content.startswith("```json"):
        content = content[7:].strip()
    elif content.startswith("```"):
        content = content[3:].strip()

    if content.endswith("```"):
        content = content[:-3].strip()

    return content.strip()


def _normalize_json_text(content):
    """
    Fix the specific format returned by the model in which the JSON
    contains literal escape sequences such as:

        {\\n "summary": "...",\\n "risk_level": "LOW"}

    This converts those formatting escapes into actual whitespace before
    attempting json.loads().
    """
    normalized = content

    normalized = normalized.replace("\\r\\n", "\n")
    normalized = normalized.replace("\\n", "\n")
    normalized = normalized.replace("\\r", "\r")
    normalized = normalized.replace("\\t", "\t")

    return normalized


def _extract_json_object(content):
    """
    Extract the first balanced JSON object from model output.

    Handles:
    - normal JSON
    - JSON with surrounding prose
    - JSON with literal escaped newlines
    """
    if not content:
        return None

    # -----------------------------------------------------
    # Attempt 1: exact JSON
    # -----------------------------------------------------
    candidates = [content]

    normalized = _normalize_json_text(content)

    if normalized != content:
        candidates.append(normalized)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)

            # Handle a JSON string that itself contains JSON.
            if isinstance(parsed, str):
                try:
                    parsed = json.loads(parsed)
                except json.JSONDecodeError:
                    pass

            if isinstance(parsed, dict):
                return parsed

        except (json.JSONDecodeError, TypeError):
            pass

    # -----------------------------------------------------
    # Attempt 2: find a balanced JSON object
    # -----------------------------------------------------
    for candidate in candidates:
        for start_match in re.finditer(r"\{", candidate):
            start = start_match.start()

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
                        candidate_json = candidate[start:index + 1]

                        try:
                            parsed = json.loads(candidate_json)

                            if isinstance(parsed, dict):
                                return parsed

                        except json.JSONDecodeError:
                            pass

                        break

    return None


# =========================================================
# RECOMMENDATION NODE
# =========================================================

def recommendation_node(state):
    """
    Final Recommendation Agent.

    Combines specialist outputs and rule-based risk assessment.
    Only compact data is sent to the LLM.
    Complete specialist data remains available in the returned
    recommendation.
    """

    user_question = str(
        state.get("user_question", "")
    ).strip()[:MAX_QUESTION_CHARS]

    risk_signals = state.get("risk_signals") or {}

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
        # Compact specialist data
        # -------------------------------------------------

        compact_agent_data = _compact_for_llm(agent_data)

        agent_data_json = _safe_json_text(
            compact_agent_data,
            MAX_AGENT_DATA_CHARS,
        )

        risk_json = _safe_json_text(
            risk_signals,
            MAX_RISK_CHARS,
        )

        # -------------------------------------------------
        # Build ONE prompt and reuse it
        # -------------------------------------------------

        prompt_text = RECOMMENDATION_PROMPT.format(
            user_question=user_question,
            agent_data=agent_data_json,
            risk_signals=risk_json,
        )

        print(
            f"[recommendation_node] prompt_chars={len(prompt_text)}"
        )

        # -------------------------------------------------
        # Call LLM
        # -------------------------------------------------

        response = recommendation_llm.invoke(prompt_text)

        content = _clean_model_content(
            getattr(response, "content", "")
        )

        print(
            "[recommendation_node] "
            f"raw_response_chars={len(content)}"
        )

        print(
            "[recommendation_node] "
            f"raw_response={repr(content[:4000])}"
        )

        # -------------------------------------------------
        # Parse JSON
        # -------------------------------------------------

        if not content:
            raise ValueError(
                "Recommendation model returned an empty response."
            )

        recommendation = _extract_json_object(content)

        if recommendation is None:
            raise ValueError(
                "Recommendation model returned non-JSON content: "
                f"{content[:1200]!r}"
            )

        # -------------------------------------------------
        # Ensure required fields exist
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

        # Always keep the complete specialist outputs available
        # to the caller/UI.
        recommendation["agent_findings"] = agent_data

        # -------------------------------------------------
        # Normalize risk level
        # -------------------------------------------------

        risk_level = str(
            recommendation.get("risk_level", "MODERATE")
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
        # Schema validation
        # -------------------------------------------------

        try:
            Recommendation.model_validate(recommendation)

        except ValidationError as ve:
            # Do not fail the entire graph because of an optional
            # schema mismatch. Preserve the generated recommendation.
            recommendation["schema_warning"] = str(ve)

        return {
            "recommendation": recommendation,
            "status": "SUCCESS",
        }

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
