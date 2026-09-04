# recommendation_node.py
#
# Final Marine Intelligence Recommendation Agent.
#
# Important:
# - Uses GPT-OSS 120B by default.
# - Uses low reasoning to preserve completion budget.
# - Does NOT use Groq response_format/json_schema.
# - Locally parses and validates JSON.
# - Handles normal JSON, fenced JSON, surrounding text,
#   and double-encoded JSON.
# - Keeps complete specialist data in the returned recommendation.
# - Keeps the LLM prompt compact for the Groq 8K TPM limit.

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
        "openai/gpt-oss-120b",
    ),
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
    max_retries=2,

    # GPT-OSS is a reasoning model.
    # Low reasoning leaves enough budget for the JSON answer.
    reasoning_effort="low",

    # Do not include reasoning in response.content.
    include_reasoning=False,

    # Give the model enough room for the final JSON.
    max_completion_tokens=300,
)


# =========================================================
# PROMPT
# =========================================================

RECOMMENDATION_PROMPT = ChatPromptTemplate.from_template(
    """
You are the Final Marine Intelligence and Recommendation Agent.

Answer the user's current question using ONLY the supplied marine agent data
and rule-based risk assessment.

RULES:
- Do not invent data.
- Use only agents whose data is present.
- Missing/null agent data means that agent is unavailable.
- Prioritize safety.
- Explain the main reasons for the recommendation.
- If dangerous conditions are present, clearly warn the user.
- risk_level MUST match overall_rule_based_severity when it is not
  "insufficient_data".
- If overall_rule_based_severity is "insufficient_data", choose risk_level
  from the available data and say that the automated risk check had
  insufficient data.
- Keep the response concise.
- Return ONLY one valid JSON object.
- No markdown.
- No explanation outside JSON.
- Do NOT wrap the JSON object inside a JSON string.
- Do NOT escape the entire JSON object.

USER QUESTION:
{user_question}

RULE-BASED RISK:
{risk_signals}

MARINE AGENT DATA:
{agent_data}

JSON FORMAT:
{{
  "summary": "Short overall assessment",
  "risk_level": "LOW",
  "recommendation": "Clear recommendation",
  "key_findings": ["finding 1", "finding 2"],
  "safety_advice": ["advice 1", "advice 2"],
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

risk_level must be exactly one of:
LOW, MODERATE, HIGH, SEVERE
"""
)


# =========================================================
# PROMPT SIZE LIMITS
# =========================================================

MAX_LIST_LEN = 3
MAX_COMPACT_DEPTH = 4

# Keep this conservative for the 8K TPM limit.
MAX_AGENT_DATA_CHARS = 8000

MAX_QUESTION_CHARS = 1800

MAX_RISK_CHARS = 2000


# =========================================================
# DATA COMPACTION
# =========================================================

def _compact_for_llm(
    obj,
    max_list_len=MAX_LIST_LEN,
    max_depth=MAX_COMPACT_DEPTH,
    _depth=0,
):
    """
    Recursively reduce large specialist outputs before
    sending them to the LLM.
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

            kept_count = max(1, max_list_len - 1)

            compacted = [
                _compact_for_llm(
                    item,
                    max_list_len,
                    max_depth,
                    _depth + 1,
                )
                for item in obj[:kept_count]
            ]

            omitted = len(obj) - kept_count

            compacted.append(
                f"...{omitted} more entries omitted..."
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


# =========================================================
# SAFE JSON SERIALIZATION
# =========================================================

def _safe_json_text(obj, max_chars):
    """
    Serialize data as valid JSON.

    IMPORTANT:
    Never cut raw JSON at an arbitrary character position because
    that can produce malformed JSON.

    If the serialized data is too large, return a valid compact
    JSON object describing the truncation.
    """

    text = json.dumps(
        obj,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )

    if len(text) <= max_chars:
        return text

    # Return VALID JSON instead of slicing the original JSON.
    return json.dumps(
        {
            "_truncated": True,
            "_message": (
                "Specialist data was reduced because it exceeded "
                "the recommendation model input budget."
            ),
            "_data_preview": text[: max(500, max_chars // 4)],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


# =========================================================
# CLEAN MODEL CONTENT
# =========================================================

def _clean_model_content(content):
    """
    Convert LangChain response content into a plain string.

    Handles content blocks and markdown fences.
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

    content = str(content or "").strip()

    # Remove ```json ... ```
    if content.startswith("```json"):

        content = content[len("```json"):].strip()

        if content.endswith("```"):
            content = content[:-3].strip()

    # Remove ``` ... ```
    elif content.startswith("```"):

        content = content[3:].strip()

        if content.endswith("```"):
            content = content[:-3].strip()

    return content.strip()


# =========================================================
# ROBUST JSON PARSER
# =========================================================

def _parse_recommendation_json(content):
    """
    Parse recommendation JSON robustly.

    Supports:

    1. Normal JSON object:
       {"risk_level":"HIGH",...}

    2. JSON surrounded by text:
       Here is the result:
       {"risk_level":"HIGH",...}

    3. Double-encoded JSON:
       "{\"risk_level\":\"HIGH\",...}"

    4. JSON returned as a quoted string containing JSON.

    5. Markdown fenced JSON.
    """

    content = _clean_model_content(content)

    print(
        "[recommendation_node] "
        f"cleaned_response_chars={len(content)}"
    )

    print(
        "[recommendation_node] "
        f"cleaned_response={repr(content[:4000])}"
    )

    if not content:
        raise ValueError(
            "Recommendation model returned an empty response."
        )

    # -----------------------------------------------------
    # Attempt 1:
    # Direct JSON parsing
    # -----------------------------------------------------

    try:

        parsed = json.loads(content)

        # Normal expected case.
        if isinstance(parsed, dict):
            return parsed

        # -------------------------------------------------
        # Double-encoded JSON
        #
        # Example:
        # "{\"risk_level\":\"HIGH\",...}"
        # -------------------------------------------------

        if isinstance(parsed, str):

            inner = parsed.strip()

            if inner:

                try:
                    inner_parsed = json.loads(inner)

                    if isinstance(inner_parsed, dict):
                        return inner_parsed

                except json.JSONDecodeError:
                    pass

    except json.JSONDecodeError:
        pass

    # -----------------------------------------------------
    # Attempt 2:
    # Find a JSON object embedded in surrounding text.
    # -----------------------------------------------------

    start_positions = [
        match.start()
        for match in re.finditer(r"\{", content)
    ]

    for start in start_positions:

        depth = 0
        in_string = False
        escaped = False

        for index in range(start, len(content)):

            char = content[index]

            # ---------------------------------------------
            # Inside JSON string
            # ---------------------------------------------

            if in_string:

                if escaped:
                    escaped = False

                elif char == "\\":
                    escaped = True

                elif char == '"':
                    in_string = False

                continue

            # ---------------------------------------------
            # Outside JSON string
            # ---------------------------------------------

            if char == '"':
                in_string = True

            elif char == "{":
                depth += 1

            elif char == "}":

                depth -= 1

                if depth == 0:

                    candidate = content[
                        start:index + 1
                    ]

                    try:

                        parsed = json.loads(candidate)

                        if isinstance(parsed, dict):
                            return parsed

                    except json.JSONDecodeError:
                        pass

                    break

    # -----------------------------------------------------
    # Nothing worked.
    # -----------------------------------------------------

    raise ValueError(
        "Recommendation model returned non-JSON content: "
        f"{content[:1200]!r}"
    )


# =========================================================
# RECOMMENDATION NODE
# =========================================================

def recommendation_node(state):
    """
    Final Recommendation Agent.

    Combines:
      - specialist agent outputs
      - rule-based risk assessment
      - current user question

    Only compact data is sent to the LLM.

    Complete specialist data is retained in the final output.
    """

    user_question = str(
        state.get("user_question", "")
    ).strip()[:MAX_QUESTION_CHARS]

    risk_signals = state.get("risk_signals") or {}

    # =====================================================
    # COLLECT SPECIALIST DATA
    # =====================================================

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

        # =================================================
        # COMPACT SPECIALIST DATA
        # =================================================

        compact_agent_data = _compact_for_llm(
            agent_data
        )

        agent_data_json = _safe_json_text(
            compact_agent_data,
            MAX_AGENT_DATA_CHARS,
        )

        risk_json = _safe_json_text(
            risk_signals,
            MAX_RISK_CHARS,
        )

        # =================================================
        # BUILD PROMPT
        # =================================================

        prompt_text = RECOMMENDATION_PROMPT.format(
            user_question=user_question,
            agent_data=agent_data_json,
            risk_signals=risk_json,
        )

        print(
            "[recommendation_node] "
            f"prompt_chars={len(prompt_text)}"
        )

        print(
            "[recommendation_node] "
            f"agent_data_chars={len(agent_data_json)}"
        )

        print(
            "[recommendation_node] "
            f"risk_chars={len(risk_json)}"
        )

        # =================================================
        # CALL LLM
        # =================================================

        response = recommendation_llm.invoke(
            prompt_text
        )

        # =================================================
        # RAW RESPONSE
        # =================================================

        raw_content = response.content

        print(
            "[recommendation_node] "
            f"response_content_type={type(raw_content)}"
        )

        # =================================================
        # PARSE RESPONSE
        # =================================================

        recommendation = _parse_recommendation_json(
            raw_content
        )

        # =================================================
        # ENSURE REQUIRED FIELDS
        # =================================================

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

        # =================================================
        # NORMALIZE LIST FIELDS
        # =================================================

        if not isinstance(
            recommendation.get("key_findings"),
            list,
        ):
            recommendation["key_findings"] = [
                str(
                    recommendation["key_findings"]
                )
            ]

        if not isinstance(
            recommendation.get("safety_advice"),
            list,
        ):
            recommendation["safety_advice"] = [
                str(
                    recommendation["safety_advice"]
                )
            ]

        # =================================================
        # NORMALIZE RISK LEVEL
        # =================================================

        allowed_risk_levels = {
            "LOW",
            "MODERATE",
            "HIGH",
            "SEVERE",
        }

        risk_level = str(
            recommendation.get(
                "risk_level",
                "MODERATE",
            )
        ).upper()

        if risk_level not in allowed_risk_levels:
            risk_level = "MODERATE"

        recommendation["risk_level"] = risk_level

        # =================================================
        # KEEP COMPLETE AGENT DATA
        # =================================================

        recommendation["agent_findings"] = agent_data

        # =================================================
        # SCHEMA VALIDATION
        # =================================================

        try:

            Recommendation.model_validate(
                recommendation
            )

        except ValidationError as ve:

            recommendation["schema_warning"] = str(ve)

            print(
                "[recommendation_node] "
                f"schema warning={str(ve)}"
            )

        # =================================================
        # SUCCESS
        # =================================================

        print(
            "[recommendation_node] "
            f"SUCCESS risk_level={recommendation.get('risk_level')}"
        )

        return {
            "recommendation": recommendation,
            "status": "SUCCESS",
        }

    # =====================================================
    # JSON / PARSING ERROR
    # =====================================================

    except json.JSONDecodeError as e:

        print(
            "[recommendation_node] "
            f"JSON parse error={repr(e)}"
        )

        return {
            "recommendation": {
                "summary": (
                    "Unable to parse the final recommendation."
                ),
                "risk_level": "MODERATE",
                "recommendation": (
                    "Please review the available marine data."
                ),
                "key_findings": [],
                "safety_advice": [],
                "agent_findings": agent_data,
                "error": str(e),
            },
            "status": "FAILED",
        }

    # =====================================================
    # GENERAL ERROR
    # =====================================================

    except Exception as e:

        print(
            "[recommendation_node] "
            "Recommendation generation failed:",
            repr(e),
        )

        return {
            "recommendation": {
                "summary": (
                    "Recommendation generation failed."
                ),
                "risk_level": "MODERATE",
                "recommendation": (
                    "Please review the available marine data."
                ),
                "key_findings": [],
                "safety_advice": [],
                "agent_findings": agent_data,
                "error": str(e),
            },
            "status": "FAILED",
        }
