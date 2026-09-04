# recommendation_node.py
#
# ORCA Marine Intelligence - Final Recommendation Agent
# - Uses Groq GPT-OSS 120B by default.
# - Keeps prompts compact for the 8K TPM limit.
# - Does not use response_format/json_schema.
# - Robustly parses normal JSON, fenced JSON, and JSON returned as a string.
# - Never slices raw JSON into malformed JSON.
# - Keeps complete specialist data in agent_findings after synthesis.

import ast
import json
import os
import re

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from schemas import Recommendation


# ============================================================
# LLM
# ============================================================

recommendation_llm = ChatGroq(
    model=os.getenv(
        "GROQ_MODEL_RECOMMENDATION",
        os.getenv(
            "GROQ_MODEL_RECOMENDATION",
            "openai/gpt-oss-120b",
        ),
    ),
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
    max_retries=2,
    reasoning_effort="low",
    include_reasoning=False,
    max_completion_tokens=350,
)


# ============================================================
# PROMPT
# ============================================================

RECOMMENDATION_PROMPT = ChatPromptTemplate.from_template(
    """
You are the Final Marine Intelligence Agent.

Answer the user's CURRENT question using only the supplied marine data and
rule-based risk information.

Rules:
- Do not invent facts.
- Use only data that is present.
- Missing/null agent data is unavailable.
- Prioritize safety.
- risk_level must be exactly LOW, MODERATE, HIGH, or SEVERE.
- If overall_rule_based_severity is available and is not
  "insufficient_data", use that severity.
- If it is "insufficient_data", select the safest justified level from the
  available evidence and state that automated risk data was incomplete.
- Keep the answer concise.
- key_findings: maximum 3 short items.
- safety_advice: maximum 3 short items.
- agent_findings should contain only very short summaries, not full raw data.
- Return ONLY a JSON object. No markdown. No text before or after JSON.

USER QUESTION:
{user_question}

RULE-BASED RISK:
{risk_signals}

MARINE DATA:
{agent_data}

Return exactly:
{{
  "summary": "Short overall assessment",
  "risk_level": "MODERATE",
  "recommendation": "Clear actionable recommendation",
  "key_findings": ["finding 1"],
  "safety_advice": ["advice 1"],
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
"""
)


# ============================================================
# SIZE LIMITS
# ============================================================

MAX_QUESTION_CHARS = 1800
MAX_AGENT_DATA_CHARS = 7000
MAX_RISK_CHARS = 1800
MAX_LIST_LEN = 3
MAX_COMPACT_DEPTH = 3


# ============================================================
# HELPERS
# ============================================================

def _compact_for_llm(obj, depth=0):
    """Reduce nested specialist payloads without changing the source data."""

    if depth >= MAX_COMPACT_DEPTH:
        if isinstance(obj, (dict, list)):
            return "[nested data omitted]"
        return obj

    if isinstance(obj, dict):
        result = {}

        for key, value in obj.items():
            result[key] = _compact_for_llm(value, depth + 1)

        return result

    if isinstance(obj, list):
        items = obj[:MAX_LIST_LEN]

        result = [
            _compact_for_llm(item, depth + 1)
            for item in items
        ]

        if len(obj) > MAX_LIST_LEN:
            result.append(
                f"...{len(obj) - MAX_LIST_LEN} more entries omitted..."
            )

        return result

    return obj


def _safe_json_text(obj, max_chars):
    """
    Return valid JSON that is below the requested character budget.

    Never cut an existing JSON document in the middle because that creates
    invalid JSON and can cause downstream failures.
    """

    text = json.dumps(
        obj,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )

    if len(text) <= max_chars:
        return text

    # Compact again with a much smaller structure.
    compact = _compact_for_llm(obj, depth=0)

    text = json.dumps(
        compact,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )

    if len(text) <= max_chars:
        return text

    # Final guaranteed-valid fallback.
    return json.dumps(
        {
            "truncated": True,
            "message": "Marine data was compacted to protect the model token budget.",
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _content_to_text(content):
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


def _unwrap_string_json(text):
    """
    Handle model output like:

        '{\\n "summary": "...", ... }'

    and convert it into the actual JSON text:

        {
          "summary": "..."
        }
    """

    text = text.strip()

    if len(text) < 2:
        return text

    if text[0] not in ("'", '"') or text[-1] != text[0]:
        return text

    # Python-style string wrapper.
    try:
        value = ast.literal_eval(text)

        if isinstance(value, str):
            return value.strip()

        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)

    except (ValueError, SyntaxError):
        pass

    # JSON string wrapper.
    try:
        value = json.loads(text)

        if isinstance(value, str):
            return value.strip()

    except json.JSONDecodeError:
        pass

    return text


def _extract_json_object(text):
    """Parse normal/fenced/wrapped JSON or extract a balanced JSON object."""

    text = _content_to_text(text)

    if not text:
        raise ValueError(
            "Recommendation model returned an empty response."
        )

    # Remove markdown fences.
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*```$", "", text).strip()

    # Unwrap quoted JSON.
    text = _unwrap_string_json(text)

    # Try complete JSON.
    try:
        value = json.loads(text)

        if isinstance(value, dict):
            return value

        if isinstance(value, str):
            text = value.strip()

    except json.JSONDecodeError:
        pass

    # Try Python literal dictionary as a last wrapper case.
    try:
        value = ast.literal_eval(text)

        if isinstance(value, dict):
            return value

        if isinstance(value, str):
            inner = value.strip()
            value = json.loads(inner)

            if isinstance(value, dict):
                return value

    except (ValueError, SyntaxError, json.JSONDecodeError):
        pass

    # Extract the first balanced JSON object.
    for start in [m.start() for m in re.finditer(r"\{", text)]:
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
        "Recommendation model returned non-JSON content: "
        f"{text[:1600]!r}"
    )


def _normalize_list(value, max_items=3):
    if not isinstance(value, list):
        return []

    result = []

    for item in value:
        item = str(item).strip()

        if item and item not in result:
            result.append(item)

        if len(result) >= max_items:
            break

    return result


def _risk_from_signals(risk_signals):
    """
    Recover a useful risk level from the rule-based state when the LLM fails.

    Supports the common keys used by ORCA and also searches nested dictionaries.
    """

    allowed = {"LOW", "MODERATE", "HIGH", "SEVERE"}

    def search(obj):
        if isinstance(obj, dict):
            for key in (
                "overall_rule_based_severity",
                "risk_level",
                "severity",
                "overall_severity",
            ):
                value = str(obj.get(key, "")).upper().strip()

                if value in allowed:
                    return value

            for value in obj.values():
                found = search(value)

                if found:
                    return found

        elif isinstance(obj, list):
            for value in obj:
                found = search(value)

                if found:
                    return found

        return None

    return search(risk_signals) or "MODERATE"


# ============================================================
# NODE
# ============================================================

def recommendation_node(state):
    """Synthesize specialist outputs into the final user-facing result."""

    user_question = str(
        state.get("user_question") or ""
    ).strip()[:MAX_QUESTION_CHARS]

    risk_signals = state.get("risk_signals") or {}

    # Keep the complete data untouched for the UI/state.
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
        compact_agent_data = _compact_for_llm(agent_data)

        agent_data_json = _safe_json_text(
            compact_agent_data,
            MAX_AGENT_DATA_CHARS,
        )

        risk_json = _safe_json_text(
            risk_signals,
            MAX_RISK_CHARS,
        )

        prompt_text = RECOMMENDATION_PROMPT.format(
            user_question=user_question,
            risk_signals=risk_json,
            agent_data=agent_data_json,
        )

        print(
            f"[recommendation_node] prompt_chars={len(prompt_text)}"
        )

        response = recommendation_llm.invoke(prompt_text)

        content = _content_to_text(response.content)

        print(
            f"[recommendation_node] raw_response_chars={len(content)}"
        )
        print(
            f"[recommendation_node] raw_response={content[:3000]!r}"
        )

        recommendation = _extract_json_object(content)

        # --------------------------------------------------------
        # Normalize final fields
        # --------------------------------------------------------

        recommendation["summary"] = str(
            recommendation.get(
                "summary",
                "Marine assessment completed.",
            )
        ).strip()

        risk_level = str(
            recommendation.get(
                "risk_level",
                _risk_from_signals(risk_signals),
            )
        ).upper().strip()

        if risk_level not in {"LOW", "MODERATE", "HIGH", "SEVERE"}:
            risk_level = _risk_from_signals(risk_signals)

        recommendation["risk_level"] = risk_level

        recommendation["recommendation"] = str(
            recommendation.get(
                "recommendation",
                "Review the available marine conditions before proceeding.",
            )
        ).strip()

        recommendation["key_findings"] = _normalize_list(
            recommendation.get("key_findings", [])
        )

        recommendation["safety_advice"] = _normalize_list(
            recommendation.get("safety_advice", [])
        )

        # The complete raw agent data belongs in application state, not in
        # the model's small response.
        recommendation["agent_findings"] = agent_data

        # Validate when possible, but do not destroy a valid final answer
        # because of optional schema differences.
        try:
            Recommendation.model_validate(recommendation)
        except ValidationError as exc:
            recommendation["schema_warning"] = str(exc)

        return {
            "recommendation": recommendation,
            "status": "SUCCESS",
        }

    except Exception as exc:
        import traceback
        traceback.print_exc()

        fallback_risk = _risk_from_signals(risk_signals)

        return {
            "recommendation": {
                "summary": "The final assessment could not be fully generated.",
                "risk_level": fallback_risk,
                "recommendation": (
                    "Please review the available marine data before proceeding."
                ),
                "key_findings": [],
                "safety_advice": [],
                "agent_findings": agent_data,
                "error": str(exc),
            },
            "status": "FAILED",
        }
