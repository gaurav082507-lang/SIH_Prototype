# recommendation_node.py
#
# Groq 8K-friendly final recommendation node.
# - Uses openai/gpt-oss-120b by default.
# - Keeps the prompt deliberately small.
# - Does NOT use response_format/json_schema, avoiding Groq JSON validation
#   failures while keeping local JSON parsing + Pydantic validation.
# - Sends only compacted specialist data to the model.
# - Keeps complete agent data in the returned recommendation.
# - Conversation history is intentionally omitted from the LLM prompt to
#   protect the 8K TPM budget. The current user question remains authoritative.

import json
import os
import re

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from schemas import Recommendation
print("🔥 NEW RECOMMENDATION NODE LOADED")
print("FILE:", __file__)

# ---------------------------------------------------------
# LLM
# ---------------------------------------------------------

recommendation_llm = ChatGroq(
    model=os.getenv("GROQ_MODEL_RECOMENDATION", "openai/gpt-oss-120b"),
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
    max_retries=2,
    reasoning_effort="low",
    include_reasoning=False,
    max_completion_tokens=300,
)


# ---------------------------------------------------------
# PROMPT
# ---------------------------------------------------------
# Keep this prompt short. Groq's free limit for GPT-OSS 120B is 8K TPM,
# so the data budget below is intentionally conservative.

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
- Return ONLY one valid JSON object. No markdown. No explanation outside JSON.

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


# ---------------------------------------------------------
# PROMPT-SIZE LIMITS
# ---------------------------------------------------------
# These limits are deliberately conservative for Groq's 8K TPM limit.
# The model sees compacted data only; the full data is still returned.

MAX_LIST_LEN = 3
MAX_COMPACT_DEPTH = 4

# Maximum characters of specialist data sent to the model.
# 10,000 chars is intentionally below the previous 24,000-char budget.
MAX_AGENT_DATA_CHARS = 10000

# Maximum user-question size sent to the model.
MAX_QUESTION_CHARS = 1800

# Maximum risk JSON size sent to the model.
MAX_RISK_CHARS = 2500


def _compact_for_llm(
    obj,
    max_list_len=MAX_LIST_LEN,
    max_depth=MAX_COMPACT_DEPTH,
    _depth=0,
):
    """
    Recursively shrink long lists and deeply nested structures.

    Keeps only a few representative list entries so time-series/grid data
    cannot consume the entire LLM prompt.
    """
    if _depth >= max_depth:
        # At excessive nesting depth, return a compact string representation.
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
    """Serialize an object to compact JSON and cap its size."""
    text = json.dumps(
        obj,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )

    if len(text) <= max_chars:
        return text

    return (
        text[:max_chars]
        + ',"_truncated":"additional data omitted to protect LLM token budget"}'
        if text.startswith("{")
        else text[:max_chars]
        + "\n... [truncated to protect LLM token budget] ..."
    )


def _clean_model_content(content):
    """Remove accidental markdown fences from model output."""
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

    content = str(content).strip()

    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]

    if content.endswith("```"):
        content = content[:-3]

    return content.strip()


# ---------------------------------------------------------
# RECOMMENDATION NODE
# ---------------------------------------------------------

def recommendation_node(state):
    """
    Final Recommendation Agent.

    Combines specialist outputs and rule-based risk assessment.
    Only compact data is sent to the LLM. Complete specialist data remains
    available in the returned recommendation.
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
        # Debug information
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

        response = recommendation_llm.invoke(
            RECOMMENDATION_PROMPT.format(
                user_question=user_question,
                agent_data=agent_data_json,
                risk_signals=risk_json,
            )
        )

        content = _clean_model_content(response.content)

        print(
            "[recommendation_node] "
            f"raw_response_chars={len(content)}"
        )
        print(
            "[recommendation_node] "
            f"raw_response={repr(content[:4000])}"
        )

        # -------------------------------------------------
        # Parse JSON locally, safely
        # -------------------------------------------------

        if not content:
            raise ValueError(
                "Recommendation model returned an empty response. "
                "The GPT-OSS completion may have exhausted its budget "
                "during reasoning."
            )

        # First try the complete response.
        recommendation = None

        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                recommendation = parsed
        except json.JSONDecodeError:
            pass

        # If the model added text around the JSON, extract the first
        # balanced JSON object.
        if recommendation is None:
            start_positions = [
                match.start()
                for match in re.finditer(r"\\{", content)
            ]

            for start in start_positions:
                depth = 0
                in_string = False
                escaped = False

                for index in range(start, len(content)):
                    char = content[index]

                    if in_string:
                        if escaped:
                            escaped = False
                        elif char == "\\\\":
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
                            candidate = content[start:index + 1]

                            try:
                                parsed = json.loads(candidate)
                                if isinstance(parsed, dict):
                                    recommendation = parsed
                            except json.JSONDecodeError:
                                pass

                            break

                if recommendation is not None:
                    break

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

        # Keep the complete specialist outputs available to the caller/UI.
        recommendation["agent_findings"] = agent_data

        # -------------------------------------------------
        # Schema validation
        # -------------------------------------------------

        try:
            Recommendation.model_validate(recommendation)
        except ValidationError as ve:
            # Do not fail the whole graph just because the optional schema
            # has a mismatch. Preserve the generated recommendation and
            # expose the validation warning for debugging.
            recommendation["schema_warning"] = str(ve)

        return {
            "recommendation": recommendation,
            "status": "SUCCESS",
        }

    except json.JSONDecodeError as e:
        print(
            f"[recommendation_node] JSON parse error: {e}"
        )

        return {
            "recommendation": {
                "summary": "Unable to parse the final recommendation.",
                "risk_level": "MODERATE",
                "recommendation": "Please review the available marine data.",
                "key_findings": [],
                "safety_advice": [],
                "agent_findings": agent_data,
                "error": str(e),
            },
            "status": "FAILED",
        }

    except Exception as e:
        print(
            f"[recommendation_node] Recommendation generation failed: {e}"
        )

        return {
            "recommendation": {
                "summary": "Recommendation generation failed.",
                "risk_level": "MODERATE",
                "recommendation": "Please review the available marine data.",
                "key_findings": [],
                "safety_advice": [],
                "agent_findings": agent_data,
                "error": str(e),
            },
            "status": "FAILED",
        }
