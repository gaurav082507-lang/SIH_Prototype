# recommendation_node.py
#
# UPDATED: swapped ChatMistralAI -> ChatGroq (JSON mode forced via
# response_format to reduce parse failures, since Llama models are
# slightly less reliable at "return only JSON" than Mistral was).
#
# Retained from before:
#   1. Chat memory — includes conversation_history (compacted) in the
#      prompt, so the model can reference prior turns ("as found
#      earlier...") instead of treating every message as a cold start.
#   2. Risk/decision separation — receives risk_signals (computed by
#      risk_rules.risk_node, which now runs before this node in
#      graph.py) and is told to treat it as ground truth for
#      risk_level rather than re-deriving severity purely from vibes.
#   3. Schema validation — the parsed JSON is checked against
#      schemas.Recommendation before being returned.

import os
import json

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from schemas import Recommendation


# ---------------------------------------------------------
# LLM
# ---------------------------------------------------------

recommendation_llm = ChatGroq(
    model=os.getenv(
        "GROQ_MODEL_RECOMMENDATION",
        "openai/gpt-oss-120b"
    ),
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
    max_retries=6,
    model_kwargs={"response_format": {"type": "json_object"}},
)


# ---------------------------------------------------------
# PROMPT
# ---------------------------------------------------------

RECOMMENDATION_PROMPT = ChatPromptTemplate.from_template(
"""
You are the Final Marine Intelligence and Recommendation Agent.

You receive structured data collected by specialist marine
agents, a rule-based risk pre-assessment, and (if this is not the
first message) a short history of the conversation so far.

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
- RISK PRE-ASSESSMENT below is computed by fixed, documented
  thresholds (wind, wave height, cyclone distance), not by you. Set
  "risk_level" to match its "overall_rule_based_severity" unless it is
  "insufficient_data", in which case use your own judgment from
  whatever agent data is available and say explicitly that the
  automated risk check had insufficient data.
- If CONVERSATION HISTORY is non-empty, treat this as a continuing
  conversation: reference earlier findings where relevant instead of
  repeating a full fresh analysis, and answer the user's actual new
  question directly.
- Some agent data below may be abbreviated ("...N more entries
  omitted for brevity...") to keep this prompt a reasonable size —
  treat that as "additional similar data points were collected but
  not shown", not as missing data.

Conversation History:

{conversation_history}

Risk Pre-Assessment (rule-based, authoritative for severity):

{risk_signals}

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
# PROMPT-SIZE COMPACTION (unchanged from original)
# ---------------------------------------------------------

MAX_LIST_LEN = 6
MAX_COMPACT_DEPTH = 6
MAX_PROMPT_CHARS = 24000
MAX_HISTORY_TURNS_IN_PROMPT = 5


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


def _format_history_for_prompt(conversation_history):
    """Same idea as planner_node._format_history: only question +
    prior recommendation summary/risk_level, never raw agent_data, so
    history never dominates the prompt-size budget."""
    if not conversation_history:
        return "(no prior turns — this is the first message)"

    recent = conversation_history[-MAX_HISTORY_TURNS_IN_PROMPT:]
    lines = []
    for turn in recent:
        question = turn.get("question", "")
        rec = turn.get("recommendation") or {}
        summary = rec.get("summary", "") if isinstance(rec, dict) else ""
        risk_level = rec.get("risk_level", "") if isinstance(rec, dict) else ""
        lines.append(f"- User asked: {question!r} -> risk_level={risk_level!r}, summary={summary!r}")
    return "\n".join(lines)


# ---------------------------------------------------------
# RECOMMENDATION NODE
# ---------------------------------------------------------

def recommendation_node(state):
    """
    Final Recommendation Agent.

    Combines outputs from all executed specialist agents, the
    rule-based risk pre-assessment, and any conversation history, and
    uses the LLM to generate the final assessment.
    """

    user_question = state.get(
        "user_question",
        ""
    )

    conversation_history = state.get("conversation_history", [])
    risk_signals = state.get("risk_signals", {})

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
        # Call the LLM
        # -------------------------------------------------

        response = recommendation_llm.invoke(
            RECOMMENDATION_PROMPT.format(
                user_question=user_question,
                agent_data=agent_data_json,
                conversation_history=_format_history_for_prompt(conversation_history),
                risk_signals=json.dumps(risk_signals, ensure_ascii=False, indent=2, default=str),
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

        # -------------------------------------------------
        # Schema validation
        # -------------------------------------------------

        try:
            Recommendation.model_validate(recommendation)
        except ValidationError as ve:
            recommendation["schema_warning"] = str(ve)

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
