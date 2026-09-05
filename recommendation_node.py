import json
import os
import re

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from schemas import Recommendation


print("🔥 RECOMMENDATION NODE LOADED")
print("FILE:", __file__)


# =========================================================
# GEMINI
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
# PROMPT
# =========================================================

RECOMMENDATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the Final Marine Intelligence and Recommendation Agent.

Answer the user's current question using ONLY the supplied marine agent data
and rule-based risk assessment.

DATA INTERPRETATION:
- NULL values are VALID.
- A null field does NOT automatically mean insufficient data.
- Ignore individual null fields and continue using all other available evidence.
- Never invent a value for a null field.
- Partial agent data is acceptable.
- Do not increase the risk level merely because some fields are null.

NO-DATA RULE:
- Only say that marine data could not be retrieved when ALL seven agent
  outputs are null, missing, or empty:
  weather, ocean, tide, cyclone, ecosystem, pfz, gis.
- If even ONE agent contains meaningful data, perform a normal assessment.
- Never describe partial data as insufficient data.
- Never assume that null means the location is invalid.

CYCLONE RULE:
- If cyclone is null, interpret it as:
  "No cyclone threat is indicated by the available cyclone assessment."
- Do NOT say:
  "insufficient cyclone data"
  "cyclone data unavailable"
  "unable to assess cyclone conditions"
  simply because cyclone is null.
- Do not invent cyclone information.

OTHER NULL VALUES:
- weather = null → ignore that weather finding.
- ocean = null → ignore that ocean finding.
- tide = null → ignore that tide finding.
- ecosystem = null → ignore that ecosystem finding.
- pfz = null → ignore that PFZ finding.
- gis = null → ignore that GIS finding.

VALUE-DISPLAY RULE:
If the user explicitly asks for values, measurements, coordinates, times,
conditions, or factual data, display the actual requested values from the
supplied marine agent data whenever those values exist.

Examples:
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
- PFZ latitude
- PFZ longitude
- distance from shore
- port distance
- GIS restricted-zone information
- any other numerical or factual value supplied by an agent

Rules for requested values:
- If the requested value exists, display the ACTUAL value.
- Do not replace an available value with a vague statement.
- Never invent missing values.
- If one requested value is null but others are available, display the
  available values and continue the assessment normally.
- If a requested value is null, omit that value.
- Preserve units supplied by the agent.
- Do not invent units.
- Keep the answer concise.

SAFETY:
- Prioritize safety.
- Explain the main reasons for the recommendation.
- Clearly warn the user when dangerous conditions are present.
- Use the rule-based severity whenever it provides a usable risk level.
- If the rule-based severity says insufficient_data but meaningful agent data
  exists, determine the best supported risk level from the available data.

OUTPUT:
Return ONLY ONE NORMAL JSON OBJECT.

Do not use markdown.
Do not use code fences.
Do not return a JSON-encoded string.
Do not put the JSON inside quotes.
Do not add explanations before or after the JSON.

The JSON must contain these fields:
summary
risk_level
recommendation
key_findings
safety_advice

summary:
- One sentence only.

risk_level:
- Must be exactly one of:
  LOW
  MODERATE
  HIGH
  SEVERE

recommendation:
- One sentence only.
- It must be actionable.

key_findings:
- Maximum 2 items.
- Use this field to display explicitly requested factual values when appropriate.

safety_advice:
- Maximum 2 items.

Do NOT include agent_findings in the JSON returned by the model.

NO-DATA RESPONSE:
If and ONLY if every agent has no meaningful data, return:

summary: Marine data could not be retrieved for this location right now.
risk_level: LOW
recommendation: Try again shortly, or verify the coordinates and retry.
key_findings: []
safety_advice: []

Remember:
- One null field is NOT insufficient data.
- cyclone null means no cyclone threat is indicated by the available
  cyclone assessment.
- Partial agent data is valid.
- Use actual requested values whenever they are present.
""",
        ),
        (
            "human",
            """
USER QUESTION:
{user_question}

RULE-BASED RISK:
{risk_signals}

MARINE AGENT DATA:
{agent_data}

Return the final JSON object now.
""",
        ),
    ]
)


# =========================================================
# LIMITS
# =========================================================

MAX_LIST_LEN = 5
MAX_COMPACT_DEPTH = 5

MAX_AGENT_DATA_CHARS = 30000
MAX_RISK_CHARS = 5000
MAX_QUESTION_CHARS = 3000


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
    Reduce very large nested agent responses while preserving
    useful marine information.
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

            kept = obj[:max_list_len]

            compacted = [
                _compact_for_llm(
                    item,
                    max_list_len,
                    max_depth,
                    _depth + 1,
                )
                for item in kept
            ]

            compacted.append(
                f"...{len(obj) - len(kept)} more entries omitted..."
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
    Always returns valid JSON.

    Never cuts JSON in the middle.
    """

    text = json.dumps(
        obj,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )

    if len(text) <= max_chars:
        return text

    if isinstance(obj, dict):

        reduced = {}

        for key, value in obj.items():

            candidate = dict(reduced)
            candidate[key] = value

            candidate_text = json.dumps(
                candidate,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            )

            if len(candidate_text) <= max_chars:
                reduced[key] = value

        reduced["_truncated"] = True

        return json.dumps(
            reduced,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )

    return json.dumps(
        {
            "_truncated": True,
            "message": "Data shortened for recommendation processing.",
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


# =========================================================
# MODEL RESPONSE CLEANING
# =========================================================

def _clean_model_content(content):
    """
    Convert Gemini content into plain text.
    """

    if content is None:
        return ""

    # Gemini/LangChain can sometimes return a list of content blocks.
    if isinstance(content, list):

        parts = []

        for item in content:

            if isinstance(item, dict):

                text = item.get("text")

                if text is not None:
                    parts.append(str(text))

            else:
                parts.append(str(item))

        content = "".join(parts)

    # Handle dictionary content.
    if isinstance(content, dict):

        if "text" in content:
            content = content["text"]

        else:
            content = json.dumps(
                content,
                ensure_ascii=False,
            )

    return str(content).strip()


def _strip_code_fences(text):
    """
    Remove accidental markdown fences.
    """

    text = text.strip()

    text = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```\s*$",
        "",
        text,
    )

    return text.strip()


# =========================================================
# ROBUST JSON PARSER
# =========================================================

def _try_parse_json(value):
    """
    Handles:

    1. Normal JSON
    2. JSON wrapped in quotes
    3. JSON containing literal \\n
    4. Markdown code fences
    5. JSON surrounded by other text
    """

    if isinstance(value, dict):
        return value

    if not isinstance(value, str):
        return None

    candidate = _strip_code_fences(value)

    # -----------------------------------------------------
    # 1. Normal JSON
    # -----------------------------------------------------

    try:

        parsed = json.loads(candidate)

        if isinstance(parsed, dict):
            return parsed

        # Gemini sometimes returns:
        #
        # "{\n \"summary\": \"...\"}"
        #
        # In this case the first json.loads() produces a string.

        if isinstance(parsed, str):

            inner = parsed.strip()

            try:

                inner_parsed = json.loads(inner)

                if isinstance(inner_parsed, dict):
                    return inner_parsed

            except (
                json.JSONDecodeError,
                TypeError,
            ):
                pass

    except (
        json.JSONDecodeError,
        TypeError,
    ):
        pass

    # -----------------------------------------------------
    # 2. Literal escaped newlines
    # -----------------------------------------------------

    normalized = candidate

    normalized = normalized.replace(
        "\\r\\n",
        "\n",
    )

    normalized = normalized.replace(
        "\\n",
        "\n",
    )

    normalized = normalized.replace(
        "\\r",
        "\r",
    )

    normalized = normalized.replace(
        "\\t",
        "\t",
    )

    if normalized != candidate:

        try:

            parsed = json.loads(normalized)

            if isinstance(parsed, dict):
                return parsed

            if isinstance(parsed, str):

                try:

                    inner = json.loads(parsed)

                    if isinstance(inner, dict):
                        return inner

                except (
                    json.JSONDecodeError,
                    TypeError,
                ):
                    pass

        except (
            json.JSONDecodeError,
            TypeError,
        ):
            pass

    # -----------------------------------------------------
    # 3. Extract balanced JSON object
    # -----------------------------------------------------

    for source in (
        candidate,
        normalized,
    ):

        for match in re.finditer(
            r"\{",
            source,
        ):

            start = match.start()

            depth = 0
            in_string = False
            escaped = False

            for index in range(
                start,
                len(source),
            ):

                char = source[index]

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

                        json_text = source[
                            start:index + 1
                        ]

                        try:

                            parsed = json.loads(
                                json_text
                            )

                            if isinstance(
                                parsed,
                                dict,
                            ):
                                return parsed

                        except (
                            json.JSONDecodeError,
                            TypeError,
                        ):
                            pass

                        break

    return None


# =========================================================
# MEANINGFUL DATA CHECK
# =========================================================

def _has_meaningful_value(value):

    if value is None:
        return False

    if isinstance(value, str):
        return bool(value.strip())

    if isinstance(value, dict):
        return bool(value)

    if isinstance(value, list):
        return bool(value)

    return True


def _has_meaningful_data(agent_data):

    return any(
        _has_meaningful_value(value)
        for value in agent_data.values()
    )


# =========================================================
# NORMALIZE RECOMMENDATION
# =========================================================

def _normalize_recommendation(
    recommendation,
    agent_data,
):

    if not isinstance(
        recommendation,
        dict,
    ):
        recommendation = {}

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
        "Review the available marine conditions before proceeding.",
    )

    recommendation.setdefault(
        "key_findings",
        [],
    )

    recommendation.setdefault(
        "safety_advice",
        [],
    )

    # -----------------------------------------------------
    # Risk
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Findings
    # -----------------------------------------------------

    if not isinstance(
        recommendation["key_findings"],
        list,
    ):

        recommendation["key_findings"] = [
            str(
                recommendation["key_findings"]
            )
        ]

    if not isinstance(
        recommendation["safety_advice"],
        list,
    ):

        recommendation["safety_advice"] = [
            str(
                recommendation["safety_advice"]
            )
        ]

    recommendation["key_findings"] = [
        str(item)
        for item in recommendation[
            "key_findings"
        ][:2]
    ]

    recommendation["safety_advice"] = [
        str(item)
        for item in recommendation[
            "safety_advice"
        ][:2]
    ]

    # -----------------------------------------------------
    # Preserve complete agent data for UI
    # -----------------------------------------------------

    recommendation["agent_findings"] = agent_data

    return recommendation


# =========================================================
# RECOMMENDATION NODE
# =========================================================

def recommendation_node(state):

    print("\n" + "=" * 70)
    print("🚢 RECOMMENDATION AGENT STARTED")
    print("=" * 70)

    user_question = str(
        state.get(
            "user_question",
            "",
        )
    ).strip()

    user_question = user_question[
        :MAX_QUESTION_CHARS
    ]

    risk_signals = (
        state.get("risk_signals")
        or {}
    )

    # -----------------------------------------------------
    # Collect all specialist outputs
    # -----------------------------------------------------

    agent_data = {

        "weather": state.get(
            "weather_data"
        ),

        "ocean": state.get(
            "ocean_data"
        ),

        "tide": state.get(
            "tide_data"
        ),

        "cyclone": state.get(
            "cyclone_data"
        ),

        "ecosystem": state.get(
            "ecosystem_data"
        ),

        "pfz": state.get(
            "pfz_data"
        ),

        "gis": state.get(
            "gis_data"
        ),
    }

    # -----------------------------------------------------
    # Debug
    # -----------------------------------------------------

    print(
        "[recommendation_node] agent_data_presence="
        + json.dumps(
            {
                key: _has_meaningful_value(
                    value
                )
                for key, value
                in agent_data.items()
            }
        )
    )

    print(
        "[recommendation_node] meaningful_agent_data="
        + str(
            _has_meaningful_data(
                agent_data
            )
        )
    )

    # -----------------------------------------------------
    # Prepare prompt
    # -----------------------------------------------------

    try:

        compact_agent_data = (
            _compact_for_llm(
                agent_data
            )
        )

        agent_data_json = (
            _safe_json_text(
                compact_agent_data,
                MAX_AGENT_DATA_CHARS,
            )
        )

        risk_json = _safe_json_text(
            risk_signals,
            MAX_RISK_CHARS,
        )

        # IMPORTANT:
        # Use format_messages() here.
        # The JSON example is NOT embedded inside the template,
        # so the old "\n summary" KeyError problem cannot occur.

        messages = (
            RECOMMENDATION_PROMPT.format_messages(
                user_question=user_question,
                risk_signals=risk_json,
                agent_data=agent_data_json,
            )
        )

        prompt_length = sum(
            len(str(message.content))
            for message in messages
        )

        print(
            "[recommendation_node] prompt_chars="
            + str(prompt_length)
        )

        # -------------------------------------------------
        # Gemini call
        # -------------------------------------------------

        response = (
            recommendation_llm.invoke(
                messages
            )
        )

        content = _clean_model_content(
            getattr(
                response,
                "content",
                "",
            )
        )

        print(
            "[recommendation_node] "
            "raw_response_chars="
            + str(len(content))
        )

        print(
            "[recommendation_node] "
            "raw_response="
            + repr(content[:5000])
        )

        # -------------------------------------------------
        # Empty response
        # -------------------------------------------------

        if not content:

            raise ValueError(
                "Recommendation model returned an empty response."
            )

        # -------------------------------------------------
        # Parse Gemini JSON
        # -------------------------------------------------

        recommendation = (
            _try_parse_json(
                content
            )
        )

        if recommendation is None:

            raise ValueError(
                "Recommendation model returned "
                "non-JSON content: "
                + repr(
                    content[:1500]
                )
            )

        # -------------------------------------------------
        # Normalize
        # -------------------------------------------------

        recommendation = (
            _normalize_recommendation(
                recommendation,
                agent_data,
            )
        )

        # -------------------------------------------------
        # Pydantic validation
        # -------------------------------------------------

        try:

            Recommendation.model_validate(
                recommendation
            )

        except ValidationError as validation_error:

            print(
                "[recommendation_node] "
                "schema_warning="
                + str(validation_error)
            )

        # -------------------------------------------------
        # SUCCESS
        # -------------------------------------------------

        print(
            "✅ RECOMMENDATION AGENT SUCCESS"
        )

        print("=" * 70)

        return {
            "recommendation": recommendation,
            "status": "SUCCESS",
        }

    # =====================================================
    # ERROR HANDLING
    # =====================================================

    except Exception as error:

        print(
            "❌ RECOMMENDATION AGENT FAILED"
        )

        print(
            "[recommendation_node] "
            + repr(error)
        )

        print("=" * 70)

        meaningful_data = (
            _has_meaningful_data(
                agent_data
            )
        )

        # -------------------------------------------------
        # Deterministic fallback
        # -------------------------------------------------

        if meaningful_data:

            fallback_summary = (
                "Marine assessment completed "
                "from the available specialist data."
            )

            fallback_recommendation = (
                "Review the available marine "
                "conditions and follow the "
                "stated safety advice."
            )

            fallback_risk = "MODERATE"

        else:

            fallback_summary = (
                "Marine data could not be "
                "retrieved for this location "
                "right now."
            )

            fallback_recommendation = (
                "Try again shortly, or verify "
                "the coordinates and retry."
            )

            fallback_risk = "LOW"

        fallback = {

            "summary": fallback_summary,

            "risk_level": fallback_risk,

            "recommendation":
                fallback_recommendation,

            "key_findings": [],

            "safety_advice": [],

            # Keep all specialist data
            # available to the UI.

            "agent_findings":
                agent_data,

            "error": str(error),
        }

        return {

            "recommendation":
                fallback,

            "status":
                "FAILED",
        }
