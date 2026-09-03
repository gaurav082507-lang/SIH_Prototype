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
    api_key=os.getenv("MISTRAL_API_KEY")
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
        "pfz": null
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
    }

    try:

        # -------------------------------------------------
        # Convert to JSON for the LLM
        # -------------------------------------------------

        agent_data_json = json.dumps(
            agent_data,
            ensure_ascii=False,
            indent=2,
            default=str
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