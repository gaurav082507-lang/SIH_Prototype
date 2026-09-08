# server.py

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# IMPORT YOUR LANGGRAPH GRAPH
# ============================================================

from graph import marine_graph


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="ORCAWA Marine Conditions Intelligence API",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "index.html"


# ============================================================
# REQUEST MODEL
# ============================================================

class AskRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    question: str = Field(..., min_length=1)


# ============================================================
# ROOT ROUTE
# ============================================================

@app.get("/")
def home():
    """
    Serve the ORCAWA frontend.
    """

    if not INDEX_FILE.exists():
        return {
            "status": "error",
            "message": "index.html not found",
            "expected_path": str(INDEX_FILE),
        }

    return FileResponse(
        INDEX_FILE,
        media_type="text/html",
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "ORCAWA",
    }


# ============================================================
# HELPER: CONVERT OBJECTS TO JSON-SAFE VALUES
# ============================================================

def _json_safe(value: Any) -> Any:
    """
    Convert arbitrary Python objects into JSON-safe values.

    This prevents SSE / JSON serialization from breaking if
    the LangGraph state contains objects such as datetime,
    Pydantic models, etc.
    """

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {
            str(k): _json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            _json_safe(v)
            for v in value
        ]

    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump())
        except Exception:
            pass

    if hasattr(value, "dict"):
        try:
            return _json_safe(value.dict())
        except Exception:
            pass

    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass

    return str(value)


# ============================================================
# HELPER: SSE EVENT
# ============================================================

def _sse(event: dict[str, Any]) -> str:
    """
    Convert a dictionary into a Server-Sent Event message.
    """

    safe_event = _json_safe(event)

    return (
        f"data: {json.dumps(safe_event, ensure_ascii=False)}\n\n"
    )


# ============================================================
# HELPER: PARSE PLANNER OUTPUT
# ============================================================

def _parse_plan(state: dict[str, Any]) -> dict[str, Any]:
    """
    Extract the planner output from the LangGraph state.

    This function is deliberately tolerant because the planner
    output may be stored under different keys depending on the
    graph implementation.
    """

    # --------------------------------------------------------
    # Direct "plan" key
    # --------------------------------------------------------

    plan = state.get("plan")

    if isinstance(plan, dict):
        return _json_safe(plan)

    if hasattr(plan, "model_dump"):
        try:
            return _json_safe(plan.model_dump())
        except Exception:
            pass

    # --------------------------------------------------------
    # Planner output
    # --------------------------------------------------------

    planner_output = state.get("planner")

    if isinstance(planner_output, dict):
        return _json_safe(planner_output)

    if hasattr(planner_output, "model_dump"):
        try:
            return _json_safe(planner_output.model_dump())
        except Exception:
            pass

    # --------------------------------------------------------
    # Common individual planner fields
    # --------------------------------------------------------

    possible_keys = [
        "activity",
        "date",
        "planned_datetime",
        "required_agents",
        "required_tools",
        "agents",
    ]

    extracted: dict[str, Any] = {}

    for key in possible_keys:
        if key in state:
            extracted[key] = _json_safe(state[key])

    return extracted


# ============================================================
# HELPER: NORMALIZE RECOMMENDATION
# ============================================================

def _normalize_recommendation(
    state: dict[str, Any]
) -> dict[str, Any]:
    """
    Extract the final recommendation from the graph state.

    The frontend expects:

    recommendation: {
        risk_level,
        summary,
        recommendation,
        key_findings,
        safety_advice,
        agent_findings
    }
    """

    recommendation = state.get("recommendation")

    # --------------------------------------------------------
    # Recommendation is already a dictionary
    # --------------------------------------------------------

    if isinstance(recommendation, dict):

        result = dict(recommendation)

        # Normalize common alternate names
        if "risk_level" not in result:
            if "risk" in result:
                result["risk_level"] = result["risk"]

        if "summary" not in result:
            if "assessment" in result:
                result["summary"] = result["assessment"]

        if "recommendation" not in result:
            if "advice" in result:
                result["recommendation"] = result["advice"]

        if "key_findings" not in result:
            result["key_findings"] = []

        if "safety_advice" not in result:
            result["safety_advice"] = []

        if "agent_findings" not in result:
            result["agent_findings"] = {}

        return _json_safe(result)

    # --------------------------------------------------------
    # Recommendation is a Pydantic model
    # --------------------------------------------------------

    if hasattr(recommendation, "model_dump"):
        try:
            result = recommendation.model_dump()

            if "key_findings" not in result:
                result["key_findings"] = []

            if "safety_advice" not in result:
                result["safety_advice"] = []

            if "agent_findings" not in result:
                result["agent_findings"] = {}

            return _json_safe(result)

        except Exception:
            pass

    # --------------------------------------------------------
    # Search for individual recommendation fields
    # --------------------------------------------------------

    result = {
        "risk_level": state.get(
            "risk_level",
            state.get("risk", "UNKNOWN"),
        ),

        "summary": state.get(
            "summary",
            state.get("assessment", ""),
        ),

        "recommendation": state.get(
            "recommendation_text",
            state.get("advice", ""),
        ),

        "key_findings": state.get(
            "key_findings",
            [],
        ),

        "safety_advice": state.get(
            "safety_advice",
            [],
        ),

        "agent_findings": state.get(
            "agent_findings",
            {},
        ),
    }

    return _json_safe(result)


# ============================================================
# NORMALIZE AGENT NODE NAME
# ============================================================

def _normalize_node_name(node_name: str) -> str:
    """
    Convert internal LangGraph node names into the names
    expected by the ORCAWA frontend.
    """

    name = str(node_name).strip().lower()

    aliases = {
        "gis_agent": "gis",
        "coastal_agent": "gis",
        "coastal_check": "gis",

        "weather_agent": "weather",

        "ocean_agent": "ocean",
        "sea_state": "ocean",

        "tide_agent": "tide",

        "cyclone_agent": "cyclone",

        "ecosystem_agent": "ecosystem",

        "pfz_agent": "pfz",
        "fishing_zone": "pfz",

        "recommendation_agent": "recommendation",
        "final_assessment": "recommendation",
        "assessment": "recommendation",
    }

    return aliases.get(name, name)


# ============================================================
# STREAMING PIPELINE
# ============================================================

def _stream_ask_events(
    payload: AskRequest,
):

    initial_state: dict[str, Any] = {
        "latitude": payload.latitude,
        "longitude": payload.longitude,
        "user_question": payload.question.strip(),
        "status": "STARTED",
    }

    started = time.monotonic()

    final_state: dict[str, Any] = dict(initial_state)

    completed_nodes: list[str] = []

    try:

        # ====================================================
        # RUN LANGGRAPH
        # ====================================================

        for step_output in marine_graph.stream(
            initial_state,
            stream_mode="updates",
        ):

            if not isinstance(step_output, dict):
                continue

            # ------------------------------------------------
            # Each update normally looks like:
            #
            # {
            #     "planner": {...}
            # }
            #
            # or
            #
            # {
            #     "weather": {...}
            # }
            # ------------------------------------------------

            for raw_node_name, node_update in step_output.items():

                node_name = _normalize_node_name(
                    str(raw_node_name)
                )

                # --------------------------------------------
                # Save completed node
                # --------------------------------------------

                completed_nodes.append(node_name)

                # --------------------------------------------
                # Merge node state into final state
                # --------------------------------------------

                if isinstance(node_update, dict):

                    final_state.update(
                        node_update
                    )

                elif hasattr(node_update, "model_dump"):

                    try:

                        final_state.update(
                            node_update.model_dump()
                        )

                    except Exception:
                        pass

                # --------------------------------------------
                # Create frontend event
                # --------------------------------------------

                event: dict[str, Any] = {
                    "type": "node_done",
                    "node": node_name,
                }

                # --------------------------------------------
                # Planner event
                # --------------------------------------------

                if node_name == "planner":

                    event["plan"] = _parse_plan(
                        final_state
                    )

                # --------------------------------------------
                # Recommendation event
                # --------------------------------------------

                if node_name == "recommendation":

                    event["recommendation"] = (
                        _normalize_recommendation(
                            final_state
                        )
                    )

                # --------------------------------------------
                # Send event to frontend
                # --------------------------------------------

                yield _sse(event)

        # ====================================================
        # PIPELINE FINISHED
        # ====================================================

        duration = round(
            time.monotonic() - started,
            2,
        )

        # Remove duplicate node names while preserving order
        unique_nodes = list(
            dict.fromkeys(completed_nodes)
        )

        # ====================================================
        # FINAL EVENT
        # ====================================================

        yield _sse(
            {
                "type": "final",
                "status": "SUCCESS",

                "latitude": payload.latitude,
                "longitude": payload.longitude,

                "question": payload.question,

                "plan": _parse_plan(
                    final_state
                ),

                "completed_nodes": unique_nodes,

                "recommendation": (
                    _normalize_recommendation(
                        final_state
                    )
                ),

                "duration_seconds": duration,
            }
        )

    except Exception as exc:

        # ----------------------------------------------------
        # Print complete traceback to Render logs
        # ----------------------------------------------------

        traceback.print_exc()

        # ----------------------------------------------------
        # Send error to frontend
        # ----------------------------------------------------

        yield _sse(
            {
                "type": "error",
                "detail": str(exc),
            }
        )


# ============================================================
# STREAMING API ENDPOINT
# ============================================================

@app.post("/api/ask/stream")
def ask_stream(
    payload: AskRequest,
):

    return StreamingResponse(
        _stream_ask_events(payload),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# NORMAL NON-STREAMING API
# ============================================================

@app.post("/api/ask")
def ask(
    payload: AskRequest,
):

    initial_state: dict[str, Any] = {
        "latitude": payload.latitude,
        "longitude": payload.longitude,
        "user_question": payload.question.strip(),
        "status": "STARTED",
    }

    started = time.monotonic()

    final_state: dict[str, Any] = dict(
        initial_state
    )

    completed_nodes: list[str] = []

    try:

        # Run the graph normally
        result = marine_graph.invoke(
            initial_state
        )

        if isinstance(result, dict):

            final_state.update(
                result
            )

        # Get node information if available
        if isinstance(result, dict):

            completed_nodes = [
                str(key)
                for key in result.keys()
            ]

        duration = round(
            time.monotonic() - started,
            2,
        )

        return {
            "status": "SUCCESS",

            "latitude": payload.latitude,
            "longitude": payload.longitude,

            "question": payload.question,

            "plan": _parse_plan(
                final_state
            ),

            "completed_nodes": completed_nodes,

            "recommendation": (
                _normalize_recommendation(
                    final_state
                )
            ),

            "duration_seconds": duration,
        }

    except Exception as exc:

        traceback.print_exc()

        return {
            "status": "ERROR",
            "detail": str(exc),
        }


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    import os
    import uvicorn

    port = int(
        os.environ.get(
            "PORT",
            "8000",
        )
    )

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )
