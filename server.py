from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

from schemas import AskRequest
from graph import marine_graph


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="ORCAWA Marine Intelligence API",
    version="1.0.0",
)


# ============================================================
# PATHS
# ============================================================

# server.py is in the project root.
#
# Project structure:
#
# project/
# ├── server.py
# ├── graph.py
# ├── schemas.py
# ├── ...
# └── static/
#     └── index.html
#
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"


# ============================================================
# FRONTEND
# ============================================================

@app.get("/")
def serve_frontend():
    """
    Serve the ORCAWA frontend.

    index.html is located at:
        static/index.html
    """

    if not INDEX_FILE.exists():
        return {
            "status": "error",
            "message": "index.html not found",
            "expected_path": str(INDEX_FILE),
        }

    return FileResponse(
        path=str(INDEX_FILE),
        media_type="text/html",
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():
    """
    Simple endpoint for Render/service health checks.
    """

    return {
        "status": "ok",
        "service": "ORCAWA Marine Intelligence API",
        "frontend": INDEX_FILE.exists(),
    }


# ============================================================
# HELPER: PARSE PLANNER OUTPUT
# ============================================================

def _parse_plan(state: dict[str, Any]) -> dict[str, Any]:
    """
    Extract the planner result from the LangGraph state.

    The planner may store its output under different keys depending
    on the current graph implementation. This helper keeps the
    server tolerant of those variations.
    """

    # --------------------------------------------------------
    # Direct "plan" key
    # --------------------------------------------------------

    plan = state.get("plan")

    if isinstance(plan, dict):
        return plan

    # --------------------------------------------------------
    # Planner output may itself be stored as planner
    # --------------------------------------------------------

    planner = state.get("planner")

    if isinstance(planner, dict):
        if isinstance(planner.get("plan"), dict):
            return planner["plan"]

        return planner

    # --------------------------------------------------------
    # Some implementations may use planner_result
    # --------------------------------------------------------

    planner_result = state.get("planner_result")

    if isinstance(planner_result, dict):
        if isinstance(planner_result.get("plan"), dict):
            return planner_result["plan"]

        return planner_result

    # --------------------------------------------------------
    # Nothing available
    # --------------------------------------------------------

    return {}


# ============================================================
# HELPER: NORMALIZE RECOMMENDATION
# ============================================================

def _normalize_recommendation(
    state: dict[str, Any],
) -> dict[str, Any]:
    """
    Normalize the recommendation node output so that the frontend
    always receives a predictable object.

    Expected frontend structure:

    recommendation:
    {
        "risk_level": "...",
        "summary": "...",
        "recommendation": "...",
        "key_findings": [...],
        "safety_advice": [...],
        "agent_findings": {...}
    }
    """

    recommendation = state.get("recommendation")

    # --------------------------------------------------------
    # If recommendation is already a dictionary
    # --------------------------------------------------------

    if isinstance(recommendation, dict):
        result = dict(recommendation)

    else:
        result = {}

        # ----------------------------------------------------
        # Try alternate recommendation keys
        # ----------------------------------------------------

        for key in (
            "recommendation_result",
            "recommendation_output",
            "final_recommendation",
        ):
            candidate = state.get(key)

            if isinstance(candidate, dict):
                result = dict(candidate)
                break

    # --------------------------------------------------------
    # Normalize risk level
    # --------------------------------------------------------

    if "risk_level" not in result:
        for key in ("risk", "overall_risk", "riskLevel"):
            if key in result:
                result["risk_level"] = result[key]
                break

    if "risk_level" not in result:
        result["risk_level"] = "UNKNOWN"

    # --------------------------------------------------------
    # Normalize summary
    # --------------------------------------------------------

    if "summary" not in result:
        for key in (
            "assessment",
            "overall_assessment",
            "message",
        ):
            if key in result:
                result["summary"] = result[key]
                break

    if "summary" not in result:
        result["summary"] = ""

    # --------------------------------------------------------
    # Normalize recommendation text
    # --------------------------------------------------------

    if "recommendation" not in result:
        for key in (
            "recommendation_text",
            "advice",
            "decision",
        ):
            if key in result:
                result["recommendation"] = result[key]
                break

    if "recommendation" not in result:
        result["recommendation"] = ""

    # --------------------------------------------------------
    # Normalize key findings
    # --------------------------------------------------------

    if not isinstance(result.get("key_findings"), list):
        result["key_findings"] = []

    # --------------------------------------------------------
    # Normalize safety advice
    # --------------------------------------------------------

    if not isinstance(result.get("safety_advice"), list):
        result["safety_advice"] = []

    # --------------------------------------------------------
    # Agent findings
    #
    # Prefer an explicitly generated agent_findings object.
    # Otherwise expose useful state data without breaking the UI.
    # --------------------------------------------------------

    if not isinstance(result.get("agent_findings"), dict):
        agent_findings = {}

        for key, value in state.items():
            if key in {
                "latitude",
                "longitude",
                "user_question",
                "status",
                "plan",
                "planner",
                "planner_result",
                "recommendation",
                "recommendation_result",
                "recommendation_output",
                "final_recommendation",
            }:
                continue

            try:
                json.dumps(value)
                agent_findings[key] = value
            except (TypeError, ValueError):
                agent_findings[key] = str(value)

        result["agent_findings"] = agent_findings

    return result


# ============================================================
# HELPER: SSE FORMATTER
# ============================================================

def _sse(event: dict[str, Any]) -> str:
    """
    Convert a Python dictionary into a Server-Sent Event.

    Example:

        data: {"type":"node_done","node":"weather"}

    followed by two newlines.
    """

    return f"data: {json.dumps(event, default=str)}\n\n"


# ============================================================
# CREATE INITIAL GRAPH STATE
# ============================================================

def _create_initial_state(payload: AskRequest) -> dict[str, Any]:
    """
    Build the state passed into the LangGraph.
    """

    return {
        "latitude": payload.latitude,
        "longitude": payload.longitude,
        "user_question": payload.question.strip(),
        "status": "STARTED",
    }


# ============================================================
# NORMAL EXECUTION
# ============================================================

@app.post("/api/ask")
def ask(payload: AskRequest):
    """
    Normal non-streaming ORCAWA assessment endpoint.

    Used when the caller wants the final result directly.
    """

    initial_state = _create_initial_state(payload)

    started = time.monotonic()

    try:
        # ----------------------------------------------------
        # Execute LangGraph
        # ----------------------------------------------------

        final_state = marine_graph.invoke(initial_state)

        if not isinstance(final_state, dict):
            final_state = dict(initial_state)

        duration = round(time.monotonic() - started, 2)

        # ----------------------------------------------------
        # Extract outputs
        # ----------------------------------------------------

        plan = _parse_plan(final_state)

        recommendation = _normalize_recommendation(
            final_state
        )

        # ----------------------------------------------------
        # Return API response
        # ----------------------------------------------------

        return {
            "status": "SUCCESS",
            "latitude": payload.latitude,
            "longitude": payload.longitude,
            "question": payload.question,
            "plan": plan,
            "recommendation": recommendation,
            "duration_seconds": duration,
        }

    except Exception as exc:
        traceback.print_exc()

        return {
            "status": "ERROR",
            "message": str(exc),
        }


# ============================================================
# STREAMING EXECUTION
# ============================================================

def _stream_ask_events(
    payload: AskRequest,
) -> Iterator[str]:
    """
    Execute the LangGraph while sending SSE events to the frontend.

    Frontend endpoint:

        POST /api/ask/stream

    Events sent:

        node_done
        final
        error
    """

    initial_state = _create_initial_state(payload)

    started = time.monotonic()

    final_state: dict[str, Any] = dict(initial_state)

    completed_nodes: list[str] = []

    try:

        # ====================================================
        # STREAM LANGGRAPH
        # ====================================================

        for step_output in marine_graph.stream(
            initial_state,
            stream_mode="updates",
        ):

            if not isinstance(step_output, dict):
                continue

            # ------------------------------------------------
            # Each graph update normally looks like:
            #
            # {
            #     "planner": {...}
            # }
            #
            # or:
            #
            # {
            #     "weather": {...}
            # }
            # ------------------------------------------------

            for node_name, node_update in step_output.items():

                node_name = str(node_name)

                # --------------------------------------------
                # Track completed node
                # --------------------------------------------

                completed_nodes.append(node_name)

                # --------------------------------------------
                # Merge node output into final state
                # --------------------------------------------

                if isinstance(node_update, dict):
                    final_state.update(node_update)

                # --------------------------------------------
                # Create SSE event
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

                elif node_name in {
                    "recommendation",
                    "recommendation_node",
                }:
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
        # GRAPH FINISHED
        # ====================================================

        duration = round(
            time.monotonic() - started,
            2,
        )

        # ----------------------------------------------------
        # Prepare final result
        # ----------------------------------------------------

        plan = _parse_plan(final_state)

        recommendation = _normalize_recommendation(
            final_state
        )

        # ----------------------------------------------------
        # Send final SSE event
        # ----------------------------------------------------

        yield _sse(
            {
                "type": "final",
                "status": "SUCCESS",
                "latitude": payload.latitude,
                "longitude": payload.longitude,
                "question": payload.question,
                "plan": plan,
                "completed_nodes": completed_nodes,
                "recommendation": recommendation,
                "duration_seconds": duration,
            }
        )

    except Exception as exc:

        # ====================================================
        # GRAPH ERROR
        # ====================================================

        traceback.print_exc()

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
def ask_stream(payload: AskRequest):
    """
    Streaming ORCAWA assessment endpoint.

    The index.html frontend calls:

        /api/ask/stream
    """

    return StreamingResponse(
        _stream_ask_events(payload),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )
