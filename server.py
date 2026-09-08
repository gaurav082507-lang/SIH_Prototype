import json
import time
import traceback
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from graph import marine_graph


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="ORCA Marine Intelligence API",
    description="Agentic AI platform for ocean, weather, tide, cyclone and ecosystem analysis.",
    version="1.0.0",
)


# ============================================================
# STATIC FILES
# ============================================================

if STATIC_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(STATIC_DIR)),
        name="static",
    )


# ============================================================
# REQUEST SCHEMA
# ============================================================

class AskRequest(BaseModel):
    latitude: float = Field(..., description="Latitude of the requested location")
    longitude: float = Field(..., description="Longitude of the requested location")
    question: str = Field(..., min_length=1, description="User's marine-related question")


# ============================================================
# BASIC ROUTES
# ============================================================

@app.get("/")
def home():
    """
    Serve the frontend index.html.

    Project structure expected:

    src/
    ├── server.py
    ├── static/
    │   └── index.html
    └── ...
    """

    if not INDEX_FILE.exists():
        return {
            "status": "error",
            "message": "index.html not found",
            "expected_path": str(INDEX_FILE),
        }

    return FileResponse(
        str(INDEX_FILE),
        media_type="text/html",
    )


@app.get("/health")
def health():
    """
    Health check endpoint for Render.
    """
    return {
        "status": "ok",
        "service": "ORCA Marine Intelligence API",
    }


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _sse(event: dict[str, Any]) -> str:
    """
    Convert a Python dictionary into a Server-Sent Event.
    """

    return f"data: {json.dumps(event, default=str)}\n\n"


def _parse_plan(state: dict[str, Any]) -> Any:
    """
    Safely extract the planner output from the LangGraph state.

    Different versions of the planner may store the plan under
    different keys, so this function checks several possibilities.
    """

    possible_keys = [
        "plan",
        "planner_output",
        "planner_result",
        "planning",
        "steps",
    ]

    for key in possible_keys:
        value = state.get(key)

        if value is not None:
            # If already a dictionary/list, return it directly.
            if isinstance(value, (dict, list)):
                return value

            # If the planner returned JSON as a string, parse it.
            if isinstance(value, str):
                cleaned = value.strip()

                if not cleaned:
                    continue

                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    return cleaned

    return None


def _normalize_recommendation(state: dict[str, Any]) -> Any:
    """
    Safely extract the final recommendation from the graph state.
    """

    possible_keys = [
        "recommendation",
        "recommendations",
        "final_recommendation",
        "recommendation_output",
        "recommendation_result",
    ]

    for key in possible_keys:
        value = state.get(key)

        if value is not None:
            return value

    return None


def _clean_for_json(value: Any) -> Any:
    """
    Convert potentially complex objects into JSON-safe values.

    This prevents serialization errors if a graph node returns
    objects such as Pydantic models or other custom classes.
    """

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {
            str(k): _clean_for_json(v)
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [
            _clean_for_json(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            _clean_for_json(item)
            for item in value
        ]

    # Pydantic model
    if hasattr(value, "model_dump"):
        try:
            return _clean_for_json(value.model_dump())
        except Exception:
            pass

    # Older Pydantic versions
    if hasattr(value, "dict"):
        try:
            return _clean_for_json(value.dict())
        except Exception:
            pass

    return str(value)


# ============================================================
# NORMAL / NON-STREAMING ASK ENDPOINT
# ============================================================

@app.post("/api/ask")
def ask(payload: AskRequest):
    """
    Execute the complete marine LangGraph and return the final result.
    """

    initial_state: dict[str, Any] = {
        "latitude": payload.latitude,
        "longitude": payload.longitude,
        "user_question": payload.question.strip(),
        "status": "STARTED",
    }

    started = time.monotonic()

    try:

        final_state = marine_graph.invoke(initial_state)

        if not isinstance(final_state, dict):
            final_state = dict(initial_state)

        duration = round(
            time.monotonic() - started,
            2,
        )

        return {
            "status": "SUCCESS",
            "latitude": payload.latitude,
            "longitude": payload.longitude,
            "question": payload.question,
            "plan": _clean_for_json(
                _parse_plan(final_state)
            ),
            "recommendation": _clean_for_json(
                _normalize_recommendation(final_state)
            ),
            "state": _clean_for_json(final_state),
            "duration_seconds": duration,
        }

    except Exception as exc:

        traceback.print_exc()

        return {
            "status": "ERROR",
            "message": str(exc),
            "latitude": payload.latitude,
            "longitude": payload.longitude,
            "question": payload.question,
        }


# ============================================================
# STREAMING GRAPH EXECUTION
# ============================================================

def _stream_ask_events(payload: AskRequest):
    """
    Stream LangGraph node execution using Server-Sent Events.
    """

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

        # ----------------------------------------------------
        # Run LangGraph in streaming mode
        # ----------------------------------------------------

        for step_output in marine_graph.stream(
            initial_state,
            stream_mode="updates",
        ):

            if not isinstance(step_output, dict):
                continue

            # Each update normally looks like:
            #
            # {
            #     "planner": {
            #         ...
            #     }
            # }
            #
            # or another node name.

            for node_name, node_update in step_output.items():

                node_name = str(node_name)

                completed_nodes.append(node_name)

                # ------------------------------------------------
                # Update our final state
                # ------------------------------------------------

                if isinstance(node_update, dict):
                    final_state.update(node_update)

                # ------------------------------------------------
                # Send node completion event
                # ------------------------------------------------

                event: dict[str, Any] = {
                    "type": "node_done",
                    "node": node_name,
                }

                # ------------------------------------------------
                # If planner completed, send its plan
                # ------------------------------------------------

                if node_name == "planner":

                    event["plan"] = _clean_for_json(
                        _parse_plan(final_state)
                    )

                yield _sse(event)

        # ----------------------------------------------------
        # Graph completed
        # ----------------------------------------------------

        duration = round(
            time.monotonic() - started,
            2,
        )

        final_event = {
            "type": "final",
            "status": "SUCCESS",

            "latitude": payload.latitude,
            "longitude": payload.longitude,

            "question": payload.question,

            "plan": _clean_for_json(
                _parse_plan(final_state)
            ),

            "completed_nodes": completed_nodes,

            "recommendation": _clean_for_json(
                _normalize_recommendation(final_state)
            ),

            "duration_seconds": duration,
        }

        yield _sse(final_event)

    except Exception as exc:

        traceback.print_exc()

        error_event = {
            "type": "error",
            "status": "ERROR",
            "detail": str(exc),
        }

        yield _sse(error_event)


# ============================================================
# STREAMING ENDPOINT
# ============================================================

@app.post("/api/ask/stream")
def ask_stream(payload: AskRequest):

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
# OPTIONAL API INFORMATION
# ============================================================

@app.get("/api")
def api_info():

    return {
        "name": "ORCA Marine Intelligence API",
        "version": "1.0.0",
        "endpoints": {
            "frontend": "/",
            "health": "/health",
            "ask": "/api/ask",
            "stream": "/api/ask/stream",
        },
    }
