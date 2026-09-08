import json
import time
import traceback
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from graph import marine_graph


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Your repository structure is:
#
# src/
# ├── server.py
# ├── static/
# │   └── index.html
# ├── graph.py
# ├── schemas.py
# └── ...
#
# Therefore static files live here:
STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="ORCA Marine Intelligence API",
    description=(
        "Agentic AI platform for ocean, weather, tide, "
        "cyclone and ecosystem analysis."
    ),
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

# IMPORTANT:
# Do NOT import AskRequest from schemas.py.
# Your current schemas.py does not contain AskRequest.
#
# The frontend sends:
#
# {
#     "latitude": 19.076,
#     "longitude": 72.8777,
#     "question": "Is it safe to venture into the sea tomorrow morning?"
# }

class AskRequest(BaseModel):
    latitude: float = Field(
        ...,
        description="Latitude of the requested location",
    )

    longitude: float = Field(
        ...,
        description="Longitude of the requested location",
    )

    question: str = Field(
        ...,
        min_length=1,
        description="User's marine-related question",
    )


# ============================================================
# BASIC ROUTES
# ============================================================

@app.get("/")
def home():
    """
    Serve the ORCAWA frontend.

    Expected structure:

    src/
    ├── server.py
    └── static/
        └── index.html
    """

    # Primary location
    if INDEX_FILE.exists():
        return FileResponse(
            str(INDEX_FILE),
            media_type="text/html",
        )

    # Extra fallback:
    # This makes deployment more tolerant if index.html
    # accidentally exists beside server.py.
    fallback_index = BASE_DIR / "index.html"

    if fallback_index.exists():
        return FileResponse(
            str(fallback_index),
            media_type="text/html",
        )

    return {
        "status": "error",
        "message": "index.html not found",
        "expected_paths": [
            str(INDEX_FILE),
            str(fallback_index),
        ],
        "base_dir": str(BASE_DIR),
        "static_dir": str(STATIC_DIR),
        "static_exists": STATIC_DIR.exists(),
    }


@app.get("/health")
def health():
    """
    Health check endpoint for Render.
    """

    return {
        "status": "ok",
        "service": "ORCA Marine Intelligence API",
    }


@app.get("/api")
def api_info():
    """
    API information.
    """

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


# ============================================================
# JSON / SERIALIZATION HELPERS
# ============================================================

def _clean_for_json(value: Any) -> Any:
    """
    Convert potentially complex Python objects into
    JSON-safe values.

    Handles:
    - dict
    - list
    - tuple
    - Pydantic models
    - primitive values
    - custom objects
    """

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {
            str(key): _clean_for_json(item)
            for key, item in value.items()
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

    # Pydantic v2
    if hasattr(value, "model_dump"):
        try:
            return _clean_for_json(
                value.model_dump()
            )
        except Exception:
            pass

    # Pydantic v1
    if hasattr(value, "dict"):
        try:
            return _clean_for_json(
                value.dict()
            )
        except Exception:
            pass

    return str(value)


def _sse(event: dict[str, Any]) -> str:
    """
    Convert a Python dictionary into a Server-Sent Event.

    Example:

    data: {"type":"node_done","node":"planner"}

    """

    safe_event = _clean_for_json(event)

    return (
        f"data: {json.dumps(safe_event, ensure_ascii=False)}"
        "\n\n"
    )


# ============================================================
# STATE EXTRACTION HELPERS
# ============================================================

def _parse_plan(state: dict[str, Any]) -> Any:
    """
    Safely extract planner output.

    Different versions of the planner may store the plan
    under different keys.
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

        if value is None:
            continue

        # Already structured
        if isinstance(value, (dict, list)):
            return value

        # JSON string
        if isinstance(value, str):

            cleaned = value.strip()

            if not cleaned:
                continue

            try:
                return json.loads(cleaned)

            except json.JSONDecodeError:
                return cleaned

    return None


def _normalize_recommendation(
    state: dict[str, Any],
) -> Any:
    """
    Safely extract final recommendation.
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


# ============================================================
# INITIAL GRAPH STATE
# ============================================================

def _build_initial_state(
    payload: AskRequest,
) -> dict[str, Any]:
    """
    Build the state expected by the marine graph.
    """

    return {
        "latitude": payload.latitude,
        "longitude": payload.longitude,
        "user_question": payload.question.strip(),
        "status": "STARTED",
    }


# ============================================================
# NORMAL / NON-STREAMING ASK ENDPOINT
# ============================================================

@app.post("/api/ask")
def ask(payload: AskRequest):
    """
    Execute the complete marine LangGraph and return
    the final result.
    """

    initial_state = _build_initial_state(payload)

    started = time.monotonic()

    try:

        # ----------------------------------------------------
        # Execute graph
        # ----------------------------------------------------

        final_state = marine_graph.invoke(
            initial_state
        )

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if not isinstance(final_state, dict):
            final_state = dict(initial_state)

        # ----------------------------------------------------
        # Duration
        # ----------------------------------------------------

        duration = round(
            time.monotonic() - started,
            2,
        )

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

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

            "state": _clean_for_json(
                final_state
            ),

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

def _stream_ask_events(
    payload: AskRequest,
) -> Iterator[str]:
    """
    Execute LangGraph using stream_mode='updates'
    and convert every completed node into an SSE event.

    The frontend expects events like:

    {
        "type": "node_done",
        "node": "planner"
    }

    and finally:

    {
        "type": "final",
        ...
    }
    """

    initial_state = _build_initial_state(payload)

    started = time.monotonic()

    final_state: dict[str, Any] = dict(
        initial_state
    )

    completed_nodes: list[str] = []

    final_sent = False

    # --------------------------------------------------------
    # Tell frontend that stream has started
    # --------------------------------------------------------

    yield _sse({
        "type": "stream_started",
        "status": "STARTED",
    })

    try:

        # ----------------------------------------------------
        # Stream LangGraph
        # ----------------------------------------------------

        stream = marine_graph.stream(
            initial_state,
            stream_mode="updates",
        )

        for step_output in stream:

            # ------------------------------------------------
            # Ignore unexpected output
            # ------------------------------------------------

            if not isinstance(step_output, dict):
                continue

            # ------------------------------------------------
            # LangGraph update usually looks like:
            #
            # {
            #     "planner": {
            #         ...
            #     }
            # }
            #
            # or:
            #
            # {
            #     "weather": {
            #         ...
            #     }
            # }
            # ------------------------------------------------

            for node_name, node_update in step_output.items():

                node_name = str(node_name)

                # Avoid duplicate completion events
                if node_name not in completed_nodes:
                    completed_nodes.append(node_name)

                # ------------------------------------------------
                # Merge graph state
                # ------------------------------------------------

                if isinstance(node_update, dict):

                    final_state.update(
                        node_update
                    )

                # ------------------------------------------------
                # Build node completion event
                # ------------------------------------------------

                event: dict[str, Any] = {
                    "type": "node_done",
                    "node": node_name,
                }

                # ------------------------------------------------
                # Planner event
                # ------------------------------------------------

                if node_name == "planner":

                    event["plan"] = _clean_for_json(
                        _parse_plan(final_state)
                    )

                # ------------------------------------------------
                # Send event immediately
                # ------------------------------------------------

                yield _sse(event)

                # ------------------------------------------------
                # Small heartbeat event.
                #
                # This helps keep the HTTP stream alive on
                # deployment platforms/proxies.
                # ------------------------------------------------

                yield _sse({
                    "type": "heartbeat",
                    "node": node_name,
                })

        # ====================================================
        # GRAPH COMPLETED SUCCESSFULLY
        # ====================================================

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

        final_sent = True

        yield _sse(final_event)

    except Exception as exc:

        # ----------------------------------------------------
        # IMPORTANT:
        # Never allow the generator to silently terminate.
        # The frontend otherwise displays:
        #
        # "The pipeline stream ended without a final result."
        # ----------------------------------------------------

        traceback.print_exc()

        duration = round(
            time.monotonic() - started,
            2,
        )

        # Send explicit error event
        yield _sse({
            "type": "error",
            "status": "ERROR",
            "detail": str(exc),
            "completed_nodes": completed_nodes,
            "duration_seconds": duration,
        })

        # Do NOT send another final event after an error.
        # The frontend already handles the error event.


# ============================================================
# STREAMING ENDPOINT
# ============================================================

@app.post("/api/ask/stream")
def ask_stream(payload: AskRequest):
    """
    Streaming endpoint used by the ORCAWA frontend.
    """

    return StreamingResponse(
        _stream_ask_events(payload),

        media_type="text/event-stream",

        headers={
            # Prevent proxy buffering
            "X-Accel-Buffering": "no",

            # Do not cache stream
            "Cache-Control": (
                "no-cache, no-store, "
                "must-revalidate"
            ),

            # Keep HTTP connection alive
            "Connection": "keep-alive",

            # Explicit SSE header
            "Content-Type": (
                "text/event-stream; "
                "charset=utf-8"
            ),
        },
    )
