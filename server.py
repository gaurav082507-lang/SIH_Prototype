import json
import time
import traceback
from pathlib import Path
from typing import Any, Iterator
from threading import Thread
from queue import Queue, Empty

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
# REQUEST MODEL
# ============================================================

# IMPORTANT:
# AskRequest is defined here.
#
# Do NOT use:
# from schemas import AskRequest
#
# because your current schemas.py does not expose AskRequest.

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
    Serve ORCAWA frontend.

    Expected Render structure:

    src/
    ├── server.py
    ├── static/
    │   └── index.html
    ├── graph.py
    └── ...
    """

    if INDEX_FILE.exists():
        return FileResponse(
            str(INDEX_FILE),
            media_type="text/html",
        )

    # Extra fallback
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
    Render health check.
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
# JSON HELPERS
# ============================================================

def _clean_for_json(value: Any) -> Any:
    """
    Convert arbitrary Python values into JSON-safe values.
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
    Convert a dictionary into a Server-Sent Event.

    Example:

    data: {"type":"node_done","node":"planner"}

    """

    safe_event = _clean_for_json(event)

    return (
        "data: "
        + json.dumps(
            safe_event,
            ensure_ascii=False,
        )
        + "\n\n"
    )


# ============================================================
# STATE HELPERS
# ============================================================

def _build_initial_state(
    payload: AskRequest,
) -> dict[str, Any]:
    """
    Build the initial LangGraph state.

    This keeps the same contract as the existing server.
    """

    return {
        "latitude": payload.latitude,
        "longitude": payload.longitude,
        "user_question": payload.question.strip(),
        "status": "STARTED",
    }


def _parse_plan(
    state: dict[str, Any],
) -> Any:
    """
    Extract planner output from graph state.
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
    Extract final recommendation from graph state.
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
# NORMAL ASK ENDPOINT
# ============================================================

@app.post("/api/ask")
def ask(payload: AskRequest):
    """
    Normal non-streaming graph execution.
    """

    initial_state = _build_initial_state(
        payload
    )

    started = time.monotonic()

    try:

        final_state = marine_graph.invoke(
            initial_state
        )

        if not isinstance(final_state, dict):
            final_state = dict(
                initial_state
            )

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
                _normalize_recommendation(
                    final_state
                )
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
# BACKGROUND GRAPH WORKER
# ============================================================

def _run_graph_worker(
    initial_state: dict[str, Any],
    event_queue: Queue,
):
    """
    Run LangGraph in a background thread.

    Why?

    Previously:

        HTTP request
              |
              v
        marine_graph.stream()
              |
              v
        wait...

    If a node took a long time, nothing was sent to
    the browser during that period.

    Now:

        HTTP request
              |
              +----------------------+
              |                      |
              v                      v
        SSE generator          Graph worker
              |                      |
              |                      v
              |                marine_graph
              |                      |
              +<----- Queue <--------+
    """

    try:

        for step_output in marine_graph.stream(
            initial_state,
            stream_mode="updates",
        ):

            event_queue.put(
                (
                    "graph_update",
                    step_output,
                )
            )

        # Graph finished normally
        event_queue.put(
            (
                "graph_finished",
                None,
            )
        )

    except Exception as exc:

        traceback.print_exc()

        event_queue.put(
            (
                "graph_error",
                exc,
            )
        )


# ============================================================
# STREAMING ENDPOINT
# ============================================================

def _stream_ask_events(
    payload: AskRequest,
) -> Iterator[str]:
    """
    Robust Server-Sent Event stream.

    This version:

    1. Immediately tells frontend the stream started.
    2. Runs LangGraph in a background thread.
    3. Sends heartbeat events while graph nodes are running.
    4. Sends node_done when a LangGraph update arrives.
    5. Sends planner data when planner completes.
    6. Sends final when graph finishes.
    7. Sends error if graph crashes.
    """

    initial_state = _build_initial_state(
        payload
    )

    started = time.monotonic()

    final_state: dict[str, Any] = dict(
        initial_state
    )

    completed_nodes: list[str] = []

    event_queue: Queue = Queue()

    # ========================================================
    # START EVENT
    # ========================================================

    yield _sse({
        "type": "stream_started",
        "status": "STARTED",
        "latitude": payload.latitude,
        "longitude": payload.longitude,
    })

    # ========================================================
    # START GRAPH WORKER
    # ========================================================

    worker = Thread(
        target=_run_graph_worker,
        args=(
            initial_state,
            event_queue,
        ),
        daemon=True,
    )

    worker.start()

    # ========================================================
    # STREAM LOOP
    # ========================================================

    graph_finished = False

    while not graph_finished:

        try:

            # Wait at most 2 seconds.
            #
            # If no graph update arrives, we send
            # a heartbeat instead of allowing the HTTP
            # connection to remain silent.

            event_type, data = event_queue.get(
                timeout=2.0
            )

        except Empty:

            elapsed = round(
                time.monotonic() - started,
                1,
            )

            yield _sse({
                "type": "heartbeat",
                "status": "RUNNING",
                "elapsed_seconds": elapsed,
                "completed_nodes": completed_nodes,
            })

            continue

        # ====================================================
        # GRAPH UPDATE
        # ====================================================

        if event_type == "graph_update":

            step_output = data

            if not isinstance(
                step_output,
                dict,
            ):
                continue

            for node_name, node_update in (
                step_output.items()
            ):

                node_name = str(
                    node_name
                )

                # Avoid duplicate node names
                if node_name not in completed_nodes:
                    completed_nodes.append(
                        node_name
                    )

                # --------------------------------------------
                # Merge graph state
                # --------------------------------------------

                if isinstance(
                    node_update,
                    dict,
                ):

                    final_state.update(
                        node_update
                    )

                # --------------------------------------------
                # Build node event
                # --------------------------------------------

                event = {
                    "type": "node_done",
                    "node": node_name,
                }

                # --------------------------------------------
                # Planner result
                # --------------------------------------------

                if node_name == "planner":

                    event["plan"] = (
                        _clean_for_json(
                            _parse_plan(
                                final_state
                            )
                        )
                    )

                # --------------------------------------------
                # Send node event
                # --------------------------------------------

                yield _sse(event)

        # ====================================================
        # GRAPH FINISHED
        # ====================================================

        elif event_type == "graph_finished":

            graph_finished = True

        # ====================================================
        # GRAPH ERROR
        # ====================================================

        elif event_type == "graph_error":

            exc = data

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

            # Also send a final event with ERROR status.
            #
            # This is useful because the frontend is designed
            # around the final event as the terminal state.

            yield _sse({
                "type": "final",
                "status": "ERROR",

                "latitude": payload.latitude,
                "longitude": payload.longitude,

                "question": payload.question,

                "plan": _clean_for_json(
                    _parse_plan(
                        final_state
                    )
                ),

                "completed_nodes": completed_nodes,

                "recommendation": None,

                "duration_seconds": duration,

                "error": str(exc),
            })

            return

    # ========================================================
    # GRAPH SUCCESSFULLY COMPLETED
    # ========================================================

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
            _normalize_recommendation(
                final_state
            )
        ),

        "duration_seconds": duration,
    }

    yield _sse(final_event)


# ============================================================
# STREAM API
# ============================================================

@app.post("/api/ask/stream")
def ask_stream(
    payload: AskRequest,
):
    """
    Streaming ORCAWA assessment endpoint.
    """

    return StreamingResponse(

        _stream_ask_events(
            payload
        ),

        media_type="text/event-stream",

        headers={

            # Never cache SSE
            "Cache-Control": (
                "no-cache, "
                "no-store, "
                "must-revalidate"
            ),

            # Disable proxy buffering
            "X-Accel-Buffering": "no",

            # Keep connection alive
            "Connection": "keep-alive",

            # Explicit SSE content type
            "Content-Type": (
                "text/event-stream; "
                "charset=utf-8"
            ),

            # Helps prevent some intermediaries
            # from buffering the response.
            "X-Content-Type-Options": "nosniff",
        },
    )
