# server.py
#
# ORCAWA / ORCA Marine Intelligence API.
#
# This module is a thin HTTP layer over the LangGraph pipeline in
# graph.py. It deliberately owns NO pipeline logic of its own:
#
#   * The node names come from graph.py and are checked against the
#     compiled graph at import time.
#   * Which specialists will run for a given plan is answered by
#     graph.py's own `route_after_planner`, not by a second copy of
#     that logic living here. The frontend therefore cannot drift out
#     of sync with the graph.
#   * The state keys it reads back (plan / *_data / recommendation)
#     are exactly the keys declared in state.MarineState.
#
# Endpoints
#   GET  /                  ORCAWA frontend (static/index.html)
#   GET  /health            liveness probe
#   GET  /api/health        same probe (Render's healthCheckPath)
#   GET  /api               API metadata
#   GET  /api/graph         pipeline topology, for the UI flow diagram
#   POST /api/ask           run the graph, return once (blocking)
#   POST /api/ask/stream    run the graph, stream progress over SSE

from __future__ import annotations

import asyncio
import json
import os
import time
import traceback
from pathlib import Path
from threading import Thread
from typing import Any, AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from graph import marine_graph, route_after_planner

try:
    # Optional: present once graph.py wraps its nodes with deadlines.
    # Guarded so an older graph.py cannot stop the app from booting.
    from node_timeout import all_timeouts
except ImportError:  # pragma: no cover
    all_timeouts = None


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"


# ============================================================
# TUNABLES
# ============================================================

# How often to emit a keep-alive event while the graph is working.
# Render and most proxies drop an idle response after ~60s, so this
# has to stay well below that.
HEARTBEAT_INTERVAL_S = float(os.getenv("SSE_HEARTBEAT_SECONDS", "2.0"))

# Hard ceiling on a single assessment. Without this a hung upstream
# marine service would keep an SSE connection alive forever.
GRAPH_TIMEOUT_S = float(os.getenv("GRAPH_TIMEOUT_SECONDS", "300"))

# Set SSE_LOG=0 to silence the per-run progress lines. They are on by
# default because a stalled assessment is otherwise invisible in the
# Render logs.
SSE_LOG = os.getenv("SSE_LOG", "1").strip().lower() not in ("0", "false", "no")


def _log(message: str) -> None:
    if SSE_LOG:
        print(f"[stream] {message}", flush=True)


# ============================================================
# GRAPH TOPOLOGY
# ============================================================
#
# Mirrors graph.py:
#
#     START -> planner
#     planner -> (route_after_planner) -> gis + selected specialists
#                                      -> recommendation (if rejected)
#     every specialist -> recommendation
#     recommendation -> END

PLANNER_NODE = "planner"
FINAL_NODE = "recommendation"

# Order here is display order in the UI. gis first because graph.py
# runs it for every accepted query.
SPECIALIST_NODES: tuple[str, ...] = (
    "gis",
    "weather",
    "ocean",
    "tide",
    "cyclone",
    "ecosystem",
    "pfz",
)

ALL_NODES: tuple[str, ...] = (PLANNER_NODE, *SPECIALIST_NODES, FINAL_NODE)

# state.MarineState key written by each node.
NODE_STATE_KEY: dict[str, str] = {
    PLANNER_NODE: "plan",
    FINAL_NODE: "recommendation",
    **{node: f"{node}_data" for node in SPECIALIST_NODES},
}

# Presentation metadata, served to the frontend so the flow diagram is
# built from the graph rather than from a hardcoded copy of it.
NODE_LABELS: dict[str, dict[str, str]] = {
    "planner": {"label": "Planner", "label_hi": "योजना"},
    "gis": {"label": "Coastal check", "label_hi": "तटीय जांच"},
    "weather": {"label": "Weather", "label_hi": "मौसम"},
    "ocean": {"label": "Sea state", "label_hi": "समुद्री स्थिति"},
    "tide": {"label": "Tide", "label_hi": "ज्वार-भाटा"},
    "cyclone": {"label": "Cyclone", "label_hi": "चक्रवात"},
    "ecosystem": {"label": "Ecosystem", "label_hi": "पारिस्थितिकी"},
    "pfz": {"label": "Fishing zone", "label_hi": "मछली क्षेत्र"},
    "recommendation": {"label": "Final assessment", "label_hi": "आकलन"},
}


def _compiled_node_names() -> set[str]:
    """
    Node names actually present in the compiled graph.

    Used only to warn on drift — if graph.py gains or loses a node and
    this module isn't updated, the log says so at boot instead of the
    UI quietly showing a stale diagram.
    """

    try:
        return {
            str(name)
            for name in marine_graph.get_graph().nodes
            if not str(name).startswith("__")
        }
    except Exception:  # pragma: no cover - depends on langgraph internals
        return set()


def _warn_on_topology_drift() -> None:
    compiled = _compiled_node_names()

    if not compiled:
        return

    declared = set(ALL_NODES)

    missing_here = sorted(compiled - declared)
    missing_there = sorted(declared - compiled)

    if missing_here:
        print(
            "[server] WARNING: graph.py has nodes this server does not "
            f"know about: {missing_here}"
        )

    if missing_there:
        print(
            "[server] WARNING: this server lists nodes that are not in "
            f"graph.py: {missing_there}"
        )


_warn_on_topology_drift()


def _graph_topology() -> dict[str, Any]:
    """Serializable description of the pipeline, for GET /api/graph."""

    return {
        "entry": PLANNER_NODE,
        "final": FINAL_NODE,
        "specialists": list(SPECIALIST_NODES),
        "always_run": ["gis"],
        "nodes": [
            {
                "id": node,
                "kind": (
                    "planner"
                    if node == PLANNER_NODE
                    else "final"
                    if node == FINAL_NODE
                    else "specialist"
                ),
                "state_key": NODE_STATE_KEY[node],
                **NODE_LABELS.get(node, {"label": node, "label_hi": ""}),
            }
            for node in ALL_NODES
        ],
        "edges": (
            [{"from": PLANNER_NODE, "to": node} for node in SPECIALIST_NODES]
            + [{"from": PLANNER_NODE, "to": FINAL_NODE, "when": "rejected"}]
            + [{"from": node, "to": FINAL_NODE} for node in SPECIALIST_NODES]
        ),
    }


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="ORCA Marine Intelligence API",
    description=(
        "Agentic AI platform for ocean, weather, tide, "
        "cyclone and ecosystem analysis."
    ),
    # Bump this whenever the wire behaviour changes — /api/health is the
    # only way to tell from outside which build Render is actually running.
    version="1.5.0",
)


# ------------------------------------------------------------
# CORS
# ------------------------------------------------------------
# render.yaml sets ALLOWED_ORIGINS; without this middleware that
# variable did nothing. Same-origin deployments are unaffected.

_raw_origins = os.getenv("ALLOWED_ORIGINS", "*").strip()

if _raw_origins in ("", "*"):
    _allow_origins = ["*"]
    _allow_credentials = False  # "*" + credentials is rejected by browsers
else:
    _allow_origins = [
        origin.strip() for origin in _raw_origins.split(",") if origin.strip()
    ]
    _allow_credentials = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ------------------------------------------------------------
# STATIC FILES
# ------------------------------------------------------------

if STATIC_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(STATIC_DIR)),
        name="static",
    )


# ============================================================
# REQUEST MODEL
# ============================================================

MAX_QUESTION_CHARS = 600


class AskRequest(BaseModel):
    """
    Body for POST /api/ask and POST /api/ask/stream.

    Defined here rather than imported from schemas.py — schemas.py
    validates LLM output, not HTTP input, and does not expose this
    model.
    """

    latitude: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="Latitude of the requested location",
    )

    longitude: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="Longitude of the requested location",
    )

    question: str = Field(
        ...,
        min_length=1,
        max_length=MAX_QUESTION_CHARS,
        description="User's marine-related question",
    )


# ============================================================
# BASIC ROUTES
# ============================================================

_NO_STORE = {
    "Cache-Control": "no-store, no-cache, must-revalidate",
    "Pragma": "no-cache",
}


@app.get("/", include_in_schema=False)
def home():
    """
    Serve the ORCAWA frontend.

    Expected layout:

        repo-root/
        ├── server.py
        ├── graph.py
        └── static/
            └── index.html
    """

    for candidate in (INDEX_FILE, BASE_DIR / "index.html"):
        if candidate.exists():
            # no-store: the frontend is versioned by deploy, and a
            # cached copy against a newer API is the classic
            # "it works locally" bug.
            return FileResponse(
                str(candidate),
                media_type="text/html",
                headers=_NO_STORE,
            )

    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "index.html not found",
            "expected_paths": [
                str(INDEX_FILE),
                str(BASE_DIR / "index.html"),
            ],
            "base_dir": str(BASE_DIR),
            "static_dir": str(STATIC_DIR),
            "static_exists": STATIC_DIR.exists(),
        },
    )


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Silence the browser's automatic favicon request."""

    icon = STATIC_DIR / "favicon.ico"

    if icon.exists():
        return FileResponse(str(icon))

    return Response(status_code=204)


def _rss_mb() -> float | None:
    """
    Resident memory of this process, in MB.

    A container that is killed for exceeding its memory limit dies by
    SIGKILL: no traceback, no Python-level handler, and any open SSE
    stream simply closes. Reporting RSS is the only way to see that
    coming from outside.
    """

    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024.0, 1)
    except Exception:
        pass

    try:
        import resource

        # ru_maxrss is KB on Linux, bytes on macOS.
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round((peak / 1024.0) if peak > 1_000_000 else peak / 1024.0, 1)
    except Exception:
        return None


def _health_payload() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "ORCA Marine Intelligence API",
        "version": app.version,
        # "asyncio-queue" means the SSE loop no longer polls the ASGI
        # receive channel. If you see "polling" here, the older build is
        # still deployed.
        "stream_mode": "asyncio-queue",
        "heartbeat_seconds": HEARTBEAT_INTERVAL_S,
        "graph_timeout_seconds": GRAPH_TIMEOUT_S,
        "memory_rss_mb": _rss_mb(),
        "node_timeouts": all_timeouts(ALL_NODES) if all_timeouts else None,
        "graph_nodes": list(ALL_NODES),
        "frontend": INDEX_FILE.exists(),
    }


# Both paths are registered on purpose: render.yaml points its
# healthCheckPath at /api/health, while /health is the conventional
# probe path. Previously only /health existed, so Render's health
# check 404'd on every deploy.
@app.get("/health")
def health():
    """Liveness probe."""

    return _health_payload()


@app.get("/api/health")
def api_health():
    """Liveness probe (Render healthCheckPath)."""

    return _health_payload()


@app.get("/api")
def api_info():
    """API metadata."""

    return {
        "name": "ORCA Marine Intelligence API",
        "version": app.version,
        "endpoints": {
            "frontend": "/",
            "health": "/api/health",
            "selftest": "/api/selftest",
            "graph": "/api/graph",
            "ask": "/api/ask",
            "stream": "/api/ask/stream",
        },
    }


@app.get("/api/selftest")
def selftest(
    step: str = "all",
    latitude: float = 19.0760,
    longitude: float = 72.8777,
):
    """
    Isolate which part of the planner is killing the process.

    The SSE stream closing mid-run with heartbeats still flowing means
    the worker process died without raising — which a Python handler
    cannot catch and a traceback cannot show. Run the planner's steps
    one at a time here and watch `rss_mb_after`:

        /api/selftest?step=mem       - baseline memory, imports only
        /api/selftest?step=coastal   - the land-mask coastal check
        /api/selftest?step=planner   - the full planner node (LLM call)

    If a step never returns and the service restarts, that step is the
    one exhausting the instance.
    """

    order = ["mem", "coastal", "planner"]
    steps = order if step == "all" else [step]

    results: list[dict[str, Any]] = []

    for name in steps:
        started = time.monotonic()
        entry: dict[str, Any] = {"step": name, "rss_mb_before": _rss_mb()}

        try:
            if name == "mem":
                entry["ok"] = True

            elif name == "coastal":
                from tools import check_coastal_proximity

                entry["result"] = _clean_for_json(
                    check_coastal_proximity.invoke(
                        {
                            "latitude": latitude,
                            "longitude": longitude,
                            "max_radius_km": 50,
                        }
                    )
                )
                entry["ok"] = True

            elif name == "planner":
                from planner_node import planner_node

                output = planner_node(
                    {
                        "latitude": latitude,
                        "longitude": longitude,
                        "user_question": (
                            "Is it safe to venture into the sea tomorrow morning?"
                        ),
                        "status": "STARTED",
                    }
                )

                plan = _coerce_plan((output or {}).get("plan"))

                entry["ok"] = True
                entry["rejected"] = bool(plan.get("rejected", False))
                entry["rejection_reason"] = plan.get("rejection_reason")
                entry["required_agents"] = plan.get("required_agents")
                entry["grid_points"] = len(plan.get("grid_points") or [])

            else:
                entry["ok"] = False
                entry["error"] = f"unknown step {name!r} (try: {', '.join(order)})"

        except Exception as exc:
            traceback.print_exc()
            entry["ok"] = False
            entry["error"] = f"{type(exc).__name__}: {exc}"

        entry["seconds"] = round(time.monotonic() - started, 2)
        entry["rss_mb_after"] = _rss_mb()
        results.append(entry)

    return {"steps": results}


@app.get("/api/graph")
def api_graph():
    """
    Pipeline topology.

    The frontend builds its flow diagram from this, so adding a
    specialist to graph.py and to SPECIALIST_NODES is enough — no
    frontend edit required.
    """

    return _graph_topology()


# ============================================================
# JSON HELPERS
# ============================================================

_MAX_JSON_DEPTH = 24


def _clean_for_json(value: Any, _depth: int = 0, _seen: set[int] | None = None) -> Any:
    """
    Convert arbitrary Python values into JSON-safe values.

    Depth- and cycle-guarded: agent envelopes are nested dicts built
    from third-party responses, and one self-referencing object used
    to be enough to blow the stack inside the SSE generator (which
    surfaces as a truncated stream, not a clean error).
    """

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if _depth >= _MAX_JSON_DEPTH:
        return str(value)

    if _seen is None:
        _seen = set()

    marker = id(value)

    if marker in _seen:
        return "<circular reference>"

    _seen = _seen | {marker}

    if isinstance(value, dict):
        return {
            str(key): _clean_for_json(item, _depth + 1, _seen)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set, frozenset)):
        return [_clean_for_json(item, _depth + 1, _seen) for item in value]

    # Pydantic v2 then v1.
    for method in ("model_dump", "dict"):
        dumper = getattr(value, method, None)

        if callable(dumper):
            try:
                return _clean_for_json(dumper(), _depth + 1, _seen)
            except Exception:
                pass

    return str(value)


def _sse(event: dict[str, Any]) -> str:
    """
    Render one Server-Sent Event.

        data: {"type":"node_done","node":"planner"}

    """

    return (
        "data: "
        + json.dumps(_clean_for_json(event), ensure_ascii=False)
        + "\n\n"
    )


# ============================================================
# STATE HELPERS
# ============================================================


def _build_initial_state(payload: AskRequest) -> dict[str, Any]:
    """Build the initial MarineState."""

    return {
        "latitude": payload.latitude,
        "longitude": payload.longitude,
        "user_question": payload.question.strip(),
        "status": "STARTED",
    }


def _coerce_plan(value: Any) -> dict[str, Any]:
    """
    Normalize state["plan"] into a dict.

    planner_node.py returns a dict, but it builds that dict from LLM
    output, and a stringified plan has been seen in the wild. graph.py
    tolerates both, so this does too.
    """

    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        text = value.strip()

        if text:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return {}

            if isinstance(parsed, dict):
                return parsed

    return {}


def _plan_of(state: dict[str, Any]) -> dict[str, Any]:
    return _coerce_plan(state.get("plan"))


def _recommendation_of(state: dict[str, Any]) -> Any:
    return state.get("recommendation")


def _agents_of(state: dict[str, Any]) -> dict[str, Any]:
    """Every specialist envelope currently present in the state."""

    return {
        node: state[NODE_STATE_KEY[node]]
        for node in SPECIALIST_NODES
        if state.get(NODE_STATE_KEY[node]) is not None
    }


def _planned_nodes(plan: dict[str, Any]) -> list[str]:
    """
    Which nodes graph.py will run for this plan.

    Delegates to graph.route_after_planner so there is exactly one
    definition of the routing rules. This is what lets the UI grey out
    skipped specialists correctly — including the rejected case, where
    the graph goes straight to `recommendation` and even gis does NOT
    run.
    """

    try:
        routes = route_after_planner({"plan": plan})
    except Exception:
        traceback.print_exc()
        # Unknown routing: assume everything, so the UI shows nodes as
        # pending rather than wrongly marking them skipped.
        return list(ALL_NODES)

    if isinstance(routes, str):
        routes = [routes]

    if not isinstance(routes, (list, tuple, set)):
        return list(ALL_NODES)

    selected = {str(route).strip().lower() for route in routes}

    # Keep canonical display order.
    return [node for node in ALL_NODES if node in selected]


def _status_of_envelope(value: Any) -> str | None:
    """Read the SUCCESS / PARTIAL / FAILED marker out of an envelope."""

    if isinstance(value, dict):
        status = value.get("status")

        if isinstance(status, str) and status.strip():
            return status.strip().upper()

    return None


def _error_of_envelope(value: Any) -> str | None:
    """Best-effort human-readable error out of an agent envelope."""

    if not isinstance(value, dict):
        return None

    for key in ("error", "message", "reason", "detail"):
        candidate = value.get(key)

        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    errors = value.get("errors")

    if isinstance(errors, list) and errors:
        first = errors[0]

        if isinstance(first, str):
            return first

        if isinstance(first, dict):
            for key in ("message", "error", "detail", "reason"):
                candidate = first.get(key)

                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()

    return None


def _node_event(
    node: str,
    state: dict[str, Any],
    completed: list[str],
    elapsed: float,
) -> dict[str, Any]:
    """
    Build the node_done event for one finished node.

    Every node carries its own status so the UI can distinguish
    "ran and returned data" from "ran and failed" — previously a
    failed agent was rendered exactly like a successful one.
    """

    state_key = NODE_STATE_KEY.get(node)
    value = state.get(state_key) if state_key else None

    event: dict[str, Any] = {
        "type": "node_done",
        "node": node,
        "state_key": state_key,
        "completed_nodes": list(completed),
        "elapsed_seconds": round(elapsed, 1),
    }

    if node == PLANNER_NODE:
        plan = _coerce_plan(value)
        planned = _planned_nodes(plan)

        event["plan"] = plan
        event["rejected"] = bool(plan.get("rejected", False))
        event["timed_out"] = bool(plan.get("timed_out", False))
        event["rejection_reason"] = plan.get("rejection_reason")
        event["planned_nodes"] = planned
        event["skipped_nodes"] = [
            candidate for candidate in SPECIALIST_NODES if candidate not in planned
        ]
        event["node_status"] = "REJECTED" if event["rejected"] else "PLANNED"

        return event

    if node == FINAL_NODE:
        event["node_status"] = _status_of_envelope(value) or (
            "SUCCESS" if value else "FAILED"
        )

        return event

    event["node_status"] = _status_of_envelope(value) or "SUCCESS"
    event["error"] = _error_of_envelope(value)

    return event


def _final_event(
    payload: AskRequest,
    state: dict[str, Any],
    completed: list[str],
    duration: float,
    status: str = "SUCCESS",
    error: str | None = None,
) -> dict[str, Any]:
    """Terminal event. The frontend treats this as the end of the run."""

    plan = _plan_of(state)

    return {
        "type": "final",
        "status": status,
        "latitude": payload.latitude,
        "longitude": payload.longitude,
        "question": payload.question,
        "plan": plan,
        "rejected": bool(plan.get("rejected", False)),
        "timed_out": bool(plan.get("timed_out", False)),
        "rejection_reason": plan.get("rejection_reason"),
        "recommendation": _recommendation_of(state),
        # Raw per-specialist envelopes, straight from MarineState. The
        # UI's technical readout used to rely on the LLM populating
        # recommendation.agent_findings, which is frequently empty.
        "agents": _agents_of(state),
        "completed_nodes": list(completed),
        "duration_seconds": round(duration, 2),
        "error": error,
    }


# ============================================================
# GRAPH EXECUTION
# ============================================================


def _run_graph(initial_state: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """
    Run the graph to completion, collecting per-node updates.

    Uses stream() rather than invoke() so the same merge logic backs
    both endpoints and /api/ask can report completed_nodes too.
    """

    state = dict(initial_state)
    completed: list[str] = []

    for step in marine_graph.stream(initial_state, stream_mode="updates"):
        if not isinstance(step, dict):
            continue

        for node_name, node_update in step.items():
            node_name = str(node_name)

            if node_name not in completed:
                completed.append(node_name)

            if isinstance(node_update, dict):
                state.update(node_update)

    return state, completed


def _run_graph_worker(initial_state: dict[str, Any], emit) -> None:
    """
    Run the graph on a background thread, handing updates to `emit`.

    The SSE generator consumes them, so a slow node never leaves the
    HTTP response silent:

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
              +<----- emit() <-------+

    BaseException, not Exception: if this thread ever dies without
    emitting, the stream has nothing to report and the browser just
    sees the connection close — which is exactly the failure that is
    impossible to diagnose from the client side.
    """

    try:
        for step_output in marine_graph.stream(
            initial_state,
            stream_mode="updates",
        ):
            emit(("graph_update", step_output))

        emit(("graph_finished", None))

    except BaseException as exc:  # noqa: BLE001
        traceback.print_exc()

        try:
            emit(("graph_error", exc))
        except Exception:
            traceback.print_exc()


# ============================================================
# NON-STREAMING ENDPOINT
# ============================================================


@app.post("/api/ask")
def ask(payload: AskRequest):
    """
    Run the full graph and return once the assessment is ready.

    Typically 15-40s. Prefer /api/ask/stream in a browser.
    """

    initial_state = _build_initial_state(payload)
    started = time.monotonic()

    try:
        final_state, completed = _run_graph(initial_state)

    except Exception as exc:
        traceback.print_exc()

        # 500, not 200-with-an-error-body: a failed assessment that
        # answers 200 is invisible to every HTTP client, monitor and
        # `res.ok` check in the frontend.
        return JSONResponse(
            status_code=500,
            content=_clean_for_json(
                {
                    "status": "ERROR",
                    "message": str(exc),
                    "latitude": payload.latitude,
                    "longitude": payload.longitude,
                    "question": payload.question,
                    "duration_seconds": round(time.monotonic() - started, 2),
                }
            ),
        )

    duration = time.monotonic() - started
    plan = _plan_of(final_state)

    return _clean_for_json(
        {
            "status": "SUCCESS",
            "latitude": payload.latitude,
            "longitude": payload.longitude,
            "question": payload.question,
            "plan": plan,
            "rejected": bool(plan.get("rejected", False)),
            "rejection_reason": plan.get("rejection_reason"),
            "recommendation": _recommendation_of(final_state),
            "agents": _agents_of(final_state),
            "completed_nodes": completed,
            "state": final_state,
            "duration_seconds": round(duration, 2),
        }
    )


# ============================================================
# STREAMING ENDPOINT
# ============================================================


async def _stream_ask_events(payload: AskRequest) -> AsyncIterator[str]:
    """
    Server-Sent Event stream for one assessment.

    Event sequence:

        stream_started   once, immediately
        node_done        per node, carrying that node's own status
                         (the planner's also carries the plan and the
                         list of specialists the graph will actually run)
        heartbeat        every ~2s while nothing else is happening
        final            terminal, SUCCESS or ERROR
        error            only alongside a failure, before `final`
    """

    initial_state = _build_initial_state(payload)
    started = time.monotonic()

    final_state: dict[str, Any] = dict(initial_state)
    completed: list[str] = []

    # The worker runs in a thread; asyncio.Queue is not thread-safe, so
    # it is fed through call_soon_threadsafe. This replaces an earlier
    # design that polled a thread Queue and called
    # request.is_disconnected() on every tick — that hammered the ASGI
    # receive channel behind Starlette's own disconnect listener and
    # could tear the response down with no event ever reaching the
    # browser. Starlette already cancels this generator when the client
    # goes away, so there is nothing to poll for.
    loop = asyncio.get_running_loop()
    event_queue: asyncio.Queue = asyncio.Queue()

    def emit(item) -> None:
        """
        Called from the graph worker thread.

        If the client disconnected, this generator is gone and the loop
        may already be closed, while the worker keeps running to
        completion. That is expected — swallow it rather than filling
        the logs with tracebacks nobody can act on.
        """

        try:
            loop.call_soon_threadsafe(event_queue.put_nowait, item)
        except RuntimeError:
            pass

    yield _sse(
        {
            "type": "stream_started",
            "status": "STARTED",
            "latitude": payload.latitude,
            "longitude": payload.longitude,
            "question": payload.question,
            "nodes": list(ALL_NODES),
        }
    )

    _log(
        f"run started rss={_rss_mb()}MB "
        f"lat={payload.latitude} lon={payload.longitude}"
    )

    worker = Thread(
        target=_run_graph_worker,
        args=(initial_state, emit),
        daemon=True,
    )
    worker.start()

    try:
        while True:
            try:
                event_type, data = await asyncio.wait_for(
                    event_queue.get(),
                    timeout=HEARTBEAT_INTERVAL_S,
                )

            except asyncio.TimeoutError:
                elapsed = time.monotonic() - started

                if elapsed > GRAPH_TIMEOUT_S:
                    _log(f"timeout after {elapsed:.1f}s, completed={completed}")

                    yield _sse(
                        {
                            "type": "error",
                            "status": "ERROR",
                            "detail": (
                                f"The assessment exceeded {int(GRAPH_TIMEOUT_S)}s "
                                "and was stopped."
                            ),
                            "completed_nodes": list(completed),
                            "duration_seconds": round(elapsed, 2),
                        }
                    )

                    yield _sse(
                        _final_event(
                            payload,
                            final_state,
                            completed,
                            elapsed,
                            status="ERROR",
                            error="timeout",
                        )
                    )

                    return

                # RSS rides along on every heartbeat. If the process is
                # being killed for exceeding the instance's memory, the
                # last heartbeat the browser received is the last
                # reading before death — which is the only evidence a
                # SIGKILL leaves anywhere.
                rss = _rss_mb()

                _log(
                    f"heartbeat {elapsed:.0f}s rss={rss}MB "
                    f"completed={completed or '[]'}"
                )

                yield _sse(
                    {
                        "type": "heartbeat",
                        "status": "RUNNING",
                        "elapsed_seconds": round(elapsed, 1),
                        "memory_rss_mb": rss,
                        "completed_nodes": list(completed),
                    }
                )

                continue

            # ------------------------------------------------
            # A node (or several, when they ran in parallel) finished
            # ------------------------------------------------

            if event_type == "graph_update":
                if not isinstance(data, dict):
                    continue

                for node_name, node_update in data.items():
                    node_name = str(node_name)

                    if node_name not in completed:
                        completed.append(node_name)

                    if isinstance(node_update, dict):
                        final_state.update(node_update)

                    event = _node_event(
                        node_name,
                        final_state,
                        completed,
                        time.monotonic() - started,
                    )

                    _log(f"node_done {node_name} -> {event.get('node_status')}")

                    yield _sse(event)

                continue

            # ------------------------------------------------
            # Graph finished cleanly
            # ------------------------------------------------

            if event_type == "graph_finished":
                duration = time.monotonic() - started
                _log(f"finished in {duration:.1f}s, nodes={completed}")

                yield _sse(
                    _final_event(payload, final_state, completed, duration)
                )

                return

            # ------------------------------------------------
            # Graph raised
            # ------------------------------------------------

            if event_type == "graph_error":
                duration = time.monotonic() - started
                _log(f"graph error after {duration:.1f}s: {data!r}")

                yield _sse(
                    {
                        "type": "error",
                        "status": "ERROR",
                        "detail": str(data) or repr(data),
                        "completed_nodes": list(completed),
                        "duration_seconds": round(duration, 2),
                    }
                )

                # The frontend treats `final` as the terminal event, so
                # send one even on failure.
                yield _sse(
                    _final_event(
                        payload,
                        final_state,
                        completed,
                        duration,
                        status="ERROR",
                        error=str(data) or repr(data),
                    )
                )

                return

    except asyncio.CancelledError:
        # Client went away. Nothing to report to anyone.
        _log("client disconnected mid-stream")
        raise

    except Exception as exc:
        # Anything unexpected in this generator would otherwise close
        # the connection with no terminal event, which the browser can
        # only report as "the stream ended without a final result".
        traceback.print_exc()
        duration = time.monotonic() - started

        yield _sse(
            {
                "type": "error",
                "status": "ERROR",
                "detail": f"stream failure: {exc}",
                "completed_nodes": list(completed),
                "duration_seconds": round(duration, 2),
            }
        )

        yield _sse(
            _final_event(
                payload,
                final_state,
                completed,
                duration,
                status="ERROR",
                error=f"stream failure: {exc}",
            )
        )


@app.post("/api/ask/stream")
async def ask_stream(payload: AskRequest):
    """Streaming ORCAWA assessment."""

    return StreamingResponse(
        _stream_ask_events(payload),
        media_type="text/event-stream",
        headers={
            # Never cache an SSE stream.
            "Cache-Control": "no-cache, no-store, must-revalidate",
            # Tell nginx-style proxies (Render's included) not to buffer.
            "X-Accel-Buffering": "no",
            "X-Content-Type-Options": "nosniff",
        },
    )


# ============================================================
# LOCAL RUN
# ============================================================

if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=bool(os.getenv("RELOAD")),
    )
