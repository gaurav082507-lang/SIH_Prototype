# server.py
#
# Meridian — Marine Intelligence API
#
# Thin FastAPI wrapper around the existing LangGraph pipeline (graph.py).
# It does two jobs:
#   1. Serves the static frontend (static/index.html) at "/".
#   2. Exposes POST /api/ask, which runs marine_graph on a
#      {latitude, longitude, question} payload and returns the final
#      recommendation plus a per-node status summary the frontend uses
#      to render the pipeline strip.
#
# Run locally:
#   uvicorn server:app --reload --port 8000
#
# On Render, this is started via:
#   uvicorn server:app --host 0.0.0.0 --port $PORT
# (see render.yaml)

import json
import os
import time
import traceback
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from graph import marine_graph

# ------------------------------------------------------------------
# App setup
# ------------------------------------------------------------------

app = FastAPI(title="Meridian Marine Intelligence API", version="1.0.0")

# Allow the frontend to be hosted separately if needed (e.g. during
# local dev the static file is opened outside the API's own origin).
# In production both are served from the same Render service, so this
# is mostly a safety net.
_allowed_origins = os.getenv("ALLOWED_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _allowed_origins == "*" else _allowed_origins.split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

PIPELINE_NODES = [
    "planner", "gis", "weather", "ocean", "tide",
    "cyclone", "ecosystem", "pfz", "recommendation",
]


# ------------------------------------------------------------------
# Request / response models
# ------------------------------------------------------------------

class AskRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    question: str = Field(..., min_length=1, max_length=2000)


class AskResponse(BaseModel):
    status: str
    latitude: float
    longitude: float
    question: str
    plan: Optional[dict] = None
    completed_nodes: list[str] = []
    recommendation: Optional[dict] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _parse_plan(state: dict) -> dict:
    plan = state.get("plan", {})
    if isinstance(plan, str):
        try:
            plan = json.loads(plan)
        except Exception:
            return {}
    return plan if isinstance(plan, dict) else {}


def _normalize_recommendation(state: dict) -> dict:
    rec = state.get("recommendation")
    if isinstance(rec, str):
        try:
            rec = json.loads(rec)
        except Exception:
            rec = {"summary": rec, "risk_level": "UNKNOWN", "recommendation": ""}
    if not isinstance(rec, dict):
        rec = {
            "summary": "No final assessment was returned.",
            "risk_level": "UNKNOWN",
            "recommendation": "",
        }
    return rec


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "service": "meridian-marine-intelligence"}


@app.post("/api/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
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
        for step_output in marine_graph.stream(initial_state, stream_mode="updates"):
            if not isinstance(step_output, dict):
                continue
            for node_name, node_update in step_output.items():
                completed_nodes.append(str(node_name))
                if isinstance(node_update, dict):
                    final_state.update(node_update)

        duration = round(time.monotonic() - started, 2)
        plan = _parse_plan(final_state)
        recommendation = _normalize_recommendation(final_state)

        return AskResponse(
            status="SUCCESS",
            latitude=payload.latitude,
            longitude=payload.longitude,
            question=payload.question,
            plan=plan,
            completed_nodes=completed_nodes,
            recommendation=recommendation,
            duration_seconds=duration,
        )

    except Exception as exc:  # noqa: BLE001 — surface a clean error to the UI
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ------------------------------------------------------------------
# Static frontend
# ------------------------------------------------------------------

app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
