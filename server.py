# --- Add this import near the top of server.py, alongside your other fastapi.responses imports ---
from fastapi.responses import StreamingResponse

# --- Add this helper + route, right after your existing "@app.post("/api/ask")" route ---

def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


def _stream_ask_events(payload: AskRequest):
    """
    Runs marine_graph.stream() and yields a Server-Sent Event the moment
    each node finishes, instead of buffering everything until the end.
    This lets the frontend show real-time, per-agent progress that
    matches what route_after_planner() actually decided to run.
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
        for step_output in marine_graph.stream(initial_state, stream_mode="updates"):
            if not isinstance(step_output, dict):
                continue
            for node_name, node_update in step_output.items():
                node_name = str(node_name)
                completed_nodes.append(node_name)
                if isinstance(node_update, dict):
                    final_state.update(node_update)

                event: dict[str, Any] = {"type": "node_done", "node": node_name}
                # The planner's output is the only thing that tells the
                # frontend which agents are required — send it as soon
                # as it's known so the UI can grey out skipped agents
                # immediately instead of pretending they're running.
                if node_name == "planner":
                    event["plan"] = _parse_plan(final_state)
                yield _sse(event)

        duration = round(time.monotonic() - started, 2)
        plan = _parse_plan(final_state)
        recommendation = _normalize_recommendation(final_state)
        yield _sse({
            "type": "final",
            "status": "SUCCESS",
            "latitude": payload.latitude,
            "longitude": payload.longitude,
            "question": payload.question,
            "plan": plan,
            "completed_nodes": completed_nodes,
            "recommendation": recommendation,
            "duration_seconds": duration,
        })
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        yield _sse({"type": "error", "detail": str(exc)})


@app.post("/api/ask/stream")
def ask_stream(payload: AskRequest):
    return StreamingResponse(
        _stream_ask_events(payload),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )
