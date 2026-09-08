from fastapi import FastAPI
from fastapi.responses import StreamingResponse

# other imports...

app = FastAPI()

# AskRequest definition
# marine_graph import
# helper functions


@app.post("/api/ask")
def ask(payload: AskRequest):
    # your existing code
    ...


# Streaming helper
def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


def _stream_ask_events(payload: AskRequest):
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
        for step_output in marine_graph.stream(
            initial_state,
            stream_mode="updates"
        ):
            if not isinstance(step_output, dict):
                continue

            for node_name, node_update in step_output.items():
                node_name = str(node_name)
                completed_nodes.append(node_name)

                if isinstance(node_update, dict):
                    final_state.update(node_update)

                event: dict[str, Any] = {
                    "type": "node_done",
                    "node": node_name,
                }

                if node_name == "planner":
                    event["plan"] = _parse_plan(final_state)

                yield _sse(event)

        duration = round(time.monotonic() - started, 2)

        yield _sse({
            "type": "final",
            "status": "SUCCESS",
            "latitude": payload.latitude,
            "longitude": payload.longitude,
            "question": payload.question,
            "plan": _parse_plan(final_state),
            "completed_nodes": completed_nodes,
            "recommendation": _normalize_recommendation(final_state),
            "duration_seconds": duration,
        })

    except Exception as exc:
        traceback.print_exc()
        yield _sse({
            "type": "error",
            "detail": str(exc),
        })


@app.post("/api/ask/stream")
def ask_stream(payload: AskRequest):
    return StreamingResponse(
        _stream_ask_events(payload),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
        },
    )
