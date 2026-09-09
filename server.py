if node_name == FINAL_NODE:
    duration = time.monotonic() - started

    _log(
        f"final recommendation received in "
        f"{duration:.1f}s, nodes={completed}"
    )

    yield _sse(
        _final_event(
            payload,
            final_state,
            completed,
            duration,
            status="SUCCESS",
        )
    )

    return
