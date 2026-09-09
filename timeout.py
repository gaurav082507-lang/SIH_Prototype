# node_timeout.py
#
# Per-node deadlines for the ORCAWA LangGraph pipeline.
#
# Without this, one unresponsive marine service stalls the whole
# assessment: the graph has no deadline of its own, so a specialist
# waiting on a dead upstream holds up the final recommendation for as
# long as its HTTP client allows.
#
# Wrapping each node means a slow agent degrades to "no data" for that
# agent while every other branch still reaches the recommendation node.
#
# Defaults (override per deployment with env vars):
#
#     planner            90s   PLANNER_TIMEOUT_S
#     recommendation     90s   RECOMMENDATION_TIMEOUT_S
#     every specialist   60s   AGENT_TIMEOUT_S
#
# A single agent can be given its own budget with AGENT_TIMEOUT_<NAME>,
# e.g. AGENT_TIMEOUT_TIDE=120 for a tide service known to be slow.
#
# HONEST CAVEAT: Python cannot kill a running thread. A node that
# blows its deadline is abandoned, not stopped — the graph moves on,
# but that work keeps running in the background until it finishes on
# its own, still holding its memory and its socket. Timeouts protect
# the user's wait, not the instance's resources.

from __future__ import annotations

import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import date as _date
from typing import Any, Callable


# ============================================================
# CONFIG
# ============================================================


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)

    if raw is None or not raw.strip():
        return default

    try:
        value = float(raw)
    except ValueError:
        print(f"[timeout] ignoring non-numeric {name}={raw!r}", flush=True)
        return default

    return value if value > 0 else default


DEFAULT_AGENT_TIMEOUT_S = _env_float("AGENT_TIMEOUT_S", 60.0)
PLANNER_TIMEOUT_S = _env_float("PLANNER_TIMEOUT_S", 90.0)
RECOMMENDATION_TIMEOUT_S = _env_float("RECOMMENDATION_TIMEOUT_S", 90.0)

# Specialists run in parallel, so the pool has to be at least as wide
# as the widest fan-out (7 specialists) plus the planner and the
# recommendation node.
_POOL_SIZE = int(_env_float("NODE_POOL_SIZE", 10.0))

_EXECUTOR = ThreadPoolExecutor(
    max_workers=_POOL_SIZE,
    thread_name_prefix="orcawa-node",
)


def timeout_for(node: str) -> float:
    """Seconds this node is allowed before it is abandoned."""

    override = _env_float(f"AGENT_TIMEOUT_{node.upper()}", 0.0)

    if override:
        return override

    if node == "planner":
        return PLANNER_TIMEOUT_S

    if node == "recommendation":
        return RECOMMENDATION_TIMEOUT_S

    return DEFAULT_AGENT_TIMEOUT_S


def all_timeouts(nodes) -> dict[str, float]:
    """Every node's budget, for /api/health."""

    return {str(node): timeout_for(str(node)) for node in nodes}


# ============================================================
# FALLBACK STATE
# ============================================================


def timeout_payload(node: str, state_key: str, seconds: float) -> dict[str, Any]:
    """
    The state update a node produces when it misses its deadline.

    Shaped exactly like the node's own failure envelope, so nothing
    downstream — recommendation_node, server.py, the UI — needs to
    know a timeout is different from any other agent failure.
    """

    message = f"{node} did not respond within {seconds:g}s."

    if node == "planner":
        # A timed-out planner has selected no agents, so the graph
        # routes straight to the recommendation node. `timed_out` lets
        # the UI say "timed out" rather than "out of scope".
        return {
            "plan": {
                "rejected": True,
                "timed_out": True,
                "rejection_reason": message,
                "required_agents": [],
                "activity": "unknown",
                "date": _date.today().isoformat(),
                "grid_points": [],
            },
            "status": "TIMEOUT",
        }

    if node == "recommendation":
        return {
            "recommendation": {
                "summary": message,
                "risk_level": "UNKNOWN",
                "recommendation": (
                    "The assessment could not be completed in time. "
                    "Run it again."
                ),
                "key_findings": [],
                "safety_advice": [],
                "timed_out": True,
                "error": "timeout",
            },
            "status": "TIMEOUT",
        }

    return {
        state_key: {
            "status": "FAILED",
            "timed_out": True,
            "error": message,
            "errors": [message],
        }
    }


# ============================================================
# WRAPPER
# ============================================================


def with_timeout(
    node: str,
    fn: Callable[[dict], Any],
    state_key: str | None = None,
) -> Callable[[dict], Any]:
    """
    Wrap a LangGraph node so it cannot outlive its deadline.

    Exceptions are deliberately NOT swallowed — a node that raises
    still propagates, so a real bug surfaces as an error instead of
    being disguised as an agent with no data. Only the deadline is
    handled here.
    """

    key = state_key or f"{node}_data"

    def wrapped(state: dict) -> Any:
        seconds = timeout_for(node)  # re-read so env changes apply
        started = time.monotonic()

        future = _EXECUTOR.submit(fn, state)

        try:
            result = future.result(timeout=seconds)

        except FutureTimeoutError:
            # Cannot actually stop it; it is abandoned, not killed.
            future.cancel()

            print(
                f"[timeout] {node} exceeded {seconds:g}s — continuing "
                "without it",
                flush=True,
            )

            return timeout_payload(node, key, seconds)

        except Exception:
            print(
                f"[node] {node} raised after "
                f"{time.monotonic() - started:.1f}s",
                flush=True,
            )
            traceback.print_exc()
            raise

        print(
            f"[node] {node} finished in {time.monotonic() - started:.1f}s "
            f"(budget {seconds:g}s)",
            flush=True,
        )

        return result

    wrapped.__name__ = f"{node}_with_timeout"
    wrapped.__doc__ = f"{node} node, abandoned after {timeout_for(node):.0f}s."

    return wrapped
