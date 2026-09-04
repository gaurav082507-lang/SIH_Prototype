import streamlit as st
import pandas as pd
import json
from datetime import datetime

from graph import marine_graph


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ORCA | Marine Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONSTANTS
# ============================================================

LOCATION_PRESETS = {
    "Custom": None,
    "Mumbai, Maharashtra": (19.0760, 72.8777),
    "Chennai, Tamil Nadu": (13.0827, 80.2707),
    "Kochi, Kerala": (9.9312, 76.2673),
    "Visakhapatnam, Andhra Pradesh": (17.6868, 83.2185),
    "Goa (Panaji)": (15.4909, 73.8278),
    "Kolkata, West Bengal": (22.5726, 88.3639),
    "Kanyakumari, Tamil Nadu": (8.0883, 77.5385),
}

RISK_COLORS = {
    "LOW": "🟢",
    "MODERATE": "🟡",
    "HIGH": "🟠",
    "SEVERE": "🔴",
}

SUGGESTED_QUESTIONS = [
    "Where is the nearest Potential Fishing Zone today?",
    "Is it safe to venture into the sea tomorrow morning?",
    "What are the tide, weather, and sea conditions near my location?",
    "Are there any cyclone or lightning alerts nearby?",
]


# ============================================================
# SIMPLE CSS
# ============================================================
# This version intentionally does NOT use HTML inside st.markdown().
# The previous UI was showing literal <div>, <br>, etc. because the
# browser/Streamlit path was rendering the HTML as text.
# Native Streamlit components are used for all visible UI below.

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 10% 10%, rgba(0, 119, 182, 0.16), transparent 30%),
            radial-gradient(circle at 90% 20%, rgba(0, 180, 216, 0.10), transparent 28%),
            #050b16;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    section[data-testid="stSidebar"] {
        background: #071426;
        border-right: 1px solid rgba(0, 180, 216, 0.18);
    }

    section[data-testid="stSidebar"] * {
        color: #dcecff;
    }

    div[data-testid="stMetric"] {
        background: rgba(10, 28, 48, 0.72);
        padding: 18px;
        border-radius: 16px;
        border: 1px solid rgba(120, 190, 230, 0.14);
    }

    div[data-testid="stMetricLabel"] {
        color: #8da9c4 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #eaf7ff !important;
    }

    div[data-testid="stChatMessage"] {
        background: rgba(9, 24, 42, 0.55);
        border: 1px solid rgba(100, 180, 220, 0.10);
        border-radius: 16px;
    }

    .orca-result {
        padding: 22px;
        border-radius: 18px;
        background: rgba(7, 30, 50, 0.78);
        border: 1px solid rgba(0, 180, 216, 0.25);
        margin: 10px 0 18px 0;
    }

    .orca-risk {
        padding: 20px;
        border-radius: 18px;
        background: rgba(10, 28, 48, 0.78);
        border: 1px solid rgba(120, 190, 230, 0.16);
        margin: 10px 0 18px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HERO
# ============================================================

st.title("🌊 ORCA Marine Intelligence")
st.caption(
    "Ask in plain language — ORCA analyzes marine conditions and provides "
    "an evidence-based final assessment."
)

st.divider()


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "latitude" not in st.session_state:
    st.session_state.latitude = 19.076000

if "longitude" not in st.session_state:
    st.session_state.longitude = 72.877700

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


def _apply_preset():
    preset = LOCATION_PRESETS.get(st.session_state.location_preset)
    if preset:
        st.session_state.latitude, st.session_state.longitude = preset


def _queue_question(text):
    st.session_state.pending_question = text


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("🌊 ORCA")
    st.caption("Marine Intelligence Platform")
    st.divider()

    st.subheader("📍 Target Location")

    st.selectbox(
        "Quick Location",
        list(LOCATION_PRESETS.keys()),
        key="location_preset",
        on_change=_apply_preset,
    )

    latitude = st.number_input(
        "Latitude",
        min_value=-90.0,
        max_value=90.0,
        format="%.6f",
        key="latitude",
    )

    longitude = st.number_input(
        "Longitude",
        min_value=-180.0,
        max_value=180.0,
        format="%.6f",
        key="longitude",
    )

    st.map(
        pd.DataFrame({"lat": [latitude], "lon": [longitude]}),
        zoom=5,
        height=180,
    )

    st.divider()

    st.caption(
        "The Planner selects the marine intelligence agents required for "
        "each question. Only the final assessment is displayed below."
    )

    if st.button(
        "🧹 Clear Chat",
        disabled=not st.session_state.messages,
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()


# ============================================================
# RESULT HELPERS
# ============================================================

def _parse_plan(state):
    plan = state.get("plan")

    if isinstance(plan, str):
        try:
            plan = json.loads(plan)
        except Exception:
            return None

    return plan if isinstance(plan, dict) else None


def extract_map_points(result):
    """Extract optional PFZ points for the map. Never raises."""
    points = []

    pfz = result.get("pfz_data")

    if isinstance(pfz, dict):
        for key in ("candidates", "zones", "results", "data"):
            value = pfz.get(key)

            if not isinstance(value, list):
                continue

            for item in value:
                if not isinstance(item, dict):
                    continue

                lat = item.get("latitude", item.get("lat"))
                lon = item.get(
                    "longitude",
                    item.get("lon", item.get("lng")),
                )

                try:
                    if lat is not None and lon is not None:
                        points.append((float(lat), float(lon)))
                except (TypeError, ValueError):
                    continue

    return points


def render_final_result(result, latitude, longitude):
    """
    Display ONLY the final marine assessment and risk level.

    Planner JSON, active agents, raw agent findings, raw LangGraph output,
    and technical internals are intentionally hidden from the UI.
    """

    recommendation = result.get("recommendation")

    if isinstance(recommendation, str):
        try:
            recommendation = json.loads(recommendation)
        except Exception:
            recommendation = {
                "summary": recommendation,
                "risk_level": "MODERATE",
                "recommendation": recommendation,
            }

    if not isinstance(recommendation, dict):
        recommendation = {
            "summary": "No final recommendation was returned.",
            "risk_level": "MODERATE",
            "recommendation": "Please review the marine data and try again.",
        }

    risk_level = str(
        recommendation.get("risk_level", "MODERATE")
    ).upper()

    risk_icon = RISK_COLORS.get(risk_level, "⚪")

    summary = recommendation.get(
        "summary",
        "Marine assessment completed.",
    )

    recommendation_text = recommendation.get(
        "recommendation",
        "Please review the available marine conditions before proceeding.",
    )

    key_findings = recommendation.get("key_findings") or []
    safety_advice = recommendation.get("safety_advice") or []

    # --------------------------------------------------------
    # Optional map
    # --------------------------------------------------------

    map_points = [(latitude, longitude)] + extract_map_points(result)

    if len(map_points) > 1:
        st.map(
            pd.DataFrame(
                map_points,
                columns=["lat", "lon"],
            ),
            height=220,
        )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    st.subheader("🎯 Marine Intelligence Result")

    risk_col, location_col = st.columns([2, 1])

    with risk_col:
        st.metric(
            "Final Risk Level",
            f"{risk_icon} {risk_level}",
        )

    with location_col:
        st.metric(
            "Location",
            f"{latitude:.3f}, {longitude:.3f}",
        )

    st.markdown("### 🌊 Final Assessment")
    st.info(summary)

    st.markdown("### 🚢 Recommendation")
    st.success(recommendation_text)

    # --------------------------------------------------------
    # Findings
    # --------------------------------------------------------

    if isinstance(key_findings, list) and key_findings:
        st.markdown("### ⚠️ Key Findings")

        for finding in key_findings:
            st.markdown(f"- {finding}")

    # --------------------------------------------------------
    # Safety advice
    # --------------------------------------------------------

    if isinstance(safety_advice, list) and safety_advice:
        st.markdown("### 🧭 Safety Advice")

        for advice in safety_advice:
            st.warning(advice)

    # --------------------------------------------------------
    # Error handling
    # --------------------------------------------------------

    # The UI no longer exposes technical error details.
    # If the recommendation node failed, show a concise user-facing notice.
    if recommendation.get("error"):
        st.warning(
            "The final recommendation could not be fully generated. "
            "The displayed result is a fallback assessment."
        )


# ============================================================
# GRAPH EXECUTION
# ============================================================

def run_marine_graph(initial_state):
    """
    Invoke the graph.

    Stream updates when possible, while retaining the latest complete state.
    """

    final_state = dict(initial_state)
    streamed = False

    try:
        for step_output in marine_graph.stream(
            initial_state,
            stream_mode="updates",
        ):
            streamed = True

            if not isinstance(step_output, dict):
                continue

            for _, node_update in step_output.items():
                if isinstance(node_update, dict):
                    final_state.update(node_update)

            yield final_state

        if streamed:
            return

    except Exception:
        pass

    final_state = marine_graph.invoke(initial_state)
    yield final_state


# ============================================================
# PREVIOUS CONVERSATION
# ============================================================

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message("user"):
        st.markdown(msg["question"])

    with st.chat_message("assistant", avatar="🌊"):
        if msg.get("error"):
            st.error("Marine analysis failed.")
        elif msg.get("result"):
            render_final_result(
                msg["result"],
                msg["latitude"],
                msg["longitude"],
            )


# ============================================================
# SUGGESTED QUESTIONS
# ============================================================

if not st.session_state.messages:
    st.subheader("💡 Try asking")

    cols = st.columns(len(SUGGESTED_QUESTIONS))

    for col, question in zip(cols, SUGGESTED_QUESTIONS):
        with col:
            if st.button(
                question,
                key=f"suggested_{question}",
                use_container_width=True,
            ):
                _queue_question(question)


# ============================================================
# CHAT INPUT
# ============================================================

prompt = st.chat_input(
    "Ask about marine conditions, e.g. "
    "'Is it safe to fish near Kochi tomorrow?'"
)

if not prompt and st.session_state.pending_question:
    prompt = st.session_state.pending_question
    st.session_state.pending_question = None


if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    initial_state = {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "user_question": prompt.strip(),
        "status": "STARTED",
    }

    with st.chat_message("assistant", avatar="🌊"):
        result = None
        error = None

        with st.status(
            "🤖 Running Marine Intelligence Agents...",
            expanded=False,
        ) as status:
            try:
                for state in run_marine_graph(initial_state):
                    result = state

                status.update(
                    label="Marine intelligence analysis completed",
                    state="complete",
                    expanded=False,
                )

            except Exception as exc:
                error = exc

                status.update(
                    label="Marine analysis failed",
                    state="error",
                    expanded=True,
                )

        if error:
            st.error("❌ Marine analysis failed.")
        elif result:
            render_final_result(
                result,
                latitude,
                longitude,
            )

    st.session_state.messages.append(
        {
            "question": prompt,
            "latitude": latitude,
            "longitude": longitude,
            "result": result,
            "error": error,
            "timestamp": datetime.now().isoformat(
                timespec="seconds"
            ),
        }
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "🌊 ORCA Marine Intelligence Platform • "
    "Agentic AI • LangGraph • Marine Data Intelligence"
)
