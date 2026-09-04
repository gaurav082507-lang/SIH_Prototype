import streamlit as st
import pandas as pd
import json
import textwrap
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

AVAILABLE_AGENTS = [
    ("🌤️", "Weather", "weather"),
    ("🌊", "Ocean", "ocean"),
    ("🌙", "Tide", "tide"),
    ("🌀", "Cyclone", "cyclone"),
    ("🐟", "Ecosystem", "ecosystem"),
    ("🎣", "PFZ", "pfz"),
    ("🗺️", "GIS", "gis"),
]

# GIS is always executed for accepted queries in graph.py.
ALWAYS_ON_AGENTS = {"gis"}

RISK_COLORS = {
    "LOW": "🟢",
    "MODERATE": "🟡",
    "HIGH": "🟠",
    "SEVERE": "🔴",
    "UNKNOWN": "⚪",
}

# This is the visual pipeline shown to the user.
PIPELINE_DEFS = [
    ("planner", "🧠", "Planner"),
    ("gis", "🗺️", "GIS"),
    ("weather", "🌤️", "Weather"),
    ("ocean", "🌊", "Ocean"),
    ("tide", "🌙", "Tide"),
    ("cyclone", "🌀", "Cyclone"),
    ("ecosystem", "🐟", "Ecosystem"),
    ("pfz", "🎣", "PFZ"),
    ("recommendation", "🎯", "Final Assessment"),
]

STAGE_LABELS = {
    "pending": "Waiting",
    "active": "Running…",
    "done": "Completed",
    "skipped": "Not required",
}

SUGGESTED_QUESTIONS = [
    "Where is the nearest Potential Fishing Zone today?",
    "Is it safe to venture into the sea tomorrow morning?",
    "What are the tide, weather, and sea conditions near my location?",
    "Are there any cyclone or lightning alerts nearby?",
]


# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown(
    textwrap.dedent(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 10% 10%, rgba(0,119,182,.18), transparent 30%),
                radial-gradient(circle at 90% 20%, rgba(0,180,216,.12), transparent 28%),
                radial-gradient(circle at 50% 100%, rgba(3,64,120,.20), transparent 35%),
                #050b16;
            color: #e8f1ff;
        }

        .block-container {
            max-width: 1300px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(
                180deg,
                #071426 0%,
                #06101f 50%,
                #040b16 100%
            );
            border-right: 1px solid rgba(0,180,216,.18);
        }

        section[data-testid="stSidebar"] * {
            color: #dcecff;
        }

        .hero {
            padding: 30px 34px;
            border-radius: 24px;
            background:
                linear-gradient(
                    135deg,
                    rgba(0,119,182,.22),
                    rgba(0,180,216,.08),
                    rgba(4,20,40,.70)
                );
            border: 1px solid rgba(0,180,216,.20);
            box-shadow: 0 20px 60px rgba(0,0,0,.35);
            margin-bottom: 22px;
        }

        .hero-title {
            font-size: 40px;
            font-weight: 800;
            letter-spacing: -1px;
            background: linear-gradient(90deg,#ffffff,#8be9ff,#4cc9f0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero-subtitle {
            color: #9bb4d0;
            font-size: 15px;
            margin-top: 6px;
        }

        .status-pill {
            display: inline-block;
            padding: 7px 14px;
            border-radius: 999px;
            background: rgba(0,180,216,.12);
            border: 1px solid rgba(0,180,216,.25);
            color: #6ee7ff;
            font-size: 13px;
            font-weight: 600;
        }

        .section-title {
            font-size: 20px;
            font-weight: 700;
            color: #edf7ff;
            margin-top: 18px;
            margin-bottom: 12px;
        }

        .pipeline-shell {
            padding: 18px;
            border-radius: 18px;
            background: linear-gradient(
                135deg,
                rgba(15,35,60,.72),
                rgba(7,18,32,.72)
            );
            border: 1px solid rgba(120,190,230,.13);
            box-shadow: 0 15px 45px rgba(0,0,0,.25);
            margin-bottom: 18px;
        }

        .pipeline-caption {
            color: #82a6c9;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 12px;
        }

        .pipe-scroll {
            overflow-x: auto;
            padding: 4px 2px 10px 2px;
        }

        .pipe-row {
            display: flex;
            align-items: center;
            flex-wrap: nowrap;
            min-width: max-content;
        }

        .pipe-node {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-width: 110px;
            min-height: 82px;
            padding: 10px 8px;
            border-radius: 14px;
            text-align: center;
            border: 1.5px solid rgba(120,190,230,.18);
            background: rgba(10,28,48,.55);
            transition: all .35s ease;
        }

        .pipe-icon {
            font-size: 21px;
        }

        .pipe-name {
            font-size: 11.5px;
            font-weight: 700;
            margin-top: 4px;
            color: #dcecff;
        }

        .pipe-state {
            font-size: 10px;
            margin-top: 3px;
            letter-spacing: .3px;
        }

        .pipe-pending {
            opacity: .40;
        }

        .pipe-pending .pipe-state,
        .pipe-skipped .pipe-state {
            color: #7f97b3;
        }

        .pipe-skipped {
            opacity: .28;
            border-style: dashed;
        }

        .pipe-active {
            border-color: #00b4d8;
            opacity: 1;
            box-shadow: 0 0 0 0 rgba(0,180,216,.45);
            animation: pipe-pulse 1.3s ease-in-out infinite;
        }

        .pipe-active .pipe-state {
            color: #6ee7ff;
            font-weight: 700;
        }

        .pipe-done {
            border-color: #34d399;
            opacity: 1;
            background: rgba(16,60,50,.55);
        }

        .pipe-done .pipe-state {
            color: #5eead4;
            font-weight: 700;
        }

        .pipe-arrow {
            flex: 0 0 auto;
            padding: 0 7px;
            font-size: 15px;
            color: rgba(140,190,220,.30);
        }

        .pipe-arrow-done {
            color: #34d399;
        }

        .execution-text {
            margin-top: 8px;
            color: #9bb4d0;
            font-size: 13px;
        }

        .recommendation-card {
            padding: 24px;
            border-radius: 20px;
            background: linear-gradient(
                135deg,
                rgba(0,119,182,.18),
                rgba(6,24,43,.82)
            );
            border: 1px solid rgba(0,180,216,.25);
            box-shadow: 0 20px 55px rgba(0,0,0,.28);
            margin-top: 10px;
        }

        .card-title {
            font-size: 13px;
            color: #82a6c9;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }

        .recommendation-text {
            font-size: 18px;
            line-height: 1.7;
            color: #dcecff;
        }

        .final-recommendation {
            padding: 18px 20px;
            border-radius: 15px;
            background: rgba(16,60,50,.55);
            border: 1px solid rgba(52,211,153,.25);
            color: #dffcf3;
            font-size: 17px;
            line-height: 1.6;
        }

        .error-card {
            padding: 14px 16px;
            border-radius: 12px;
            background: rgba(127,29,29,.30);
            border: 1px solid rgba(248,113,113,.25);
            color: #fecaca;
        }

        .agent-card {
            padding: 16px;
            border-radius: 16px;
            text-align: center;
            background: linear-gradient(
                145deg,
                rgba(12,35,60,.75),
                rgba(6,17,31,.8)
            );
            border: 1px solid rgba(0,180,216,.12);
            min-height: 100px;
        }

        .agent-icon {
            font-size: 24px;
        }

        .agent-name {
            font-size: 14px;
            font-weight: 700;
            margin-top: 6px;
        }

        .agent-active {
            color: #5eead4;
            font-size: 12px;
            margin-top: 4px;
        }

        .agent-inactive {
            color: #64748b;
            font-size: 12px;
            margin-top: 4px;
        }

        .chip-btn button {
            background: rgba(0,180,216,.10) !important;
            border: 1px solid rgba(0,180,216,.30) !important;
            color: #bfe9ff !important;
            font-weight: 500 !important;
            border-radius: 999px !important;
            box-shadow: none !important;
        }

        textarea,
        input {
            color: #eaf6ff !important;
        }

        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"],
        div[data-baseweb="select"] > div {
            background-color: rgba(8,22,38,.85) !important;
            border: 1px solid rgba(80,170,210,.20) !important;
            border-radius: 12px !important;
        }

        div.stButton > button {
            width: 100%;
            border: none;
            border-radius: 12px;
            padding: 10px 18px;
            font-weight: 700;
            color: white;
            background: linear-gradient(90deg,#0077b6,#00b4d8);
            box-shadow: 0 8px 25px rgba(0,180,216,.20);
            transition: .2s ease;
        }

        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 32px rgba(0,180,216,.32);
        }

        div[data-testid="stMetric"] {
            background: rgba(10,28,48,.65);
            padding: 14px;
            border-radius: 15px;
            border: 1px solid rgba(120,190,230,.12);
        }

        div[data-testid="stMetricLabel"] {
            color: #8da9c4 !important;
        }

        div[data-testid="stMetricValue"] {
            color: #eaf7ff !important;
        }

        .footer {
            text-align: center;
            color: #526b84;
            font-size: 12px;
            margin-top: 40px;
            padding-top: 18px;
            border-top: 1px solid rgba(100,180,220,.08);
        }

        @keyframes pipe-pulse {
            0%   { box-shadow: 0 0 0 0 rgba(0,180,216,.45); }
            70%  { box-shadow: 0 0 0 9px rgba(0,180,216,0); }
            100% { box-shadow: 0 0 0 0 rgba(0,180,216,0); }
        }
        </style>
        """
    ),
    unsafe_allow_html=True,
)


# ============================================================
# HERO
# ============================================================
st.markdown(
    textwrap.dedent(
        """
        <div class="hero">
            <div class="status-pill">● AI MARINE INTELLIGENCE SYSTEM</div>
            <div class="hero-title">ORCA Marine Intelligence</div>
            <div class="hero-subtitle">
                Ask in plain language — ORCA analyzes marine conditions
                and provides an evidence-based final assessment.
            </div>
        </div>
        """
    ),
    unsafe_allow_html=True,
)


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
    st.markdown("## 🌊 ORCA")
    st.caption("Marine Intelligence Platform")
    st.divider()

    st.markdown("### 📍 Target Location")

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
    st.markdown("### 🤖 Agent Architecture")
    st.caption(
        "The Planner dynamically decides which specialist agents are required."
    )

    for icon, name, _ in AVAILABLE_AGENTS:
        st.markdown(f"{icon} {name}")

    st.divider()

    if st.button(
        "🧹 Clear Chat",
        disabled=not st.session_state.messages,
        help="Clear the conversation history and start fresh.",
    ):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()


# ============================================================
# PLAN HELPERS
# ============================================================
def parse_plan(state):
    plan = state.get("plan")

    if isinstance(plan, str):
        try:
            plan = json.loads(plan)
        except Exception:
            return {}

    return plan if isinstance(plan, dict) else {}


def selected_agents_from_state(state):
    plan = parse_plan(state)

    if plan.get("rejected"):
        return set()

    required = plan.get("required_agents", [])

    if not isinstance(required, list):
        required = []

    selected = {str(x).lower() for x in required}
    selected.update(ALWAYS_ON_AGENTS)

    return selected


# ============================================================
# DYNAMIC PIPELINE
# ============================================================
def compute_pipeline_status(state, active_node=None, completed_nodes=None):
    """
    Build visual pipeline state.

    Important:
    - Uses the actual LangGraph node updates when available.
    - The node currently being processed can be explicitly highlighted.
    - Completed nodes stay green.
    - Planner-selected specialists are shown as waiting/running/completed.
    - Non-selected specialists are skipped.
    - GIS is always selected for accepted queries.
    """
    completed_nodes = set(completed_nodes or [])

    plan = parse_plan(state)
    plan_exists = bool(state.get("plan"))
    rejected = bool(plan.get("rejected"))

    selected = selected_agents_from_state(state)

    status = {}

    # Planner
    if "planner" in completed_nodes or plan_exists:
        status["planner"] = "done"
    elif active_node == "planner":
        status["planner"] = "active"
    else:
        status["planner"] = "pending"

    # Specialist agents
    for node_id, _, _ in PIPELINE_DEFS[1:-1]:
        if rejected:
            status[node_id] = "skipped"
            continue

        if node_id not in selected:
            status[node_id] = "skipped"
            continue

        if node_id in completed_nodes:
            status[node_id] = "done"
        elif active_node == node_id:
            status[node_id] = "active"
        else:
            status[node_id] = "pending"

    # Final assessment
    if rejected:
        status["recommendation"] = "skipped"
    elif "recommendation" in completed_nodes or state.get("recommendation"):
        status["recommendation"] = "done"
    elif active_node == "recommendation":
        status["recommendation"] = "active"
    else:
        status["recommendation"] = "pending"

    return status


def render_pipeline_html(stage_status):
    nodes = []

    for index, (node_id, icon, label) in enumerate(PIPELINE_DEFS):
        current = stage_status.get(node_id, "pending")

        nodes.append(
            f"""
            <div class="pipe-node pipe-{current}">
                <div class="pipe-icon">{icon}</div>
                <div class="pipe-name">{label}</div>
                <div class="pipe-state">{STAGE_LABELS[current]}</div>
            </div>
            """
        )

        if index < len(PIPELINE_DEFS) - 1:
            next_node = PIPELINE_DEFS[index + 1][0]
            arrow_done = (
                "pipe-arrow-done"
                if current == "done"
                else ""
            )

            nodes.append(
                f'<div class="pipe-arrow {arrow_done}">➜</div>'
            )

    return (
        '<div class="pipeline-shell">'
        '<div class="pipeline-caption">LIVE PIPELINE EXECUTION</div>'
        '<div class="pipe-scroll">'
        '<div class="pipe-row">'
        + "".join(nodes)
        + "</div></div></div>"
    )


def render_pipeline(pipeline_slot, state, active_node=None, completed_nodes=None):
    statuses = compute_pipeline_status(
        state,
        active_node=active_node,
        completed_nodes=completed_nodes,
    )

    pipeline_slot.markdown(
        render_pipeline_html(statuses),
        unsafe_allow_html=True,
    )


# ============================================================
# GRAPH EXECUTION
# ============================================================
def run_marine_graph(initial_state, pipeline_slot=None):
    """
    Stream LangGraph updates and continuously refresh the pipeline.

    The UI receives a state update after every completed LangGraph node.
    Between updates, the next selected node is shown as RUNNING.
    """

    final_state = dict(initial_state)
    completed_nodes = set()

    # Initial pipeline: planner is the first active stage.
    if pipeline_slot is not None:
        render_pipeline(
            pipeline_slot,
            final_state,
            active_node="planner",
            completed_nodes=completed_nodes,
        )

    yielded_any = False

    try:
        stream = marine_graph.stream(
            initial_state,
            stream_mode="updates",
        )

        for step_output in stream:
            yielded_any = True

            if not isinstance(step_output, dict):
                continue

            for node_name, node_update in step_output.items():
                node_name = str(node_name)
                completed_nodes.add(node_name)

                if isinstance(node_update, dict):
                    final_state.update(node_update)

                # Find the next stage that should run.
                plan = parse_plan(final_state)
                selected = selected_agents_from_state(final_state)
                rejected = bool(plan.get("rejected"))

                next_active = None

                if not rejected:
                    for candidate, _, _ in PIPELINE_DEFS:
                        if candidate == "planner":
                            if "planner" not in completed_nodes:
                                next_active = "planner"
                                break
                            continue

                        if candidate == "recommendation":
                            continue

                        if candidate in completed_nodes:
                            continue

                        if candidate in selected:
                            next_active = candidate
                            break

                    if next_active is None and "recommendation" not in completed_nodes:
                        next_active = "recommendation"

                render_pipeline(
                    pipeline_slot,
                    final_state,
                    active_node=next_active,
                    completed_nodes=completed_nodes,
                )

                yield node_name, dict(final_state)

        if yielded_any:
            return

    except Exception:
        # Do not silently hide the real graph error.
        # Re-raise so the Streamlit UI shows the actual backend failure.
        raise

    # Only use invoke() if stream() produced no usable updates.
    final_state = marine_graph.invoke(initial_state)

    completed_nodes.update(
        node_id for node_id, _, _ in PIPELINE_DEFS
        if node_id != "recommendation"
    )
    completed_nodes.add("recommendation")

    if pipeline_slot is not None:
        render_pipeline(
            pipeline_slot,
            final_state,
            active_node=None,
            completed_nodes=completed_nodes,
        )

    yield None, final_state


# ============================================================
# FINAL RESULT ONLY
# ============================================================
def render_final_result(result, latitude, longitude):
    """
    Display only the final assessment/recommendation.

    Raw specialist payloads, Planner JSON, Agent Findings and complete
    LangGraph output are intentionally hidden from the main UI.
    """

    recommendation = result.get("recommendation")

    if isinstance(recommendation, str):
        try:
            recommendation = json.loads(recommendation)
        except Exception:
            recommendation = {
                "summary": recommendation,
                "risk_level": "UNKNOWN",
                "recommendation": "",
            }

    if not isinstance(recommendation, dict):
        recommendation = {
            "summary": "No final recommendation was returned.",
            "risk_level": "UNKNOWN",
            "recommendation": "",
        }

    risk_level = str(
        recommendation.get("risk_level", "UNKNOWN")
    ).upper()

    risk_icon = RISK_COLORS.get(risk_level, "⚪")

    st.markdown(
        '<div class="section-title">🎯 Marine Intelligence Result</div>',
        unsafe_allow_html=True,
    )

    risk_col, location_col = st.columns([2, 1])

    with risk_col:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="card-title">Final Risk Level</div>
                <div style="font-size:42px;font-weight:700;color:#eaf7ff;">
                    {risk_icon} {risk_level}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with location_col:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="card-title">Location</div>
                <div style="font-size:28px;font-weight:700;color:#eaf7ff;">
                    {latitude:.3f}, {longitude:.3f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # Final Assessment
    # --------------------------------------------------------
    st.markdown(
        '<div class="section-title">🌊 Final Assessment</div>',
        unsafe_allow_html=True,
    )

    summary = recommendation.get(
        "summary",
        "No final assessment was returned.",
    )

    st.markdown(
        f"""
        <div class="recommendation-card">
            <div class="card-title">Marine Analysis</div>
            <div class="recommendation-text">
                {summary}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Recommendation
    # --------------------------------------------------------
    final_recommendation = recommendation.get(
        "recommendation",
        "",
    )

    st.markdown(
        '<div class="section-title">🚢 Recommendation</div>',
        unsafe_allow_html=True,
    )

    if final_recommendation:
        st.markdown(
            f"""
            <div class="final-recommendation">
                {final_recommendation}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning(
            "The recommendation agent did not return a recommendation."
        )

    # --------------------------------------------------------
    # Backend error — only show if one really exists.
    # --------------------------------------------------------
    error_message = recommendation.get("error")

    if error_message:
        st.markdown(
            f"""
            <div class="error-card">
                <strong>Recommendation generation error:</strong><br>
                {error_message}
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# CONVERSATION HISTORY
# ============================================================
if st.session_state.messages:
    top_cols = st.columns([6, 1])

    with top_cols[1]:
        if st.button(
            "🧹 Clear Chat",
            key="clear_chat_top",
            help="Clear the conversation history.",
        ):
            st.session_state.messages = []
            st.session_state.pending_question = None
            st.rerun()


for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message("user"):
        st.markdown(msg["question"])

    with st.chat_message("assistant", avatar="🌊"):
        if msg.get("error"):
            st.error("Marine analysis failed.")
            st.exception(msg["error"])
        else:
            # Historical messages show the completed pipeline.
            completed = {
                node_id
                for node_id, _, _ in PIPELINE_DEFS
            }

            pipeline_slot = st.empty()
            render_pipeline(
                pipeline_slot,
                msg["result"],
                active_node=None,
                completed_nodes=completed,
            )

            render_final_result(
                msg["result"],
                msg["latitude"],
                msg["longitude"],
            )


# ============================================================
# SUGGESTED QUESTIONS
# ============================================================
if not st.session_state.messages:
    st.markdown(
        '<div class="section-title">💡 Try asking</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(len(SUGGESTED_QUESTIONS))

    for col, question in zip(cols, SUGGESTED_QUESTIONS):
        with col:
            st.markdown(
                '<div class="chip-btn">',
                unsafe_allow_html=True,
            )

            if st.button(
                question,
                key=f"suggested_{question}",
            ):
                _queue_question(question)

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )


# ============================================================
# CHAT INPUT
# ============================================================
prompt = st.chat_input(
    "Ask about marine conditions, e.g. 'Is it safe to fish near Kochi tomorrow?'"
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

        # This placeholder is the LIVE pipeline.
        pipeline_slot = st.empty()

        # Start with Planner running.
        render_pipeline(
            pipeline_slot,
            initial_state,
            active_node="planner",
            completed_nodes=set(),
        )

        with st.status(
            "🤖 Running Marine Intelligence Pipeline...",
            expanded=True,
        ) as status:
            try:
                for node_name, state in run_marine_graph(
                    initial_state,
                    pipeline_slot=pipeline_slot,
                ):
                    result = state

                    if node_name:
                        status.write(
                            f"✅ **{node_name.replace('_', ' ').title()}** completed"
                        )

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
            st.exception(error)

        elif result:
            # Final completed pipeline remains visible above the final result.
            completed = {
                node_id
                for node_id, _, _ in PIPELINE_DEFS
            }

            render_pipeline(
                pipeline_slot,
                result,
                active_node=None,
                completed_nodes=completed,
            )

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
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
    )


# ============================================================
# FOOTER
# ============================================================
st.markdown(
    textwrap.dedent(
        """
        <div class="footer">
            🌊 ORCA Marine Intelligence Platform
            <br>
            Agentic AI • LangGraph • Marine Data Intelligence
        </div>
        """
    ),
    unsafe_allow_html=True,
)
