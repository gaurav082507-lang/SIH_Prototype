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
    initial_sidebar_state="expanded"
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

# GIS runs unconditionally in graph.py (baseline coastal/EEZ/restricted-zone
# context for every accepted query) rather than being gated on the planner's
# required_agents like the other six — see the "Active Intelligence Agents"
# rendering below and compute_stage_status(), which both special-case it.
ALWAYS_ON_AGENTS = {"gis"}

DATA_KEYS = [
    ("🌤️ Weather", "weather_data"),
    ("🌊 Ocean", "ocean_data"),
    ("🌙 Tide", "tide_data"),
    ("🌀 Cyclone", "cyclone_data"),
    ("🐟 Ecosystem", "ecosystem_data"),
    ("🎣 PFZ", "pfz_data"),
    ("🗺️ GIS", "gis_data"),
]

RISK_COLORS = {
    "LOW": "🟢",
    "MODERATE": "🟡",
    "HIGH": "🟠",
    "SEVERE": "🔴",
}

# Canonical pipeline stages, matched by STATE KEY rather than LangGraph node
# name — this keeps the diagram accurate no matter what the compiled graph's
# node ids actually are, since (state_key present) is the same signal the
# rest of this file already trusts (see render_result / DATA_KEYS).
STAGE_DEFS = [
    ("planner", "🧠", "Planner", "plan"),
    ("gis", "🗺️", "GIS", "gis_data"),
    ("weather", "🌤️", "Weather", "weather_data"),
    ("ocean", "🌊", "Ocean", "ocean_data"),
    ("tide", "🌙", "Tide", "tide_data"),
    ("cyclone", "🌀", "Cyclone", "cyclone_data"),
    ("ecosystem", "🐟", "Ecosystem", "ecosystem_data"),
    ("pfz", "🎣", "PFZ", "pfz_data"),
    ("synthesis", "🎯", "Synthesis", "recommendation"),
]
STAGE_LABELS = {"pending": "Waiting", "active": "Running…", "done": "Done", "skipped": "Not needed"}

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
    textwrap.dedent("""
    <style>
    .stApp {
        background:
            radial-gradient(circle at 10% 10%, rgba(0, 119, 182, 0.18), transparent 30%),
            radial-gradient(circle at 90% 20%, rgba(0, 180, 216, 0.12), transparent 28%),
            radial-gradient(circle at 50% 100%, rgba(3, 64, 120, 0.20), transparent 35%),
            #050b16;
        color: #e8f1ff;
    }
    .block-container { max-width: 1300px; padding-top: 2rem; padding-bottom: 3rem; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #071426 0%, #06101f 50%, #040b16 100%);
        border-right: 1px solid rgba(0, 180, 216, 0.18);
    }
    section[data-testid="stSidebar"] * { color: #dcecff; }
    .hero {
        padding: 30px 34px; border-radius: 24px;
        background: linear-gradient(135deg, rgba(0, 119, 182, 0.22), rgba(0, 180, 216, 0.08), rgba(4, 20, 40, 0.70));
        border: 1px solid rgba(0, 180, 216, 0.20);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.35);
        margin-bottom: 22px;
    }
    .hero-title {
        font-size: 40px; font-weight: 800; letter-spacing: -1px;
        background: linear-gradient(90deg, #ffffff, #8be9ff, #4cc9f0);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero-subtitle { color: #9bb4d0; font-size: 15px; margin-top: 6px; }
    .status-pill {
        display: inline-block; padding: 7px 14px; border-radius: 999px;
        background: rgba(0, 180, 216, 0.12); border: 1px solid rgba(0, 180, 216, 0.25);
        color: #6ee7ff; font-size: 13px; font-weight: 600;
    }
    .glass-card {
        background: linear-gradient(135deg, rgba(15, 35, 60, 0.72), rgba(7, 18, 32, 0.72));
        border: 1px solid rgba(120, 190, 230, 0.13); border-radius: 18px; padding: 18px;
        box-shadow: 0 15px 45px rgba(0, 0, 0, 0.25); backdrop-filter: blur(12px);
        margin-bottom: 10px;
    }
    .card-title { font-size: 13px; color: #82a6c9; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
    .agent-card {
        padding: 16px; border-radius: 16px; text-align: center;
        background: linear-gradient(145deg, rgba(12, 35, 60, 0.75), rgba(6, 17, 31, 0.8));
        border: 1px solid rgba(0, 180, 216, 0.12); min-height: 100px; transition: 0.25s ease;
    }
    .agent-card:hover { transform: translateY(-3px); border-color: rgba(0, 180, 216, 0.38); }
    .agent-icon { font-size: 24px; }
    .agent-name { font-size: 14px; font-weight: 700; margin-top: 6px; }
    .agent-active { color: #5eead4; font-size: 12px; margin-top: 4px; }
    .agent-inactive { color: #64748b; font-size: 12px; margin-top: 4px; }
    .section-title { font-size: 20px; font-weight: 700; color: #edf7ff; margin-top: 18px; margin-bottom: 12px; }
    .recommendation-card {
        padding: 24px; border-radius: 20px;
        background: linear-gradient(135deg, rgba(0, 119, 182, 0.18), rgba(6, 24, 43, 0.82));
        border: 1px solid rgba(0, 180, 216, 0.25); box-shadow: 0 20px 55px rgba(0, 0, 0, 0.28);
    }
    .recommendation-text { font-size: 17px; line-height: 1.7; color: #dcecff; }
    .chip-btn button {
        background: rgba(0, 180, 216, 0.10) !important;
        border: 1px solid rgba(0, 180, 216, 0.30) !important;
        color: #bfe9ff !important; font-weight: 500 !important;
        border-radius: 999px !important; box-shadow: none !important;
    }
    textarea, input { color: #eaf6ff !important; }
    div[data-baseweb="input"] > div, div[data-baseweb="textarea"], div[data-baseweb="select"] > div {
        background-color: rgba(8, 22, 38, 0.85) !important;
        border: 1px solid rgba(80, 170, 210, 0.20) !important; border-radius: 12px !important;
    }
    div.stButton > button {
        width: 100%; border: none; border-radius: 12px; padding: 10px 18px;
        font-weight: 700; color: white;
        background: linear-gradient(90deg, #0077b6, #00b4d8);
        box-shadow: 0 8px 25px rgba(0, 180, 216, 0.20); transition: 0.2s ease;
    }
    div.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 12px 32px rgba(0, 180, 216, 0.32); }
    div[data-testid="stMetric"] {
        background: rgba(10, 28, 48, 0.65); padding: 14px; border-radius: 15px;
        border: 1px solid rgba(120, 190, 230, 0.12);
    }
    div[data-testid="stMetricLabel"] { color: #8da9c4 !important; }
    div[data-testid="stMetricValue"] { color: #eaf7ff !important; }
    button[data-baseweb="tab"] { color: #8faac4 !important; }
    button[data-baseweb="tab"][aria-selected="true"] { color: #67e8f9 !important; }
    details { background: rgba(8, 22, 38, 0.65); border: 1px solid rgba(100, 180, 220, 0.12); border-radius: 12px; }
    .pipe-scroll { overflow-x: auto; padding: 4px 2px 10px 2px; }
    .pipe-row { display: flex; align-items: center; flex-wrap: nowrap; min-width: min-content; }
    .pipe-node {
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        min-width: 92px; padding: 10px 8px; border-radius: 14px; text-align: center;
        border: 1.5px solid rgba(120, 190, 230, 0.18);
        background: rgba(10, 28, 48, 0.55);
        transition: all 0.35s ease;
    }
    .pipe-icon { font-size: 20px; }
    .pipe-name { font-size: 11.5px; font-weight: 700; margin-top: 4px; color: #dcecff; }
    .pipe-state { font-size: 10px; margin-top: 2px; letter-spacing: 0.3px; }
    .pipe-pending { opacity: 0.42; }
    .pipe-pending .pipe-state { color: #7f97b3; }
    .pipe-skipped { opacity: 0.30; border-style: dashed; }
    .pipe-skipped .pipe-state { color: #7f97b3; }
    .pipe-active {
        border-color: #00b4d8; opacity: 1;
        box-shadow: 0 0 0 rgba(0, 180, 216, 0.5);
        animation: pipe-pulse 1.3s ease-in-out infinite;
    }
    .pipe-active .pipe-state { color: #6ee7ff; font-weight: 600; }
    .pipe-done {
        border-color: #34d399; opacity: 1;
        background: rgba(16, 60, 50, 0.55);
    }
    .pipe-done .pipe-state { color: #5eead4; font-weight: 600; }
    .pipe-arrow { flex: 0 0 auto; padding: 0 6px; font-size: 15px; color: rgba(140, 190, 220, 0.30); }
    .pipe-arrow-done { color: #34d399; }
    @keyframes pipe-pulse {
        0%   { box-shadow: 0 0 0 0 rgba(0, 180, 216, 0.45); }
        70%  { box-shadow: 0 0 0 9px rgba(0, 180, 216, 0); }
        100% { box-shadow: 0 0 0 0 rgba(0, 180, 216, 0); }
    }
    div[data-testid="stChatMessage"] {
        background: rgba(9, 24, 42, 0.55); border: 1px solid rgba(100, 180, 220, 0.10);
        border-radius: 16px; padding: 6px 4px;
    }
    .footer {
        text-align: center; color: #526b84; font-size: 12px; margin-top: 40px;
        padding-top: 18px; border-top: 1px solid rgba(100, 180, 220, 0.08);
    }
    </style>
    """),
    unsafe_allow_html=True
)

# ============================================================
# HERO
# ============================================================
st.markdown(
    textwrap.dedent("""
    <div class="hero">
        <div class="status-pill">● AI MARINE INTELLIGENCE SYSTEM</div>
        <div class="hero-title">ORCA Marine Intelligence</div>
        <div class="hero-subtitle">
            Ask in plain language — ORCA's collaborative agents plan, retrieve, and
            reason over weather, ocean, tide, cyclone, ecosystem and PFZ data to
            give you an explainable, evidence-based answer.
        </div>
    </div>
    """),
    unsafe_allow_html=True
)

# ============================================================
# SESSION STATE
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {question, latitude, longitude, result, error}
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
        "Latitude", min_value=-90.0, max_value=90.0, format="%.6f", key="latitude"
    )
    longitude = st.number_input(
        "Longitude", min_value=-180.0, max_value=180.0, format="%.6f", key="longitude"
    )
    st.map(
        pd.DataFrame({"lat": [latitude], "lon": [longitude]}),
        zoom=5,
        height=180,
    )

    st.divider()
    st.markdown("### 🤖 Agent Architecture")
    st.caption("The Planner Agent dynamically decides which specialist agents are required.")
    st.markdown(
        "".join(f"{icon} {name}  \n" for icon, name, _ in AVAILABLE_AGENTS)
    )

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
# RESULT RENDERING
# ============================================================

def extract_map_points(result):
    """Best-effort extraction of extra lat/lon points (e.g. PFZ candidates)
    from agent payloads, so they can be layered on the map. Falls back to
    nothing if the shape isn't recognized — never raises."""
    points = []
    pfz = result.get("pfz_data")
    if isinstance(pfz, dict):
        for key in ("candidates", "zones", "results", "data"):
            val = pfz.get(key)
            if isinstance(val, list):
                for item in val:
                    if not isinstance(item, dict):
                        continue
                    lat = item.get("latitude", item.get("lat"))
                    lon = item.get("longitude", item.get("lon", item.get("lng")))
                    try:
                        if lat is not None and lon is not None:
                            points.append((float(lat), float(lon)))
                    except (TypeError, ValueError):
                        continue
    return points


def _parse_plan(state):
    plan = state.get("plan")
    if isinstance(plan, str):
        try:
            plan = json.loads(plan)
        except Exception:
            return None
    return plan if isinstance(plan, dict) else None


def compute_stage_status(state):
    """Derive each pipeline stage's status purely from which keys are
    present in the (possibly partial) graph state. This is independent of
    whatever the compiled graph's internal node names are, so it works the
    same whether we got here via streamed partial updates or a single
    final invoke()."""
    plan = _parse_plan(state)
    plan_done = bool(state.get("plan"))
    rejected = bool(plan and plan.get("rejected"))
    required = plan.get("required_agents", []) if plan else []

    status = {"planner": "done" if plan_done else "active"}

    agents_pending = False
    for sid, _, _, data_key in STAGE_DEFS[1:-1]:
        if sid in ALWAYS_ON_AGENTS:
            # Runs for every ACCEPTED query regardless of required_agents
            # (see route_after_planner in graph.py) — but a rejected plan
            # never reaches it either, so it's "skipped" there too.
            if rejected:
                status[sid] = "skipped"
            elif not plan_done:
                status[sid] = "pending"
                agents_pending = True
            elif state.get(data_key):
                status[sid] = "done"
            else:
                status[sid] = "active"
                agents_pending = True
            continue

        if not plan_done:
            status[sid] = "pending"
            agents_pending = True
        elif sid not in required:
            status[sid] = "skipped"
        elif state.get(data_key):
            status[sid] = "done"
        else:
            status[sid] = "active"
            agents_pending = True

    if rejected:
        status["synthesis"] = "skipped"
    elif state.get("recommendation"):
        status["synthesis"] = "done"
    elif plan_done and not agents_pending:
        status["synthesis"] = "active"
    else:
        status["synthesis"] = "pending"

    return status


def render_pipeline_html(stage_status):
    nodes = []
    for i, (sid, icon, label, _) in enumerate(STAGE_DEFS):
        state = stage_status.get(sid, "pending")
        nodes.append(
            f'<div class="pipe-node pipe-{state}">'
            f'<div class="pipe-icon">{icon}</div>'
            f'<div class="pipe-name">{label}</div>'
            f'<div class="pipe-state">{STAGE_LABELS[state]}</div>'
            f'</div>'
        )
        if i < len(STAGE_DEFS) - 1:
            arrow_done = "pipe-arrow-done" if state == "done" else ""
            nodes.append(f'<div class="pipe-arrow {arrow_done}">➜</div>')
    return f'<div class="pipe-scroll"><div class="pipe-row">{"".join(nodes)}</div></div>'


def render_result(result, latitude, longitude, key_prefix):
    # --------------------------------------------------------
    # MAP
    # --------------------------------------------------------
    map_points = [(latitude, longitude)] + extract_map_points(result)
    if len(map_points) > 1:
        st.map(pd.DataFrame(map_points, columns=["lat", "lon"]), height=220)

    # --------------------------------------------------------
    # PLANNER
    # --------------------------------------------------------
    plan = result.get("plan")
    if plan:
        if isinstance(plan, str):
            try:
                plan = json.loads(plan)
            except Exception:
                pass
        st.markdown('<div class="section-title">🧠 Planner Decision</div>', unsafe_allow_html=True)
        if isinstance(plan, dict):
            if plan.get("rejected"):
                st.error(plan.get("rejection_reason", "Request rejected by planner."))
            else:
                agents = plan.get("required_agents", [])
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Query Status", "Accepted")
                with col2:
                    st.metric("Agents Selected", len(agents))
                with col3:
                    st.metric("Location", f"{latitude:.3f}, {longitude:.3f}")

                st.markdown("### Active Intelligence Agents")
                cols = st.columns(len(AVAILABLE_AGENTS))
                for col, (icon, name, agent_key) in zip(cols, AVAILABLE_AGENTS):
                    with col:
                        # GIS always runs alongside whatever the planner
                        # selected (see ALWAYS_ON_AGENTS / route_after_planner
                        # in graph.py) — this branch is only reached for an
                        # accepted plan, so it's safe to show it as ACTIVE
                        # unconditionally rather than checking required_agents.
                        active = agent_key in ALWAYS_ON_AGENTS or agent_key in agents
                        status_html = (
                            '<div class="agent-active">● ACTIVE</div>'
                            if active else '<div class="agent-inactive">○ NOT REQUIRED</div>'
                        )
                        st.markdown(
                            textwrap.dedent(f"""
                            <div class="agent-card">
                                <div class="agent-icon">{icon}</div>
                                <div class="agent-name">{name}</div>
                                {status_html}
                            </div>
                            """),
                            unsafe_allow_html=True
                        )
                with st.expander("View Planner JSON"):
                    st.json(plan)
        else:
            st.code(str(plan))

    # --------------------------------------------------------
    # RECOMMENDATION
    # --------------------------------------------------------
    recommendation = result.get("recommendation")
    if recommendation:
        st.markdown('<div class="section-title">🎯 Marine Intelligence Result</div>', unsafe_allow_html=True)
        if isinstance(recommendation, dict):
            risk_level = recommendation.get("risk_level", "UNKNOWN")
            st.metric(
                "Risk Level",
                f"{RISK_COLORS.get(str(risk_level).upper(), '⚪')} {risk_level}"
            )

            tech_error = recommendation.get("error")
            if tech_error:
                st.error(
                    "⚠️ The recommendation agent hit an error and fell back to a "
                    "generic response — the summary/recommendation below are NOT a "
                    "real assessment."
                )
                with st.expander("🛠️ Technical error details", expanded=True):
                    st.code(str(tech_error), language="text")

            summary = recommendation.get("summary")
            if summary:
                st.markdown(
                    textwrap.dedent(f"""
                    <div class="recommendation-card">
                        <div class="card-title">Marine Analysis</div>
                        <div class="recommendation-text">{summary}</div>
                    </div>
                    """),
                    unsafe_allow_html=True
                )

            recommendation_text = recommendation.get("recommendation")
            if recommendation_text:
                st.markdown("### 🚢 Recommendation")
                st.info(recommendation_text)

            key_findings = recommendation.get("key_findings")
            if key_findings:
                st.markdown("### ⚠️ Key Findings")
                for finding in key_findings:
                    st.markdown(f'<div class="glass-card">{finding}</div>', unsafe_allow_html=True)

            safety_advice = recommendation.get("safety_advice")
            if safety_advice:
                st.markdown("### 🧭 Safety Advice")
                for advice in safety_advice:
                    st.warning(advice)

            findings = recommendation.get("agent_findings")
            if findings:
                st.markdown("### 🤖 Agent Findings")
                for i, (agent, finding) in enumerate(findings.items()):
                    with st.expander(f"{agent.capitalize()} Agent"):
                        if isinstance(finding, dict):
                            st.json(finding)
                        else:
                            st.write(finding)

            limitations = recommendation.get("limitations")
            if limitations:
                with st.expander("⚠️ Analysis Limitations"):
                    for limitation in limitations:
                        st.write(f"• {limitation}")
        else:
            st.write(recommendation)

    # --------------------------------------------------------
    # RAW AGENT DATA
    # --------------------------------------------------------
    st.markdown('<div class="section-title">📊 Agent Data</div>', unsafe_allow_html=True)
    tabs = st.tabs([label for label, _ in DATA_KEYS])
    for tab, (label, dkey) in zip(tabs, DATA_KEYS):
        with tab:
            data = result.get(dkey)
            if data:
                st.json(data)
            else:
                st.info("This agent was not selected for the current query.")

    with st.expander("🔧 Complete LangGraph Output"):
        st.json(result)


def run_marine_graph(initial_state):
    """Invoke the graph. If the compiled graph supports .stream(), surface
    live per-agent progress; otherwise fall back to a single invoke()."""
    node_order = []
    final_state = dict(initial_state)
    try:
        for step_output in marine_graph.stream(initial_state, stream_mode="updates"):
            if not isinstance(step_output, dict):
                continue
            for node_name, node_update in step_output.items():
                node_order.append(node_name)
                if isinstance(node_update, dict):
                    final_state.update(node_update)
                yield node_name, dict(final_state)
        if node_order:
            return
    except Exception:
        pass
    # Fallback: no streaming support, or it produced nothing usable.
    final_state = marine_graph.invoke(initial_state)
    yield None, final_state


# ============================================================
# CONVERSATION HISTORY
# ============================================================
if st.session_state.messages:
    top_cols = st.columns([6, 1])
    with top_cols[1]:
        if st.button("🧹 Clear Chat", key="clear_chat_top", help="Clear the conversation history."):
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
            st.markdown(render_pipeline_html(compute_stage_status(msg["result"])), unsafe_allow_html=True)
            render_result(msg["result"], msg["latitude"], msg["longitude"], key_prefix=f"msg{idx}")

# ============================================================
# SUGGESTED QUESTIONS (only while the conversation is empty)
# ============================================================
if not st.session_state.messages:
    st.markdown('<div class="section-title">💡 Try asking</div>', unsafe_allow_html=True)
    cols = st.columns(len(SUGGESTED_QUESTIONS))
    for col, question in zip(cols, SUGGESTED_QUESTIONS):
        with col:
            st.markdown('<div class="chip-btn">', unsafe_allow_html=True)
            if st.button(question, key=f"suggested_{question}"):
                _queue_question(question)
            st.markdown('</div>', unsafe_allow_html=True)

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
        pipeline_slot = st.empty()
        pipeline_slot.markdown(
            render_pipeline_html(compute_stage_status(initial_state)), unsafe_allow_html=True
        )
        with st.status("🤖 Running Marine Intelligence Agents...", expanded=True) as status:
            try:
                for node_name, state in run_marine_graph(initial_state):
                    result = state
                    pipeline_slot.markdown(
                        render_pipeline_html(compute_stage_status(state)), unsafe_allow_html=True
                    )
                    if node_name:
                        status.write(f"✅ **{node_name.replace('_', ' ').title()}** agent completed")
                status.update(
                    label="Marine intelligence analysis completed",
                    state="complete",
                    expanded=False,
                )
            except Exception as e:
                error = e
                status.update(label="Marine analysis failed", state="error", expanded=True)

        if error:
            st.error("❌ Marine analysis failed.")
            st.exception(error)
        elif result:
            render_result(result, latitude, longitude, key_prefix=f"live{len(st.session_state.messages)}")

    st.session_state.messages.append({
        "question": prompt,
        "latitude": latitude,
        "longitude": longitude,
        "result": result,
        "error": error,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    textwrap.dedent("""
    <div class="footer">
        🌊 ORCA Marine Intelligence Platform
        <br>
        Agentic AI • LangGraph • Marine Data Intelligence
    </div>
    """),
    unsafe_allow_html=True
)
