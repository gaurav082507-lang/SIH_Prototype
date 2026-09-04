import json
import textwrap
from datetime import datetime

import pandas as pd
import pydeck as pdk
import streamlit as st

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

ALWAYS_ON_AGENTS = {"gis"}

# Order matters for layout purposes: index 0 = Planner, index 1 = GIS,
# the middle block are the agents that can run in parallel once GIS/Planner
# have finished, and the last entry is the final synthesis step.
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
    "pending": "WAITING",
    "active": "RUNNING",
    "done": "COMPLETED",
    "skipped": "NOT REQUIRED",
}

RISK_COLORS = {
    "LOW": "🟢",
    "MODERATE": "🟡",
    "HIGH": "🟠",
    "SEVERE": "🔴",
    "UNKNOWN": "⚪",
}

SUGGESTED_QUESTIONS = [
    "Where is the nearest Potential Fishing Zone today?",
    "Is it safe to venture into the sea tomorrow morning?",
    "What are the tide, weather, and sea conditions near my location?",
    "Are there any cyclone or lightning alerts nearby?",
]


# ============================================================
# CSS
# ============================================================

st.markdown(
    textwrap.dedent(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 10% 10%, rgba(0,119,182,.18), transparent 30%),
                radial-gradient(circle at 90% 20%, rgba(0,180,216,.12), transparent 28%),
                #050b16;
            color: #e8f1ff;
        }

        .block-container {
            max-width: 1350px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg,#071426 0%,#06101f 50%,#040b16 100%);
            border-right: 1px solid rgba(0,180,216,.18);
        }

        section[data-testid="stSidebar"] * {
            color: #dcecff;
        }

        .hero {
            padding: 28px 32px;
            border-radius: 22px;
            background: linear-gradient(
                135deg,
                rgba(0,119,182,.22),
                rgba(0,180,216,.08),
                rgba(4,20,40,.70)
            );
            border: 1px solid rgba(0,180,216,.20);
            margin-bottom: 20px;
        }

        .hero-title {
            font-size: 38px;
            font-weight: 800;
            margin-top: 8px;
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
            font-size: 21px;
            font-weight: 750;
            color: #edf7ff;
            margin-top: 20px;
            margin-bottom: 12px;
        }

        .pipeline-shell {
            padding: 16px;
            border-radius: 18px;
            background: rgba(7,22,38,.70);
            border: 1px solid rgba(0,180,216,.18);
            margin-bottom: 18px;
        }

        .pipeline-caption {
            color: #6ee7ff;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 1.2px;
            margin-bottom: 10px;
            text-align: center;
        }

        /* ---- Parallel flow layout ---- */

        .pipeline-flow {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 6px 0 2px;
        }

        .flow-node {
            min-width: 128px;
            padding: 12px 10px;
            border-radius: 14px;
            text-align: center;
            border: 1px solid rgba(120,190,230,.18);
            background: rgba(10,28,48,.70);
        }

        .flow-node.pending {
            opacity: .60;
        }

        .flow-node.active {
            border-color: #00b4d8;
            box-shadow: 0 0 0 2px rgba(0,180,216,.12);
        }

        .flow-node.done {
            border-color: #34d399;
            background: rgba(16,60,50,.55);
        }

        .flow-node.skipped {
            opacity: .35;
            border-style: dashed;
        }

        .pipe-icon {
            font-size: 21px;
        }

        .pipe-name {
            font-size: 11px;
            font-weight: 700;
            margin-top: 4px;
            color: #dcecff;
        }

        .pipe-state {
            font-size: 9px;
            margin-top: 4px;
            letter-spacing: .3px;
        }

        .flow-node.active .pipe-state {
            color: #6ee7ff;
            font-weight: 700;
        }

        .flow-node.done .pipe-state {
            color: #5eead4;
            font-weight: 700;
        }

        .flow-node.pending .pipe-state,
        .flow-node.skipped .pipe-state {
            color: #7f97b3;
        }

        .flow-connector {
            width: 2px;
            height: 22px;
            background: rgba(120,190,230,.25);
        }

        .flow-connector.active {
            background: #00b4d8;
        }

        .flow-connector.done {
            background: #34d399;
        }

        .branch-row-wrap {
            width: 100%;
            overflow-x: auto;
            display: flex;
            justify-content: center;
            padding-bottom: 4px;
        }

        .branch-container {
            display: inline-flex;
            flex-direction: column;
            align-items: stretch;
        }

        .branch-top-line {
            height: 2px;
            background: rgba(120,190,230,.25);
            margin: 0 68px;
        }

        .branch-row {
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 14px;
            padding-top: 0;
        }

        .branch-item {
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .stem {
            width: 2px;
            height: 16px;
            background: rgba(120,190,230,.25);
        }

        .stem.active {
            background: #00b4d8;
        }

        .stem.done {
            background: #34d399;
        }

        /* ---- End parallel flow layout ---- */

        .glass-card {
            padding: 22px;
            border-radius: 20px;
            background: linear-gradient(
                135deg,
                rgba(0,119,182,.16),
                rgba(6,24,43,.84)
            );
            border: 1px solid rgba(0,180,216,.23);
            margin-bottom: 12px;
        }

        .card-title {
            font-size: 13px;
            color: #82a6c9;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }

        .risk-value {
            font-size: 42px;
            font-weight: 750;
            color: #eaf7ff;
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
        }

        .recommendation-text {
            font-size: 17px;
            line-height: 1.7;
            color: #dcecff;
        }

        .final-recommendation {
            padding: 18px 20px;
            border-radius: 14px;
            background: rgba(16,70,58,.55);
            border: 1px solid rgba(52,211,153,.30);
            color: #dfffee;
            font-size: 17px;
            line-height: 1.6;
        }

        .error-card {
            padding: 16px;
            border-radius: 14px;
            background: rgba(127,29,29,.35);
            border: 1px solid rgba(248,113,113,.35);
            color: #fecaca;
        }

        .footer {
            text-align: center;
            color: #526b84;
            font-size: 12px;
            margin-top: 40px;
            padding-top: 18px;
            border-top: 1px solid rgba(100,180,220,.08);
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
    """
    <div class="hero">
        <div class="status-pill">● AI MARINE INTELLIGENCE SYSTEM</div>
        <div class="hero-title">ORCA Marine Intelligence</div>
        <div class="hero-subtitle">
            Ask in plain language — ORCA plans, executes specialist agents,
            and produces a final evidence-based marine assessment.
        </div>
    </div>
    """,
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

    # Interactive target-location map with a visible pin marker.
    location_df = pd.DataFrame(
        {
            "lat": [latitude],
            "lon": [longitude],
            "label": ["📍"],
        }
    )

    st.pydeck_chart(
        pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(
                latitude=float(latitude),
                longitude=float(longitude),
                zoom=5,
                pitch=0,
                bearing=0,
            ),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=location_df,
                    get_position="[lon, lat]",
                    get_radius=1800,
                    pickable=True,
                ),
                pdk.Layer(
                    "TextLayer",
                    data=location_df,
                    get_position="[lon, lat]",
                    get_text="label",
                    get_size=28,
                    get_alignment_baseline="'bottom'",
                    get_text_anchor="'middle'",
                    billboard=True,
                ),
            ],
            tooltip={"text": "Selected Location\nLat: {lat}\nLon: {lon}"},
        ),
        height=180,
        use_container_width=True,
    )

    st.divider()
    st.markdown("### 🤖 Agent Architecture")
    st.caption(
        "Planner dynamically selects specialists. GIS runs for accepted queries."
    )

    for icon, name, _ in AVAILABLE_AGENTS:
        st.markdown(f"{icon} {name}")

    st.divider()

    if st.button(
        "🧹 Clear Chat",
        disabled=not st.session_state.messages,
    ):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()


# ============================================================
# PIPELINE HELPERS
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

    selected = {
        str(agent).strip().lower()
        for agent in required
    }

    selected.update(ALWAYS_ON_AGENTS)

    return selected


def compute_pipeline_status(
    state,
    active_node=None,
    completed_nodes=None,
):
    """
    Determine the visual state of every pipeline stage.

    Once Planner finishes, every selected specialist that has not completed
    is shown as RUNNING because LangGraph can execute those branches in
    parallel. This makes the UI reflect actual branch execution instead of
    pretending they are strictly sequential.
    """

    completed_nodes = {
        str(node)
        for node in (completed_nodes or set())
    }

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

    # Specialists
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
        elif "planner" in completed_nodes or plan_exists:
            # Selected branches are launched after Planner. Stream updates
            # arrive when each branch completes, so unfinished selected
            # branches are currently executing/waiting in the graph.
            status[node_id] = "active"
        else:
            status[node_id] = "pending"

    # Final synthesis
    if rejected:
        status["recommendation"] = "skipped"
    elif "recommendation" in completed_nodes or state.get("recommendation"):
        status["recommendation"] = "done"
    elif active_node == "recommendation":
        status["recommendation"] = "active"
    elif all(
        status.get(node_id) in {"done", "skipped"}
        for node_id, _, _ in PIPELINE_DEFS[1:-1]
    ):
        status["recommendation"] = "active"
    else:
        status["recommendation"] = "pending"

    return status


def _connector_class(status):
    if status == "done":
        return "done"
    if status == "active":
        return "active"
    return ""


def render_pipeline_html(stage_status):
    """
    HTML is used ONLY for the pipeline visualization.

    Rendered as a parallel flow diagram: Planner -> GIS -> a fan-out row of
    specialist agents that run concurrently -> Final Assessment. This
    mirrors how LangGraph actually executes the specialist branches (in
    parallel) instead of showing them as one long sequential chain.
    The final result itself uses native Streamlit components, so raw HTML
    cannot accidentally appear as visible text in the assessment.
    """

    def node_html(node_id, icon, label):
        current = stage_status.get(node_id, "pending")

        return (
            f'<div class="flow-node {current}">'
            f'<div class="pipe-icon">{icon}</div>'
            f'<div class="pipe-name">{label}</div>'
            f'<div class="pipe-state">{STAGE_LABELS[current]}</div>'
            f'</div>'
        )

    def connector_html(status):
        return f'<div class="flow-connector {_connector_class(status)}"></div>'

    planner_id, planner_icon, planner_label = PIPELINE_DEFS[0]
    gis_id, gis_icon, gis_label = PIPELINE_DEFS[1]
    recommendation_id, rec_icon, rec_label = PIPELINE_DEFS[-1]
    branch_defs = PIPELINE_DEFS[2:-1]

    gis_status = stage_status.get(gis_id, "pending")
    recommendation_status = stage_status.get(recommendation_id, "pending")

    branch_statuses = [
        stage_status.get(node_id, "pending") for node_id, _, _ in branch_defs
    ]

    if branch_statuses and all(status == "done" for status in branch_statuses):
        branch_overall_status = "done"
    elif any(status in ("done", "active") for status in branch_statuses):
        branch_overall_status = "active"
    else:
        branch_overall_status = "pending"

    branch_items = []

    for node_id, icon, label in branch_defs:
        stem_status = stage_status.get(node_id, "pending")

        branch_items.append(
            f'<div class="branch-item">'
            f'<div class="stem {_connector_class(stem_status)}"></div>'
            f'{node_html(node_id, icon, label)}'
            f'</div>'
        )

    branch_row_html = (
        '<div class="branch-row-wrap">'
        '<div class="branch-container">'
        '<div class="branch-top-line"></div>'
        '<div class="branch-row">'
        + "".join(branch_items)
        + "</div></div></div>"
    )

    return (
        '<div class="pipeline-shell">'
        '<div class="pipeline-caption">PARALLEL PIPELINE EXECUTION</div>'
        '<div class="pipeline-flow">'
        + node_html(planner_id, planner_icon, planner_label)
        + connector_html(gis_status)
        + node_html(gis_id, gis_icon, gis_label)
        + connector_html(branch_overall_status)
        + branch_row_html
        + connector_html(recommendation_status)
        + node_html(recommendation_id, rec_icon, rec_label)
        + "</div></div>"
    )


def render_pipeline(
    pipeline_slot,
    state,
    active_node=None,
    completed_nodes=None,
):
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

def run_marine_graph(initial_state, pipeline_slot):
    """
    Stream the real LangGraph execution.

    No silent invoke() fallback is used. If stream() fails, the real error
    is shown to the user instead of running the entire graph a second time.
    """

    final_state = dict(initial_state)
    completed_nodes = set()

    render_pipeline(
        pipeline_slot,
        final_state,
        active_node="planner",
        completed_nodes=completed_nodes,
    )

    stream = marine_graph.stream(
        initial_state,
        stream_mode="updates",
    )

    yielded_any = False

    for step_output in stream:
        if not isinstance(step_output, dict):
            continue

        for node_name, node_update in step_output.items():
            node_name = str(node_name)
            yielded_any = True

            completed_nodes.add(node_name)

            if isinstance(node_update, dict):
                final_state.update(node_update)

            # Planner just completed: selected branches are now running.
            # If a specialist completes, it becomes DONE while the other
            # selected branches remain RUNNING.
            if node_name == "planner":
                active_node = None
            elif node_name in {
                "weather",
                "ocean",
                "tide",
                "cyclone",
                "ecosystem",
                "pfz",
                "gis",
            }:
                active_node = None
            elif node_name == "recommendation":
                active_node = None
            else:
                active_node = None

            render_pipeline(
                pipeline_slot,
                final_state,
                active_node=active_node,
                completed_nodes=completed_nodes,
            )

            yield node_name, dict(final_state)

    if not yielded_any:
        raise RuntimeError(
            "LangGraph stream returned no execution updates."
        )


# ============================================================
# FINAL RESULT
# ============================================================

def _normalize_recommendation(result):
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
            "summary": "No final assessment was returned.",
            "risk_level": "UNKNOWN",
            "recommendation": "",
        }

    return recommendation


def render_final_result(result, latitude, longitude):
    recommendation = _normalize_recommendation(result)

    risk_level = str(
        recommendation.get("risk_level", "UNKNOWN")
    ).upper().strip()

    risk_icon = RISK_COLORS.get(risk_level, "⚪")

    st.markdown(
        '<div class="section-title">🎯 Marine Intelligence Result</div>',
        unsafe_allow_html=True,
    )

    risk_col, location_col = st.columns([2, 1])

    with risk_col:
        with st.container(border=True):
            st.caption("FINAL RISK LEVEL")
            st.markdown(
                f'<div class="risk-value">{risk_icon} {risk_level}</div>',
                unsafe_allow_html=True,
            )

    with location_col:
        with st.container(border=True):
            st.caption("LOCATION")
            st.markdown(
                f"### {latitude:.3f}, {longitude:.3f}"
            )

    st.markdown(
        '<div class="section-title">🌊 Final Assessment</div>',
        unsafe_allow_html=True,
    )

    summary = str(
        recommendation.get(
            "summary",
            "No final assessment was returned.",
        )
    )

    with st.container(border=True):
        st.caption("MARINE ANALYSIS")
        st.write(summary)

    st.markdown(
        '<div class="section-title">🚢 Recommendation</div>',
        unsafe_allow_html=True,
    )

    final_recommendation = str(
        recommendation.get("recommendation", "")
    ).strip()

    if final_recommendation:
        st.success(final_recommendation)
    else:
        st.warning(
            "The recommendation agent did not return a recommendation."
        )

    key_findings = recommendation.get("key_findings", [])

    if key_findings:
        st.markdown(
            '<div class="section-title">⚠️ Key Findings</div>',
            unsafe_allow_html=True,
        )

        for finding in key_findings:
            st.info(str(finding))

    safety_advice = recommendation.get("safety_advice", [])

    if safety_advice:
        st.markdown(
            '<div class="section-title">🧭 Safety Advice</div>',
            unsafe_allow_html=True,
        )

        for advice in safety_advice:
            st.warning(str(advice))

    error_message = recommendation.get("error")

    if error_message:
        st.markdown(
            '<div class="section-title">⚠️ Recommendation Error</div>',
            unsafe_allow_html=True,
        )

        st.error(str(error_message))


# ============================================================
# CHAT HISTORY
# ============================================================

if st.session_state.messages:
    top_cols = st.columns([6, 1])

    with top_cols[1]:
        if st.button(
            "🧹 Clear Chat",
            key="clear_chat_top",
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
            pipeline_slot = st.empty()

            completed = {
                node_id
                for node_id, _, _ in PIPELINE_DEFS
            }

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
            if st.button(
                question,
                key=f"suggested_{question}",
            ):
                _queue_question(question)


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
        pipeline_slot = st.empty()

        render_pipeline(
            pipeline_slot,
            initial_state,
            active_node="planner",
            completed_nodes=set(),
        )

        result = None
        error = None

        with st.status(
            "🤖 Running Marine Intelligence Pipeline...",
            expanded=True,
        ) as run_status:
            try:
                for node_name, state in run_marine_graph(
                    initial_state,
                    pipeline_slot,
                ):
                    result = state

                    run_status.write(
                        f"✅ **{node_name.replace('_', ' ').title()}** completed"
                    )

                run_status.update(
                    label="Marine intelligence analysis completed",
                    state="complete",
                    expanded=False,
                )

            except Exception as exc:
                error = exc

                run_status.update(
                    label="Marine intelligence pipeline failed",
                    state="error",
                    expanded=True,
                )

        if error:
            st.error("❌ Marine analysis failed.")
            st.exception(error)

        elif result:
            # Keep the completed pipeline visible.
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
    """
    <div class="footer">
        🌊 ORCA Marine Intelligence Platform
        <br>
        Agentic AI • LangGraph • Marine Data Intelligence
    </div>
    """,
    unsafe_allow_html=True,
)
