import streamlit as st
import pandas as pd
import textwrap
import html
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
# CUSTOM CSS
# ============================================================

st.markdown(
    textwrap.dedent(
        """
        <style>

        .stApp {
            background:
                radial-gradient(
                    circle at 10% 10%,
                    rgba(0, 119, 182, 0.18),
                    transparent 30%
                ),
                radial-gradient(
                    circle at 90% 20%,
                    rgba(0, 180, 216, 0.12),
                    transparent 28%
                ),
                radial-gradient(
                    circle at 50% 100%,
                    rgba(3, 64, 120, 0.20),
                    transparent 35%
                ),
                #050b16;

            color: #e8f1ff;
        }


        .block-container {
            max-width: 1300px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }


        section[data-testid="stSidebar"] {
            background:
                linear-gradient(
                    180deg,
                    #071426 0%,
                    #06101f 50%,
                    #040b16 100%
                );

            border-right: 1px solid rgba(0, 180, 216, 0.18);
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
                    rgba(0, 119, 182, 0.22),
                    rgba(0, 180, 216, 0.08),
                    rgba(4, 20, 40, 0.70)
                );

            border: 1px solid rgba(0, 180, 216, 0.20);

            box-shadow:
                0 20px 60px rgba(0, 0, 0, 0.35);

            margin-bottom: 22px;
        }


        .hero-title {
            font-size: 40px;
            font-weight: 800;
            letter-spacing: -1px;

            background:
                linear-gradient(
                    90deg,
                    #ffffff,
                    #8be9ff,
                    #4cc9f0
                );

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

            background: rgba(0, 180, 216, 0.12);
            border: 1px solid rgba(0, 180, 216, 0.25);

            color: #6ee7ff;
            font-size: 13px;
            font-weight: 600;
        }


        .section-title {
            font-size: 22px;
            font-weight: 700;

            color: #edf7ff;

            margin-top: 24px;
            margin-bottom: 14px;
        }


        .risk-card {
            padding: 28px;

            border-radius: 20px;

            background:
                linear-gradient(
                    135deg,
                    rgba(15, 35, 60, 0.78),
                    rgba(7, 18, 32, 0.90)
                );

            border: 1px solid rgba(120, 190, 230, 0.16);

            box-shadow:
                0 20px 55px rgba(0, 0, 0, 0.28);

            margin-bottom: 18px;
        }


        .risk-label {
            color: #8da9c4;

            font-size: 13px;

            text-transform: uppercase;

            letter-spacing: 1px;

            margin-bottom: 8px;
        }


        .risk-value {
            font-size: 38px;

            font-weight: 800;

            color: #edf7ff;
        }


        .recommendation-card {
            padding: 26px;

            border-radius: 20px;

            background:
                linear-gradient(
                    135deg,
                    rgba(0, 119, 182, 0.18),
                    rgba(6, 24, 43, 0.82)
                );

            border: 1px solid rgba(0, 180, 216, 0.25);

            box-shadow:
                0 20px 55px rgba(0, 0, 0, 0.28);

            margin-bottom: 18px;
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


        .recommendation-box {
            padding: 24px;

            border-radius: 18px;

            background:
                rgba(10, 28, 48, 0.72);

            border: 1px solid rgba(120, 190, 230, 0.14);

            color: #dcecff;

            font-size: 17px;

            line-height: 1.7;

            margin-bottom: 10px;
        }


        .chip-btn button {
            background:
                rgba(0, 180, 216, 0.10) !important;

            border:
                1px solid rgba(0, 180, 216, 0.30) !important;

            color:
                #bfe9ff !important;

            font-weight:
                500 !important;

            border-radius:
                999px !important;

            box-shadow:
                none !important;
        }


        textarea,
        input {
            color: #eaf6ff !important;
        }


        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"],
        div[data-baseweb="select"] > div {

            background-color:
                rgba(8, 22, 38, 0.85) !important;

            border:
                1px solid rgba(80, 170, 210, 0.20) !important;

            border-radius:
                12px !important;
        }


        div.stButton > button {

            width: 100%;

            border: none;

            border-radius: 12px;

            padding: 10px 18px;

            font-weight: 700;

            color: white;

            background:
                linear-gradient(
                    90deg,
                    #0077b6,
                    #00b4d8
                );

            box-shadow:
                0 8px 25px rgba(0, 180, 216, 0.20);

            transition:
                0.2s ease;
        }


        div.stButton > button:hover {

            transform:
                translateY(-2px);

            box-shadow:
                0 12px 32px rgba(0, 180, 216, 0.32);
        }


        .footer {

            text-align: center;

            color: #526b84;

            font-size: 12px;

            margin-top: 40px;

            padding-top: 18px;

            border-top:
                1px solid rgba(100, 180, 220, 0.08);
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

            <div class="status-pill">
                ● AI MARINE INTELLIGENCE SYSTEM
            </div>

            <div class="hero-title">
                ORCA Marine Intelligence
            </div>

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


# ============================================================
# HELPERS
# ============================================================

def _apply_preset():
    preset = LOCATION_PRESETS.get(
        st.session_state.location_preset
    )

    if preset:
        (
            st.session_state.latitude,
            st.session_state.longitude,
        ) = preset


def _queue_question(text):
    st.session_state.pending_question = text


def _safe_html(value):
    """
    Escape model-generated text before inserting it into HTML.
    """
    if value is None:
        return ""

    return html.escape(
        str(value)
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🌊 ORCA")

    st.caption(
        "Marine Intelligence Platform"
    )

    st.divider()

    st.markdown(
        "### 📍 Target Location"
    )

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
        pd.DataFrame(
            {
                "lat": [latitude],
                "lon": [longitude],
            }
        ),
        zoom=5,
        height=180,
    )

    st.divider()

    st.markdown(
        "### 🤖 Intelligence System"
    )

    st.caption(
        "ORCA automatically selects the marine intelligence "
        "agents required for each question."
    )

    st.markdown(
        "".join(
            f"{icon} {name}  \n"
            for icon, name, _ in AVAILABLE_AGENTS
        )
    )

    st.divider()

    if st.button(
        "🧹 Clear Chat",
        disabled=not st.session_state.messages,
        help="Clear the conversation history.",
    ):

        st.session_state.messages = []

        st.session_state.pending_question = None

        st.rerun()


# ============================================================
# FINAL RESULT RENDERING
# ============================================================

def render_final_result(result):
    """
    Display ONLY the final user-facing recommendation.

    Hidden from UI:
    - Planner decision
    - Planner JSON
    - Agent statuses
    - Agent findings
    - Raw agent data
    - Complete LangGraph state
    - Internal technical errors
    """

    if not isinstance(result, dict):
        st.warning(
            "No valid marine intelligence result was generated."
        )
        return


    recommendation = result.get(
        "recommendation"
    )


    if not recommendation:
        st.warning(
            "No final marine recommendation was generated."
        )
        return


    if not isinstance(recommendation, dict):
        st.warning(
            "Invalid final recommendation format."
        )
        return


    # ========================================================
    # RISK LEVEL
    # ========================================================

    risk_level = str(
        recommendation.get(
            "risk_level",
            "UNKNOWN"
        )
    ).upper().strip()


    risk_icon = RISK_COLORS.get(
        risk_level,
        "⚪"
    )


    st.markdown(
        '<div class="section-title">'
        '🎯 Marine Intelligence Result'
        '</div>',
        unsafe_allow_html=True,
    )


    safe_risk_icon = _safe_html(
        risk_icon
    )

    safe_risk_level = _safe_html(
        risk_level
    )


    st.markdown(
        textwrap.dedent(
            f"""
            <div class="risk-card">

                <div class="risk-label">
                    Risk Level
                </div>

                <div class="risk-value">
                    {safe_risk_icon}
                    {safe_risk_level}
                </div>

            </div>
            """
        ),
        unsafe_allow_html=True,
    )


    # ========================================================
    # MARINE ANALYSIS
    # ========================================================

    summary = recommendation.get(
        "summary"
    )


    if summary:

        safe_summary = _safe_html(
            summary
        )

        st.markdown(
            textwrap.dedent(
                f"""
                <div class="recommendation-card">

                    <div class="card-title">
                        Marine Analysis
                    </div>

                    <div class="recommendation-text">
                        {safe_summary}
                    </div>

                </div>
                """
            ),
            unsafe_allow_html=True,
        )


    # ========================================================
    # FINAL RECOMMENDATION
    # ========================================================

    final_recommendation = recommendation.get(
        "recommendation"
    )


    if final_recommendation:

        safe_recommendation = _safe_html(
            final_recommendation
        )

        st.markdown(
            "### 🚢 Recommendation"
        )

        st.markdown(
            textwrap.dedent(
                f"""
                <div class="recommendation-box">
                    {safe_recommendation}
                </div>
                """
            ),
            unsafe_allow_html=True,
        )


    # ========================================================
    # KEY FINDINGS
    # ========================================================

    key_findings = recommendation.get(
        "key_findings"
    )


    if (
        isinstance(key_findings, list)
        and key_findings
    ):

        st.markdown(
            "### ⚠️ Key Findings"
        )

        for finding in key_findings:

            safe_finding = _safe_html(
                finding
            )

            st.markdown(
                textwrap.dedent(
                    f"""
                    <div class="recommendation-box">
                        • {safe_finding}
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )


    # ========================================================
    # SAFETY ADVICE
    # ========================================================

    safety_advice = recommendation.get(
        "safety_advice"
    )


    if (
        isinstance(safety_advice, list)
        and safety_advice
    ):

        st.markdown(
            "### 🧭 Safety Advice"
        )

        for advice in safety_advice:

            st.warning(
                str(advice)
            )


# ============================================================
# GRAPH EXECUTION
# ============================================================

def run_marine_graph(initial_state):
    """
    Run the LangGraph pipeline.

    Intermediate states are intentionally not rendered
    to the user. Only the final state is returned to the UI.
    """

    final_state = dict(
        initial_state
    )


    try:

        got_updates = False


        for step_output in marine_graph.stream(
            initial_state,
            stream_mode="updates",
        ):

            if not isinstance(
                step_output,
                dict
            ):
                continue


            got_updates = True


            for _, node_update in step_output.items():

                if isinstance(
                    node_update,
                    dict
                ):

                    final_state.update(
                        node_update
                    )


        if got_updates:
            yield final_state
            return


    except Exception:

        pass


    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    final_state = marine_graph.invoke(
        initial_state
    )


    yield final_state


# ============================================================
# EXISTING CONVERSATION
# ============================================================

for msg in st.session_state.messages:

    with st.chat_message("user"):

        st.markdown(
            msg["question"]
        )


    with st.chat_message(
        "assistant",
        avatar="🌊",
    ):

        if msg.get("error"):

            st.error(
                "Marine analysis failed. "
                "Please try again."
            )

        elif msg.get("result"):

            render_final_result(
                msg["result"]
            )


# ============================================================
# SUGGESTED QUESTIONS
# ============================================================

if not st.session_state.messages:

    st.markdown(
        '<div class="section-title">'
        '💡 Try asking'
        '</div>',
        unsafe_allow_html=True,
    )


    cols = st.columns(
        len(SUGGESTED_QUESTIONS)
    )


    for col, question in zip(
        cols,
        SUGGESTED_QUESTIONS,
    ):

        with col:

            st.markdown(
                '<div class="chip-btn">',
                unsafe_allow_html=True,
            )


            if st.button(
                question,
                key=f"suggested_{question}",
            ):

                _queue_question(
                    question
                )


            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )


# ============================================================
# CHAT INPUT
# ============================================================

prompt = st.chat_input(
    "Ask about marine conditions, e.g. "
    "'Is it safe to fish near Kochi tomorrow?'"
)


if (
    not prompt
    and st.session_state.pending_question
):

    prompt = (
        st.session_state.pending_question
    )

    st.session_state.pending_question = None


# ============================================================
# PROCESS QUESTION
# ============================================================

if prompt:

    with st.chat_message("user"):

        st.markdown(
            prompt
        )


    initial_state = {
        "latitude": float(
            latitude
        ),

        "longitude": float(
            longitude
        ),

        "user_question": prompt.strip(),

        "status": "STARTED",
    }


    with st.chat_message(
        "assistant",
        avatar="🌊",
    ):

        result = None
        error = None


        with st.status(
            "🤖 Analyzing marine conditions...",
            expanded=False,
        ) as status:

            try:

                for state in run_marine_graph(
                    initial_state
                ):

                    result = state


                status.update(
                    label=(
                        "Marine intelligence "
                        "analysis completed"
                    ),
                    state="complete",
                    expanded=False,
                )


            except Exception as e:

                error = e


                status.update(
                    label="Marine analysis failed",
                    state="error",
                    expanded=False,
                )


        if error:

            st.error(
                "❌ Marine analysis failed. "
                "Please try again."
            )

        elif result:

            render_final_result(
                result
            )


    # --------------------------------------------------------
    # SAVE CONVERSATION
    # --------------------------------------------------------

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
