import streamlit as st
import json

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
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GLOBAL
       ====================================================== */

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


    /* ======================================================
       MAIN CONTAINER
       ====================================================== */

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* ======================================================
       SIDEBAR
       ====================================================== */

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


    /* ======================================================
       HEADER
       ====================================================== */

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

        margin-bottom: 28px;
    }


    .hero-title {
        font-size: 42px;
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
        font-size: 16px;
        margin-top: 8px;
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


    /* ======================================================
       GLASS CARDS
       ====================================================== */

    .glass-card {
        background:
            linear-gradient(
                135deg,
                rgba(15, 35, 60, 0.72),
                rgba(7, 18, 32, 0.72)
            );

        border: 1px solid rgba(120, 190, 230, 0.13);

        border-radius: 18px;

        padding: 22px;

        box-shadow:
            0 15px 45px rgba(0, 0, 0, 0.25);

        backdrop-filter: blur(12px);
    }


    .card-title {
        font-size: 14px;
        color: #82a6c9;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }


    .card-value {
        font-size: 25px;
        font-weight: 700;
        color: #f1f8ff;
    }


    /* ======================================================
       AGENT CARDS
       ====================================================== */

    .agent-card {
        padding: 18px;

        border-radius: 16px;

        background:
            linear-gradient(
                145deg,
                rgba(12, 35, 60, 0.75),
                rgba(6, 17, 31, 0.8)
            );

        border: 1px solid rgba(0, 180, 216, 0.12);

        min-height: 110px;

        transition: 0.25s ease;
    }


    .agent-card:hover {
        transform: translateY(-3px);

        border-color:
            rgba(0, 180, 216, 0.38);

        box-shadow:
            0 12px 30px rgba(0, 180, 216, 0.08);
    }


    .agent-icon {
        font-size: 25px;
    }


    .agent-name {
        font-size: 15px;
        font-weight: 700;
        margin-top: 8px;
    }


    .agent-active {
        color: #5eead4;
        font-size: 12px;
        margin-top: 5px;
    }


    .agent-inactive {
        color: #64748b;
        font-size: 12px;
        margin-top: 5px;
    }


    /* ======================================================
       SECTION HEADINGS
       ====================================================== */

    .section-title {
        font-size: 23px;
        font-weight: 700;

        color: #edf7ff;

        margin-top: 28px;
        margin-bottom: 15px;
    }


    /* ======================================================
       RECOMMENDATION
       ====================================================== */

    .recommendation-card {
        padding: 28px;

        border-radius: 20px;

        background:
            linear-gradient(
                135deg,
                rgba(0, 119, 182, 0.18),
                rgba(6, 24, 43, 0.82)
            );

        border: 1px solid
            rgba(0, 180, 216, 0.25);

        box-shadow:
            0 20px 55px rgba(0, 0, 0, 0.28);
    }


    .recommendation-text {
        font-size: 18px;
        line-height: 1.7;
        color: #dcecff;
    }


    /* ======================================================
       TEXT INPUTS
       ====================================================== */

    textarea,
    input {
        color: #eaf6ff !important;
    }


    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] {
        background-color: rgba(8, 22, 38, 0.85) !important;

        border: 1px solid
            rgba(80, 170, 210, 0.20) !important;

        border-radius: 12px !important;
    }


    /* ======================================================
       BUTTON
       ====================================================== */

    div.stButton > button {

        width: 100%;

        border: none;

        border-radius: 12px;

        padding: 12px 20px;

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

        transition: 0.2s ease;
    }


    div.stButton > button:hover {

        transform: translateY(-2px);

        box-shadow:
            0 12px 32px rgba(0, 180, 216, 0.32);
    }


    /* ======================================================
       METRICS
       ====================================================== */

    div[data-testid="stMetric"] {

        background:
            rgba(10, 28, 48, 0.65);

        padding: 15px;

        border-radius: 15px;

        border: 1px solid
            rgba(120, 190, 230, 0.12);
    }


    div[data-testid="stMetricLabel"] {
        color: #8da9c4 !important;
    }


    div[data-testid="stMetricValue"] {
        color: #eaf7ff !important;
    }


    /* ======================================================
       TABS
       ====================================================== */

    button[data-baseweb="tab"] {
        color: #8faac4 !important;
    }


    button[data-baseweb="tab"][aria-selected="true"] {
        color: #67e8f9 !important;
    }


    /* ======================================================
       EXPANDER
       ====================================================== */

    details {
        background:
            rgba(8, 22, 38, 0.65);

        border: 1px solid
            rgba(100, 180, 220, 0.12);

        border-radius: 12px;
    }


    /* ======================================================
       FOOTER
       ====================================================== */

    .footer {
        text-align: center;

        color: #526b84;

        font-size: 12px;

        margin-top: 50px;

        padding-top: 20px;

        border-top:
            1px solid rgba(100, 180, 220, 0.08);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="status-pill">
            ● AI MARINE INTELLIGENCE SYSTEM
        </div>

        <div class="hero-title">
            ORCA Marine Intelligence
        </div>

        <div class="hero-subtitle">
            Agentic AI platform for intelligent ocean,
            weather, tide, cyclone and ecosystem analysis.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🌊 ORCA")

    st.caption("Marine Intelligence Platform")

    st.divider()

    st.markdown("### 📍 Target Location")

    latitude = st.number_input(
        "Latitude",
        min_value=-90.0,
        max_value=90.0,
        value=19.076000,
        format="%.6f"
    )

    longitude = st.number_input(
        "Longitude",
        min_value=-180.0,
        max_value=180.0,
        value=72.877700,
        format="%.6f"
    )

    st.divider()

    st.markdown("### 🤖 Agent Architecture")

    st.caption(
        "The Planner Agent dynamically decides "
        "which specialist agents are required."
    )

    st.markdown(
        """
        **Available Agents**

        🌤️ Weather  
        🌊 Ocean  
        🌙 Tide  
        🌀 Cyclone  
        🐟 Ecosystem  
        🎣 PFZ  
        """
    )


# ============================================================
# QUESTION
# ============================================================

st.markdown(
    '<div class="section-title">💬 Marine Query</div>',
    unsafe_allow_html=True
)

question = st.text_area(
    "Ask your question",
    placeholder=(
        "Example: What are the sea conditions near Mumbai "
        "today and is it safe for fishing?"
    ),
    height=120,
    label_visibility="collapsed"
)


# ============================================================
# ANALYZE
# ============================================================

if st.button(
    "🌊 Analyze Marine Conditions",
    type="primary"
):

    if not question.strip():

        st.warning(
            "Please enter a marine-related question."
        )

        st.stop()


    # --------------------------------------------------------
    # INITIAL STATE
    # --------------------------------------------------------

    initial_state = {

        "latitude": float(latitude),

        "longitude": float(longitude),

        "user_question": question.strip(),

        "status": "STARTED"
    }


    # --------------------------------------------------------
    # GRAPH EXECUTION
    # --------------------------------------------------------

    try:

        with st.spinner(
            "🤖 Running Marine Intelligence Agents..."
        ):

            result = marine_graph.invoke(
                initial_state
            )


        st.success(
            "Marine intelligence analysis completed."
        )


        # ====================================================
        # PLANNER
        # ====================================================

        plan = result.get("plan")


        if plan:

            if isinstance(plan, str):

                try:
                    plan = json.loads(plan)

                except Exception:
                    pass


            st.markdown(
                '<div class="section-title">🧠 Planner Decision</div>',
                unsafe_allow_html=True
            )


            if isinstance(plan, dict):

                if plan.get("rejected"):

                    st.error(
                        plan.get(
                            "rejection_reason",
                            "Request rejected by planner."
                        )
                    )

                else:

                    agents = plan.get(
                        "required_agents",
                        []
                    )


                    col1, col2, col3 = st.columns(3)


                    with col1:

                        st.metric(
                            "Query Status",
                            "Accepted"
                        )


                    with col2:

                        st.metric(
                            "Agents Selected",
                            len(agents)
                        )


                    with col3:

                        st.metric(
                            "Latitude",
                            f"{latitude:.4f}"
                        )


                    # ------------------------------------------------
                    # AGENT STATUS CARDS
                    # ------------------------------------------------

                    st.markdown(
                        "### Active Intelligence Agents"
                    )


                    available_agents = [

                        (
                            "🌤️",
                            "Weather",
                            "weather"
                        ),

                        (
                            "🌊",
                            "Ocean",
                            "ocean"
                        ),

                        (
                            "🌙",
                            "Tide",
                            "tide"
                        ),

                        (
                            "🌀",
                            "Cyclone",
                            "cyclone"
                        ),

                        (
                            "🐟",
                            "Ecosystem",
                            "ecosystem"
                        ),

                        (
                            "🎣",
                            "PFZ",
                            "pfz"
                        )
                    ]


                    cols = st.columns(6)


                    for col, (icon, name, key) in zip(
                        cols,
                        available_agents
                    ):

                        with col:

                            if key in agents:

                                st.markdown(
                                    f"""
                                    <div class="agent-card">

                                        <div class="agent-icon">
                                            {icon}
                                        </div>

                                        <div class="agent-name">
                                            {name}
                                        </div>

                                        <div class="agent-active">
                                            ● ACTIVE
                                        </div>

                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                            else:

                                st.markdown(
                                    f"""
                                    <div class="agent-card">

                                        <div class="agent-icon">
                                            {icon}
                                        </div>

                                        <div class="agent-name">
                                            {name}
                                        </div>

                                        <div class="agent-inactive">
                                            ○ NOT REQUIRED
                                        </div>

                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )


                    with st.expander(
                        "View Planner JSON"
                    ):

                        st.json(plan)


        # ====================================================
        # RECOMMENDATION
        # ====================================================

        recommendation = result.get(
            "recommendation"
        )


        if recommendation:

            st.markdown(
                '<div class="section-title">🎯 Marine Intelligence Result</div>',
                unsafe_allow_html=True
            )


            if isinstance(
                recommendation,
                dict
            ):

                # ------------------------------------------------
                # METRICS
                #
                # recommendation_node.py's actual output schema is
                # {summary, risk_level, recommendation, key_findings,
                #  safety_advice, agent_findings} — this UI previously
                # read "overall_status"/"confidence"/"key_factors"/
                # "recommended_action"/"limitations", none of which the
                # node ever produces, so most of the result silently
                # never rendered.
                # ------------------------------------------------

                risk_level = recommendation.get(
                    "risk_level",
                    "UNKNOWN"
                )


                risk_colors = {
                    "LOW": "🟢",
                    "MODERATE": "🟡",
                    "HIGH": "🟠",
                    "SEVERE": "🔴",
                }


                st.metric(
                    "Risk Level",
                    f"{risk_colors.get(str(risk_level).upper(), '⚪')} {risk_level}"
                )


                # ------------------------------------------------
                # SUMMARY
                # ------------------------------------------------

                summary = recommendation.get(
                    "summary"
                )


                if summary:

                    st.markdown(
                        f"""
                        <div class="recommendation-card">

                            <div class="card-title">
                                Marine Analysis
                            </div>

                            <div class="recommendation-text">
                                {summary}
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                # ------------------------------------------------
                # RECOMMENDATION
                # ------------------------------------------------

                recommendation_text = recommendation.get(
                    "recommendation"
                )


                if recommendation_text:

                    st.markdown(
                        "### 🚢 Recommendation"
                    )

                    st.info(
                        recommendation_text
                    )


                # ------------------------------------------------
                # KEY FINDINGS
                # ------------------------------------------------

                key_findings = recommendation.get(
                    "key_findings"
                )


                if key_findings:

                    st.markdown(
                        "### ⚠️ Key Findings"
                    )


                    for finding in key_findings:

                        st.markdown(
                            f"""
                            <div class="glass-card">
                                {finding}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )


                # ------------------------------------------------
                # SAFETY ADVICE
                # ------------------------------------------------

                safety_advice = recommendation.get(
                    "safety_advice"
                )


                if safety_advice:

                    st.markdown(
                        "### 🧭 Safety Advice"
                    )

                    for advice in safety_advice:

                        st.warning(
                            advice
                        )


                # ------------------------------------------------
                # AGENT FINDINGS
                # ------------------------------------------------

                findings = recommendation.get(
                    "agent_findings"
                )


                if findings:

                    st.markdown(
                        "### 🤖 Agent Findings"
                    )


                    for agent, finding in findings.items():

                        with st.expander(
                            f"{agent.capitalize()} Agent"
                        ):

                            if isinstance(
                                finding,
                                dict
                            ):

                                st.json(finding)

                            else:

                                st.write(finding)


                # ------------------------------------------------
                # LIMITATIONS
                # ------------------------------------------------

                limitations = recommendation.get(
                    "limitations"
                )


                if limitations:

                    with st.expander(
                        "⚠️ Analysis Limitations"
                    ):

                        for limitation in limitations:

                            st.write(
                                f"• {limitation}"
                            )


        # ====================================================
        # RAW DATA
        # ====================================================

        st.markdown(
            '<div class="section-title">📊 Agent Data</div>',
            unsafe_allow_html=True
        )


        tabs = st.tabs(
            [
                "🌤️ Weather",
                "🌊 Ocean",
                "🌙 Tide",
                "🌀 Cyclone",
                "🐟 Ecosystem",
                "🎣 PFZ"
            ]
        )


        data_keys = [
            "weather_data",
            "ocean_data",
            "tide_data",
            "cyclone_data",
            "ecosystem_data",
            "pfz_data"
        ]


        for tab, key in zip(
            tabs,
            data_keys
        ):

            with tab:

                data = result.get(key)

                if data:

                    st.json(data)

                else:

                    st.info(
                        "This agent was not selected "
                        "for the current query."
                    )


        # ====================================================
        # COMPLETE OUTPUT
        # ====================================================

        with st.expander(
            "🔧 Complete LangGraph Output"
        ):

            st.json(result)


    except Exception as e:

        st.error(
            "❌ Marine analysis failed."
        )

        st.exception(e)


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
    unsafe_allow_html=True
)