# state.py

from typing import TypedDict, Any


class MarineState(TypedDict, total=False):

    # -------------------------
    # User Input
    # -------------------------
    user_question: str

    latitude: float
    longitude: float

    # -------------------------
    # Planner Output
    # -------------------------
    plan: dict

    # -------------------------
    # Specialist Agent Outputs
    # -------------------------
    weather_data: dict
    ocean_data: dict
    tide_data: dict
    cyclone_data: dict
    ecosystem_data: dict
    pfz_data: dict
    gis_data: dict

    # -------------------------
    # Final Recommendation
    # -------------------------
    recommendation: dict

    # -------------------------
    # System Status
    # -------------------------
    status: str
