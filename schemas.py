# schemas.py
#
# Pydantic schemas for validating LLM-produced JSON before the rest of
# the pipeline trusts it. Swaps the current bare `json.loads(...)` +
# manual dict poking in planner_node.py / recommendation_node.py for a
# final `Schema.model_validate(...)` check, which raises a clear
# pydantic.ValidationError on a malformed shape instead of letting a
# wrong-shaped dict silently propagate downstream.
#
# These are deliberately permissive (lots of Optional / default
# factories) because the LLM output is validated AFTER your existing
# enrichment code fills in latitude/longitude/grid_points/etc — the
# goal is to catch genuinely broken shapes, not to reject anything
# that doesn't match a rigid ideal schema.

from typing import Literal, Optional
from pydantic import BaseModel, Field, ConfigDict


AllowedAgent = Literal["weather", "ocean", "tide", "cyclone", "ecosystem", "pfz", "gis"]
RiskLevel = Literal["LOW", "MODERATE", "HIGH", "SEVERE"]


# ---------------------------------------------------------
# Planner
# ---------------------------------------------------------

class GridPoint(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    latitude: float
    longitude: float
    bearing_deg: Optional[float] = None
    distance_km: Optional[float] = None


class Plan(BaseModel):
    model_config = ConfigDict(extra="allow")

    rejected: bool = False
    rejection_reason: Optional[str] = None
    required_agents: list[AllowedAgent] = Field(default_factory=list)
    activity: Optional[str] = None
    date: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    grid_points: list[GridPoint] = Field(default_factory=list)
    error: Optional[str] = None


# ---------------------------------------------------------
# Risk pre-pass (see risk_rules.py)
# ---------------------------------------------------------

class FactorSeverity(BaseModel):
    model_config = ConfigDict(extra="allow")

    severity: str  # "LOW" | "MODERATE" | "HIGH" | "SEVERE" | "insufficient_data"


class RiskSignals(BaseModel):
    model_config = ConfigDict(extra="allow")

    wind: Optional[FactorSeverity] = None
    waves: Optional[FactorSeverity] = None
    cyclone: Optional[FactorSeverity] = None
    overall_rule_based_severity: str = "insufficient_data"


# ---------------------------------------------------------
# Recommendation
# ---------------------------------------------------------

class AgentFindings(BaseModel):
    model_config = ConfigDict(extra="allow")

    weather: Optional[dict] = None
    ocean: Optional[dict] = None
    tide: Optional[dict] = None
    cyclone: Optional[dict] = None
    ecosystem: Optional[dict] = None
    pfz: Optional[dict] = None
    gis: Optional[dict] = None


class Recommendation(BaseModel):
    model_config = ConfigDict(extra="allow")

    summary: str
    risk_level: RiskLevel
    recommendation: str
    key_findings: list[str] = Field(default_factory=list)
    safety_advice: list[str] = Field(default_factory=list)
    agent_findings: AgentFindings = Field(default_factory=AgentFindings)
    error: Optional[str] = None
