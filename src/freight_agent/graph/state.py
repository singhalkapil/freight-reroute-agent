"""State definition for freight rerouting agent workflow."""

import operator
from typing import Annotated, TypedDict

from freight_agent.models.domain import (
    IncidentAdvisoryBrief,
    LoadTenderPayload,
    RerouteAlternative,
    RerouteDecision,
    ShipmentManifest,
    WeatherDisruptionAlert,
)


class FreightAgentState(TypedDict, total=False):
    """Execution state tracked across LangGraph nodes for a shipment evaluation."""

    thread_id: str
    shipment: ShipmentManifest
    active_alerts: list[WeatherDisruptionAlert]
    candidate_routes: list[RerouteAlternative]
    decision: RerouteDecision | None
    human_approved: bool | None
    audit_log: Annotated[list[str], operator.add]
    edi_tender_payload: LoadTenderPayload | None
    current_corridor_disrupted: bool
    requires_human_review: bool
    incident_advisory: IncidentAdvisoryBrief | None
