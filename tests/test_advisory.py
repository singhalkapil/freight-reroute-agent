"""Tests for generate_advisory node and graph integration with mocked LLM."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from freight_agent.graph.advisory_node import (
    SyntheticLLMAdvisoryContent,
    make_generate_advisory_node,
)
from freight_agent.graph.state import FreightAgentState
from freight_agent.graph.workflow import create_freight_agent_graph
from freight_agent.models.domain import (
    AlertSeverity,
    Coordinates,
    IncidentAdvisoryBrief,
    PortLocation,
    RerouteAlternative,
    RerouteDecision,
    ShipmentManifest,
    ShipmentStatus,
    TransportMode,
    WeatherDisruptionAlert,
)


@pytest.fixture
def sample_manifest() -> ShipmentManifest:
    """Fixture providing a standard test manifest."""
    return ShipmentManifest(
        shipment_id="TEST-SHP-ADVISORY-01",
        origin=PortLocation(latitude=29.3108, longitude=-94.7925, port_code="USGAL"),
        destination=Coordinates(latitude=41.8781, longitude=-87.6298),
        cargo_type="Medical Diagnostics",
        perishability_hours=72.0,
        demurrage_rate_per_day=3000.0,
        sla_delivery_deadline=datetime.now(UTC) + timedelta(hours=48),
        current_status=ShipmentStatus.IN_TRANSIT,
    )


@pytest.fixture
def sample_decision() -> RerouteDecision:
    """Fixture providing a deterministic reroute decision."""
    route = RerouteAlternative(
        route_id="RR-RAIL-01",
        mode=TransportMode.RAIL,
        distance_km=1650.0,
        estimated_transit_hours=36.0,
        additional_cost_usd=2800.0,
        risk_mitigated="Bypasses Severe Flood in Louisiana corridor",
    )
    return RerouteDecision(
        shipment_id="TEST-SHP-ADVISORY-01",
        selected_route=route,
        requires_human_approval=False,
        estimated_penalty_saved=6000.0,
        rationale="Rail detour avoids flooding with cost well below threshold.",
    )


@pytest.fixture
def sample_alerts() -> list[WeatherDisruptionAlert]:
    """Fixture providing an active weather alert."""
    return [
        WeatherDisruptionAlert(
            alert_id="NWS-FL-991",
            severity=AlertSeverity.HIGH,
            event_type="Flood Warning",
            polygon_coordinates=[
                (30.0, -93.0),
                (30.0, -91.0),
                (31.0, -91.0),
                (31.0, -93.0),
            ],
            effective_until=datetime.now(UTC) + timedelta(hours=24),
            affected_corridor="I-10 Gulf Coast Freight Corridor",
        )
    ]


def test_advisory_node_deterministic_cost_grounding(
    sample_manifest, sample_decision, sample_alerts
):
    """Verify that advisory node injects exact deterministic numbers and never hallucinates."""
    from langchain_core.runnables import RunnableLambda

    mock_llm = MagicMock()
    # Mock LLM returning stylized narrative
    mock_runnable = RunnableLambda(
        lambda _: SyntheticLLMAdvisoryContent(
            driver_instructions="Take rail ramp transfer at Houston Intermodal Yard to bypass I-10 washouts.",
            weather_hazards="Heavy rain and flash flooding reported on Gulf corridors.",
            recommended_rest_stops=["Houston Rail Logistics Yard"],
            executive_rationale="Detour provides positive ROI by preventing demurrage penalties.",
        )
    )
    mock_llm.with_structured_output.return_value = mock_runnable

    node_fn = make_generate_advisory_node(llm=mock_llm)

    state: FreightAgentState = {
        "shipment": sample_manifest,
        "decision": sample_decision,
        "active_alerts": sample_alerts,
        "requires_human_review": False,
        "audit_log": [],
    }

    result = node_fn(state)

    assert "incident_advisory" in result
    brief: IncidentAdvisoryBrief = result["incident_advisory"]
    assert isinstance(brief, IncidentAdvisoryBrief)

    # STRICT GROUNDING CHECKS: Exactly equal to deterministic decision values
    assert brief.additional_cost_usd == 2800.0
    assert brief.estimated_penalty_saved == 6000.0
    assert brief.estimated_transit_hours == 36.0
    assert brief.requires_human_approval is False
    assert brief.shipment_id == "TEST-SHP-ADVISORY-01"
    assert "Houston Intermodal Yard" in brief.driver_advisory.driver_instructions
    assert (
        "Flash flooding" in brief.driver_advisory.weather_hazards
        or "flash flooding" in brief.driver_advisory.weather_hazards.lower()
    )
    assert "EDI-204 TENDER SUMMARY" in brief.edi_tender_summary


def test_advisory_node_fallback_when_llm_fails(sample_manifest, sample_decision, sample_alerts):
    """Verify advisory node falls back to deterministic narrative when LLM fails."""
    from langchain_core.runnables import RunnableLambda

    def _raise_error(_):
        raise RuntimeError("Mock LLM network timeout")

    mock_llm = MagicMock()
    mock_runnable = RunnableLambda(_raise_error)
    mock_llm.with_structured_output.return_value = mock_runnable

    node_fn = make_generate_advisory_node(llm=mock_llm)

    state: FreightAgentState = {
        "shipment": sample_manifest,
        "decision": sample_decision,
        "active_alerts": sample_alerts,
        "requires_human_review": False,
        "audit_log": [],
    }

    result = node_fn(state)
    brief: IncidentAdvisoryBrief = result["incident_advisory"]

    # In fallback mode, deterministic fields are still strictly grounded
    assert brief.additional_cost_usd == 2800.0
    assert brief.estimated_penalty_saved == 6000.0
    assert "REROUTE INSTRUCTIONS" in brief.driver_advisory.driver_instructions
    assert "Flood Warning" in brief.driver_advisory.driver_instructions


def test_workflow_generates_advisory_in_disrupted_pipeline(sample_manifest, sample_alerts):
    """Verify full LangGraph pipeline executes generate_advisory node when disrupted."""
    from langchain_core.runnables import RunnableLambda

    mock_weather = MagicMock()
    mock_weather.check_corridor_disruptions.return_value = sample_alerts

    mock_llm = MagicMock()
    mock_runnable = RunnableLambda(
        lambda _: SyntheticLLMAdvisoryContent(
            driver_instructions="Proceed safely via alternative rail corridor.",
            weather_hazards="Flood warning active.",
            recommended_rest_stops=["BNSF Gateway"],
            executive_rationale="Reroute is financially optimal.",
        )
    )
    mock_llm.with_structured_output.return_value = mock_runnable

    graph = create_freight_agent_graph(
        weather_client=mock_weather,
        advisory_llm=mock_llm,
    )

    initial_state: FreightAgentState = {
        "shipment": sample_manifest,
        "audit_log": [],
    }

    final_state = graph.invoke(
        initial_state, config={"configurable": {"thread_id": "test-advisory-flow"}}
    )

    assert "incident_advisory" in final_state
    assert final_state["incident_advisory"] is not None
    assert final_state["incident_advisory"].shipment_id == "TEST-SHP-ADVISORY-01"
    # Ensure EDI tender payload was also generated
    assert final_state["edi_tender_payload"] is not None
