"""Unit and integration tests for the LangGraph state machine and HITL approval workflow."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from freight_agent.graph.runner import FreightAgentRunner
from freight_agent.graph.workflow import (
    COST_APPROVAL_THRESHOLD_USD,
)
from freight_agent.models.domain import (
    AlertSeverity,
    Coordinates,
    PortLocation,
    RerouteAlternative,
    ShipmentManifest,
    ShipmentStatus,
    TransportMode,
    WeatherDisruptionAlert,
)
from freight_agent.tools.routing import IntermodalRouteSolver
from freight_agent.tools.weather import NOAAWeatherClient


@pytest.fixture
def base_manifest() -> ShipmentManifest:
    """Sample perishable shipment manifest from Galveston to Dallas."""
    return ShipmentManifest(
        shipment_id="SHP-TEST-100",
        origin=PortLocation(latitude=29.3013, longitude=-94.7977, port_code="USGAL"),
        destination=Coordinates(latitude=32.7767, longitude=-96.7970),
        cargo_type="Fresh Gulf Seafood",
        perishability_hours=36.0,
        demurrage_rate_per_day=6000.0,
        sla_delivery_deadline=datetime.now(UTC) + timedelta(hours=48.0),
        current_status=ShipmentStatus.IN_TRANSIT,
    )


@pytest.fixture
def severe_weather_alert() -> WeatherDisruptionAlert:
    return WeatherDisruptionAlert(
        alert_id="NWS-ALERT-HOU-99",
        severity=AlertSeverity.HIGH,
        event_type="Tropical Cyclone Warning",
        polygon_coordinates=[(29.0, -95.0), (30.5, -95.0), (30.5, -94.0), (29.0, -95.0)],
        effective_until=datetime.now(UTC) + timedelta(days=2),
        affected_corridor="I-45 Houston-Dallas Freight Corridor",
    )


class TestFreightAgentWorkflow:
    """Test suite for LangGraph workflow execution, routing decisions, and HITL interrupts."""

    def test_clean_corridor_routes_directly_to_tender(self, base_manifest):
        """When no weather bottleneck is detected, workflow skips detours and dispatches primary tender."""
        mock_weather = MagicMock(spec=NOAAWeatherClient)
        mock_weather.check_corridor_disruptions.return_value = []

        runner = FreightAgentRunner(weather_client=mock_weather)
        final_state = runner.run(base_manifest, thread_id="thread-clean-corridor")

        assert len(final_state["active_alerts"]) == 0
        assert final_state["current_corridor_disrupted"] is False
        assert final_state["candidate_routes"] == []
        assert final_state["decision"] is None
        assert final_state["edi_tender_payload"] is not None
        assert "ST*204*0001~" in final_state["edi_tender_payload"]
        assert "PRIMARY-SCHEDULE" in final_state["edi_tender_payload"]

        # Audit log verifies node execution sequence
        audit = final_state["audit_log"]
        assert any("INSPECT_MANIFEST" in log for log in audit)
        assert any("DETECT_BOTTLENECKS" in log for log in audit)
        assert any("DISPATCH_TENDER" in log for log in audit)
        assert not any("CALCULATE_ALTERNATIVES" in log for log in audit)

    def test_disrupted_corridor_under_threshold_auto_approves(self, base_manifest, severe_weather_alert):
        """Detour with surcharge < $5,000 and within SLA buffer auto-approves without manual interruption."""
        mock_weather = MagicMock(spec=NOAAWeatherClient)
        mock_weather.check_corridor_disruptions.return_value = [severe_weather_alert]

        # Cheap low-cost alternative ($1,250 surcharge, 12h transit)
        cheap_alt = RerouteAlternative(
            route_id="ALT-ROAD-CHEAP",
            mode=TransportMode.ROAD,
            distance_km=480.0,
            estimated_transit_hours=12.0,
            additional_cost_usd=1250.0,
            risk_mitigated="Detours around tropical storm",
        )
        mock_routing = MagicMock(spec=IntermodalRouteSolver)
        mock_routing.evaluate_all_modes.return_value = [cheap_alt]

        runner = FreightAgentRunner(weather_client=mock_weather, route_solver=mock_routing)
        final_state = runner.run(base_manifest, thread_id="thread-auto-approve")

        assert len(final_state["active_alerts"]) == 1
        assert final_state["current_corridor_disrupted"] is True
        assert final_state["requires_human_review"] is False
        assert final_state["decision"] is not None
        assert final_state["decision"].requires_human_approval is False
        assert final_state["decision"].selected_route.route_id == "ALT-ROAD-CHEAP"
        assert final_state["edi_tender_payload"] is not None
        assert "ALT-ROAD-CHEAP" in final_state["edi_tender_payload"]

        audit = final_state["audit_log"]
        assert any("CALCULATE_ALTERNATIVES" in log for log in audit)
        assert not any("HUMAN_REVIEW: Dispatcher" in log for log in audit)

    def test_disrupted_corridor_over_cost_threshold_triggers_interrupt_and_approval(
        self, base_manifest, severe_weather_alert
    ):
        """Detour with surcharge > $5,000 triggers LangGraph interrupt, resumes upon approval."""
        mock_weather = MagicMock(spec=NOAAWeatherClient)
        mock_weather.check_corridor_disruptions.return_value = [severe_weather_alert]

        # Expensive detour exceeding $5,000 threshold ($7,400 surcharge)
        expensive_alt = RerouteAlternative(
            route_id="ALT-RAIL-EXPENSIVE",
            mode=TransportMode.RAIL,
            distance_km=650.0,
            estimated_transit_hours=18.0,
            additional_cost_usd=7400.0,
            risk_mitigated="Bypasses Gulf Coast storm surge",
        )
        mock_routing = MagicMock(spec=IntermodalRouteSolver)
        mock_routing.evaluate_all_modes.return_value = [expensive_alt]

        runner = FreightAgentRunner(weather_client=mock_weather, route_solver=mock_routing)
        thread_id = "thread-hitl-approve"

        # 1. Run until interrupt
        state_at_interrupt = runner.run(base_manifest, thread_id=thread_id)

        # Graph paused at human_review
        assert state_at_interrupt["requires_human_review"] is True
        assert state_at_interrupt["decision"].selected_route.additional_cost_usd > COST_APPROVAL_THRESHOLD_USD

        # Checkpoint snapshot confirms pending interrupt
        snapshot = runner.get_state(thread_id)
        assert len(snapshot.tasks) > 0
        assert snapshot.tasks[0].interrupts is not None

        # 2. Resume with manual human approval
        resumed_state = runner.resume_human_review(
            thread_id=thread_id,
            approved=True,
            comments="Approved by Senior Logistics Manager",
        )

        assert resumed_state["human_approved"] is True
        assert resumed_state["edi_tender_payload"] is not None
        assert "ALT-RAIL-EXPENSIVE" in resumed_state["edi_tender_payload"]

        audit = resumed_state["audit_log"]
        assert any("HUMAN_REVIEW: Dispatcher APPROVED" in log for log in audit)
        assert any("DISPATCH_TENDER: Successfully generated EDI 204" in log for log in audit)

    def test_human_review_rejection_aborts_tender(self, base_manifest, severe_weather_alert):
        """When human dispatcher rejects the detour, tender dispatch is aborted."""
        mock_weather = MagicMock(spec=NOAAWeatherClient)
        mock_weather.check_corridor_disruptions.return_value = [severe_weather_alert]

        expensive_alt = RerouteAlternative(
            route_id="ALT-BARGE-99",
            mode=TransportMode.BARGE,
            distance_km=800.0,
            estimated_transit_hours=30.0,
            additional_cost_usd=8500.0,
            risk_mitigated="Inland river detour",
        )
        mock_routing = MagicMock(spec=IntermodalRouteSolver)
        mock_routing.evaluate_all_modes.return_value = [expensive_alt]

        runner = FreightAgentRunner(weather_client=mock_weather, route_solver=mock_routing)
        thread_id = "thread-hitl-reject"

        # 1. Run to interrupt
        runner.run(base_manifest, thread_id=thread_id)

        # 2. Reject detour
        resumed_state = runner.resume_human_review(
            thread_id=thread_id,
            approved=False,
            comments="Cost too high; client will delay delivery instead.",
        )

        assert resumed_state["human_approved"] is False
        assert resumed_state["edi_tender_payload"] is None

        audit = resumed_state["audit_log"]
        assert any("HUMAN_REVIEW: Dispatcher REJECTED" in log for log in audit)

    def test_streaming_execution_yields_events(self, base_manifest):
        """Verify stream_run yields step updates for each traversed node."""
        mock_weather = MagicMock(spec=NOAAWeatherClient)
        mock_weather.check_corridor_disruptions.return_value = []

        runner = FreightAgentRunner(weather_client=mock_weather)
        events = list(runner.stream_run(base_manifest, thread_id="thread-stream-test"))

        node_names = [e[0] for e in events]
        assert "inspect_manifest" in node_names
        assert "detect_bottlenecks" in node_names
        assert "dispatch_tender" in node_names

    def test_perishability_warning_in_audit_log(self, base_manifest):
        """Tight SLA buffer triggers inspect_manifest warning."""
        tight_manifest = base_manifest.model_copy(
            update={
                "sla_delivery_deadline": datetime.now(UTC) + timedelta(hours=10.0),
                "perishability_hours": 24.0,  # 24h > 10h SLA buffer
            }
        )

        mock_weather = MagicMock(spec=NOAAWeatherClient)
        mock_weather.check_corridor_disruptions.return_value = []

        runner = FreightAgentRunner(weather_client=mock_weather)
        result = runner.run(tight_manifest, thread_id="thread-perish-warning")

        audit = result["audit_log"]
        assert any("WARNING - Perishability shelf life" in log for log in audit)
