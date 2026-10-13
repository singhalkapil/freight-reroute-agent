"""Unit and integration tests for Google Gemini transition and Agent-Nexus readiness.

Guarantees 100% offline mocked test execution with zero live network calls.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest
from langchain_core.runnables import RunnableLambda

from freight_agent.graph import create_freight_agent_graph, create_freight_graph
from freight_agent.graph.state import FreightAgentState
from freight_agent.graph.workflow import inspect_manifest_node, make_inspect_manifest_node
from freight_agent.models.domain import Coordinates, PortLocation, ShipmentManifest, ShipmentStatus
from freight_agent.parser import get_gemini_chat_model
from freight_agent.parser.manifest_parser import RawExtractedManifest
from freight_agent.tools.weather import NOAAWeatherClient


@pytest.fixture
def base_manifest():
    """Sample baseline shipment manifest fixture for tests."""
    return ShipmentManifest(
        shipment_id="SHP-TEST-BASE",
        origin=PortLocation(latitude=33.74318, longitude=-118.26725, port_code="USLAX"),
        destination=Coordinates(latitude=41.8781, longitude=-87.6298),
        cargo_type="Precision Electronics",
        perishability_hours=72.0,
        demurrage_rate_per_day=3000.0,
        sla_delivery_deadline=datetime.now(UTC) + timedelta(days=4),
        current_status=ShipmentStatus.IN_TRANSIT,
    )


class TestGeminiProviderFactory:
    """Test suite for get_gemini_chat_model key resolution and validation."""

    def test_gemini_api_key_resolution(self, monkeypatch):
        """Resolves GEMINI_API_KEY when present in environment."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key-12345")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        model = get_gemini_chat_model(model="gemini-1.5-flash", temperature=0.2)
        assert model.model == "gemini-1.5-flash"
        assert model.temperature == pytest.approx(0.2)
        assert model.google_api_key.get_secret_value() == "test-gemini-key-12345"

    def test_google_api_key_fallback_resolution(self, monkeypatch):
        """Resolves GOOGLE_API_KEY fallback when GEMINI_API_KEY is not set."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key-67890")

        model = get_gemini_chat_model(model="gemini-1.5-pro", temperature=0.0)
        assert model.model == "gemini-1.5-pro"
        assert model.google_api_key.get_secret_value() == "test-google-key-67890"

    def test_missing_keys_raises_value_error(self, monkeypatch):
        """Raises clean, actionable ValueError when neither API key is configured."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        with pytest.raises(ValueError) as exc_info:
            get_gemini_chat_model()

        msg = str(exc_info.value)
        assert "Google Gemini API key not configured" in msg
        assert "GEMINI_API_KEY" in msg
        assert "https://aistudio.google.com/" in msg


class TestDynamicManifestIngestion:
    """Test suite for dynamic manifest inspection in inspect_manifest_node."""

    def test_inspect_manifest_with_preparsed_shipment(self, base_manifest):
        """Standard path: inspects existing ShipmentManifest in state."""
        state: FreightAgentState = {
            "shipment": base_manifest,
            "thread_id": "test-thread-preparsed",
        }

        updates = inspect_manifest_node(state)
        assert "audit_log" in updates
        assert any("INSPECT_MANIFEST: Validated shipment" in log for log in updates["audit_log"])
        assert "shipment" not in updates  # No need to overwrite existing shipment

    def test_inspect_manifest_dynamic_raw_manifest_text(self):
        """Agent-Nexus path: dynamically parses raw_manifest_text when shipment is None."""
        raw_text = """
        BOL # SHP-NEXUS-001
        From: Port of Los Angeles (USLAX)
        To: Chicago Intermodal Terminal (USCHI)
        Cargo: Temperature-Controlled Pharmaceuticals (Perishability: 36 hours)
        Demurrage Rate: $2,500/day
        SLA Deadline: 2026-11-15T18:00:00Z
        """
        state: FreightAgentState = {
            "thread_id": "test-nexus-raw",
            "raw_manifest_text": raw_text,
        }

        updates = inspect_manifest_node(state)
        assert "shipment" in updates
        shipment = updates["shipment"]
        assert isinstance(shipment, ShipmentManifest)
        assert shipment.shipment_id == "SHP-NEXUS-001"
        assert shipment.origin.port_code == "USLAX"
        assert shipment.destination.latitude == pytest.approx(41.8781, abs=0.01)
        assert shipment.demurrage_rate_per_day == 2500.0
        assert shipment.perishability_hours == 36.0

    def test_inspect_manifest_dynamic_user_prompt(self):
        """Agent-Nexus chat workbench path: parses natural user prompt."""
        prompt_text = "Reroute shipment SHP-PROMPT-99 from USNYC to USCHI carrying Consumer Electronics, demurrage $1,800/day deadline 2026-12-01T00:00:00Z"
        state: FreightAgentState = {
            "thread_id": "test-nexus-prompt",
            "user_prompt": prompt_text,
        }

        updates = inspect_manifest_node(state)
        assert "shipment" in updates
        shipment = updates["shipment"]
        assert isinstance(shipment, ShipmentManifest)
        assert shipment.shipment_id == "SHP-PROMPT-99"
        assert shipment.origin.port_code == "USNYC"
        assert shipment.demurrage_rate_per_day == 1800.0

    def test_inspect_manifest_missing_all_inputs_raises_value_error(self):
        """Raises ValueError when neither shipment, raw_manifest_text, nor user_prompt is present."""
        empty_state: FreightAgentState = {
            "thread_id": "empty-thread",
        }

        with pytest.raises(ValueError) as exc:
            inspect_manifest_node(empty_state)
        assert "State missing 'shipment', 'raw_manifest_text', or 'user_prompt'" in str(exc.value)

    def test_inspect_manifest_with_mocked_llm(self):
        """Custom manifest parser factory correctly uses injected LLM."""
        mock_extracted = RawExtractedManifest(
            shipment_id="SHP-MOCK-LLM-01",
            origin_port_code="USSAV",
            origin_latitude=32.1283,
            origin_longitude=-81.1444,
            destination_latitude=29.7275,
            destination_longitude=-95.2635,
            destination_name_or_code="USHOU",
            cargo_type="Aerospace Components",
            perishability_hours=72.0,
            demurrage_rate_per_day=4000.0,
            sla_delivery_deadline="2026-11-20T12:00:00Z",
            current_status="IN_TRANSIT",
        )
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = RunnableLambda(lambda _: mock_extracted)

        custom_node = make_inspect_manifest_node(manifest_llm=mock_llm)
        state: FreightAgentState = {
            "thread_id": "test-custom-llm-node",
            "raw_manifest_text": "Sample text to be parsed by mock LLM",
        }

        updates = custom_node(state)
        shipment = updates["shipment"]
        assert shipment.shipment_id == "SHP-MOCK-LLM-01"
        assert shipment.origin.port_code == "USSAV"
        assert shipment.cargo_type == "Aerospace Components"


class TestAgentNexusGraphIntegration:
    """Test suite for create_freight_graph alias, compilation, and event streaming."""

    def test_create_freight_graph_alias_identity(self):
        """create_freight_graph is identical alias to create_freight_agent_graph."""
        assert create_freight_graph is create_freight_agent_graph

    def test_create_freight_graph_compilation_with_checkpointer_none(self):
        """create_freight_graph compiles successfully with checkpointer=None (uses default MemorySaver)."""
        graph = create_freight_graph(checkpointer=None)
        assert graph is not None
        # Verify all 6 workflow nodes are registered
        expected_nodes = {
            "inspect_manifest",
            "detect_bottlenecks",
            "calculate_alternatives",
            "generate_advisory",
            "human_review",
            "dispatch_tender",
        }
        assert expected_nodes.issubset(set(graph.nodes.keys()))

    @pytest.mark.asyncio
    async def test_agent_nexus_async_event_streaming(self):
        """Agent-Nexus event streaming compatibility: verifies astream_events('v2') emits graph lifecycle events."""
        mock_weather = MagicMock(spec=NOAAWeatherClient)
        mock_weather.check_corridor_disruptions.return_value = []

        graph = create_freight_graph(weather_client=mock_weather)

        initial_state: FreightAgentState = {
            "thread_id": "stream-nexus-test",
            "raw_manifest_text": "Shipment SHP-STREAM-88 from USLAX to USCHI carrying Medical Devices deadline 2026-12-10T12:00:00Z",
        }

        collected_events: list[dict] = []
        async for event in graph.astream_events(
            initial_state,
            config={"configurable": {"thread_id": "stream-nexus-test"}},
            version="v2",
        ):
            collected_events.append(event)

        assert len(collected_events) > 0
        event_types = {e.get("event") for e in collected_events}
        assert "on_chain_start" in event_types
        assert "on_chain_end" in event_types

        # Verify inspect_manifest ran and emitted output
        node_end_events = [
            e
            for e in collected_events
            if e.get("event") == "on_chain_end" and e.get("name") == "inspect_manifest"
        ]
        assert len(node_end_events) >= 1
        output_data = node_end_events[0].get("data", {}).get("output", {})
        assert "shipment" in output_data
        assert output_data["shipment"].shipment_id == "SHP-STREAM-88"
