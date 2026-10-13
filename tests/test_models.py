"""Unit tests for freight agent domain models and contracts."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from freight_agent.models.domain import (
    AlertSeverity,
    Coordinates,
    PortLocation,
    RerouteAlternative,
    RerouteDecision,
    ShipmentManifest,
    ShipmentStatus,
    ShipmentTrackingState,
    TransportMode,
    WeatherDisruptionAlert,
)


@pytest.fixture
def valid_origin_data() -> dict:
    return {
        "latitude": 29.3013,
        "longitude": -94.7977,
        "port_code": "USGAL",  # Port of Galveston
    }


@pytest.fixture
def valid_destination_data() -> dict:
    return {
        "latitude": 32.7767,
        "longitude": -96.7970,  # Dallas inland depot
    }


@pytest.fixture
def valid_manifest_payload(valid_origin_data, valid_destination_data) -> dict:
    return {
        "shipment_id": "SHP-2026-X99",
        "origin": valid_origin_data,
        "destination": valid_destination_data,
        "cargo_type": "Refrigerated Pharmaceuticals",
        "perishability_hours": 72.0,
        "demurrage_rate_per_day": 3500.0,
        "sla_delivery_deadline": "2026-10-18T18:00:00Z",
        "current_status": "IN_TRANSIT",
    }


@pytest.fixture
def valid_alternative() -> RerouteAlternative:
    return RerouteAlternative(
        route_id="ALT-RAIL-01",
        mode=TransportMode.RAIL,
        distance_km=485.5,
        estimated_transit_hours=14.5,
        additional_cost_usd=1200.0,
        risk_mitigated="Bypasses Gulf Coast tropical storm corridor",
    )


class TestCoordinatesAndPortLocation:
    """Test geographic coordinate and port location validation."""

    def test_valid_coordinates(self):
        coord = Coordinates(latitude=45.0, longitude=-93.0)
        assert coord.latitude == 45.0
        assert coord.longitude == -93.0
        assert coord.as_tuple == (45.0, -93.0)

    @pytest.mark.parametrize(
        "lat,lon",
        [
            (90.1, 0.0),
            (-90.1, 0.0),
            (0.0, 180.1),
            (0.0, -180.1),
            (100.0, 200.0),
        ],
    )
    def test_invalid_coordinates(self, lat: float, lon: float):
        with pytest.raises(ValidationError):
            Coordinates(latitude=lat, longitude=lon)

    def test_valid_port_location(self, valid_origin_data):
        port = PortLocation(**valid_origin_data)
        assert port.port_code == "USGAL"
        assert port.as_tuple == (29.3013, -94.7977)

    @pytest.mark.parametrize(
        "invalid_code",
        [
            "usgal",  # lowercase
            "US",  # too short (< 3)
            "TOOLONGPORT123",  # too long (> 10)
            "US GAL",  # whitespace
            "US@#$",  # special characters
        ],
    )
    def test_invalid_port_code(self, invalid_code: str):
        with pytest.raises(ValidationError):
            PortLocation(latitude=29.3, longitude=-94.7, port_code=invalid_code)


class TestShipmentManifest:
    """Test shipment manifest parsing, validation and immutability."""

    def test_successful_manifest_parsing(self, valid_manifest_payload):
        manifest = ShipmentManifest.model_validate(valid_manifest_payload)

        assert manifest.shipment_id == "SHP-2026-X99"
        assert manifest.origin.port_code == "USGAL"
        assert manifest.destination.latitude == 32.7767
        assert manifest.perishability_hours == 72.0
        assert manifest.demurrage_rate_per_day == 3500.0
        assert manifest.current_status == ShipmentStatus.IN_TRANSIT
        assert manifest.sla_delivery_deadline.tzinfo is not None

    def test_manifest_optional_perishability(self, valid_manifest_payload):
        valid_manifest_payload["perishability_hours"] = None
        manifest = ShipmentManifest.model_validate(valid_manifest_payload)
        assert manifest.perishability_hours is None

    def test_invalid_negative_perishability(self, valid_manifest_payload):
        valid_manifest_payload["perishability_hours"] = -5.0
        with pytest.raises(ValidationError) as exc:
            ShipmentManifest.model_validate(valid_manifest_payload)
        assert "perishability_hours" in str(exc.value)

    def test_invalid_negative_demurrage(self, valid_manifest_payload):
        valid_manifest_payload["demurrage_rate_per_day"] = -100.0
        with pytest.raises(ValidationError) as exc:
            ShipmentManifest.model_validate(valid_manifest_payload)
        assert "demurrage_rate_per_day" in str(exc.value)

    @pytest.mark.parametrize(
        "bad_lat,bad_lon",
        [
            (95.0, -94.0),
            (-95.0, -94.0),
            (29.0, 190.0),
            (29.0, -195.0),
        ],
    )
    def test_invalid_origin_coordinates_in_manifest(self, valid_manifest_payload, bad_lat, bad_lon):
        valid_manifest_payload["origin"]["latitude"] = bad_lat
        valid_manifest_payload["origin"]["longitude"] = bad_lon
        with pytest.raises(ValidationError):
            ShipmentManifest.model_validate(valid_manifest_payload)

    @pytest.mark.parametrize(
        "bad_lat,bad_lon",
        [
            (91.0, 0.0),
            (-91.0, 0.0),
            (0.0, 185.0),
            (0.0, -185.0),
        ],
    )
    def test_invalid_destination_coordinates_in_manifest(
        self, valid_manifest_payload, bad_lat, bad_lon
    ):
        valid_manifest_payload["destination"]["latitude"] = bad_lat
        valid_manifest_payload["destination"]["longitude"] = bad_lon
        with pytest.raises(ValidationError):
            ShipmentManifest.model_validate(valid_manifest_payload)

    def test_deadline_enforces_timezone_awareness(self, valid_manifest_payload):
        # Naive datetime without timezone must fail validation
        valid_manifest_payload["sla_delivery_deadline"] = datetime(2026, 10, 18, 18, 0, 0)
        with pytest.raises(ValidationError) as exc:
            ShipmentManifest.model_validate(valid_manifest_payload)
        assert "sla_delivery_deadline must be timezone-aware" in str(exc.value)

    def test_deadline_invalid_string_format(self, valid_manifest_payload):
        valid_manifest_payload["sla_delivery_deadline"] = "not-a-datetime"
        with pytest.raises(ValidationError):
            ShipmentManifest.model_validate(valid_manifest_payload)

    def test_manifest_immutability(self, valid_manifest_payload):
        manifest = ShipmentManifest.model_validate(valid_manifest_payload)
        with pytest.raises(ValidationError):
            manifest.current_status = ShipmentStatus.DELAYED  # type: ignore

    def test_manifest_roundtrip_json_serialization(self, valid_manifest_payload):
        manifest = ShipmentManifest.model_validate(valid_manifest_payload)
        raw_json = manifest.model_dump_json()

        restored = ShipmentManifest.model_validate_json(raw_json)
        assert restored == manifest
        assert restored.sla_delivery_deadline == manifest.sla_delivery_deadline
        assert restored.sla_delivery_deadline.tzinfo is not None

    @pytest.mark.parametrize("missing_field", ["shipment_id", "origin", "destination"])
    def test_missing_required_fields_in_manifest(self, valid_manifest_payload, missing_field):
        del valid_manifest_payload[missing_field]
        with pytest.raises(ValidationError):
            ShipmentManifest.model_validate(valid_manifest_payload)

    def test_manifest_forbids_extra_fields(self, valid_manifest_payload):
        valid_manifest_payload["extra_field"] = "unexpected"
        with pytest.raises(ValidationError):
            ShipmentManifest.model_validate(valid_manifest_payload)


class TestWeatherDisruptionAlert:
    """Test weather alert validation, polygon checks and immutability."""

    def test_valid_alert(self):
        alert = WeatherDisruptionAlert(
            alert_id="NWS-FL-2026-081",
            severity=AlertSeverity.HIGH,
            event_type="Tropical Storm Surge",
            polygon_coordinates=[
                (29.0, -95.0),
                (30.0, -94.0),
                (29.5, -93.5),
                (29.0, -95.0),
            ],
            effective_until=datetime(2026, 10, 15, 12, 0, tzinfo=UTC),
            affected_corridor="I-10 Gulf Coastal Freight Corridor",
        )
        assert alert.alert_id == "NWS-FL-2026-081"
        assert alert.severity == AlertSeverity.HIGH
        assert len(alert.polygon_coordinates) == 4

    def test_polygon_too_few_vertices(self):
        with pytest.raises(ValidationError) as exc:
            WeatherDisruptionAlert(
                alert_id="NWS-ALERT-01",
                severity=AlertSeverity.MEDIUM,
                event_type="Flash Flood",
                polygon_coordinates=[(29.0, -95.0), (30.0, -94.0)],  # Only 2 points
                effective_until=datetime(2026, 10, 15, 12, 0, tzinfo=UTC),
                affected_corridor="Corridor-A",
            )
        assert "at least 3" in str(exc.value)

    def test_polygon_invalid_coordinates(self):
        with pytest.raises(ValidationError):
            WeatherDisruptionAlert(
                alert_id="NWS-ALERT-02",
                severity=AlertSeverity.CRITICAL,
                event_type="Tornado Warning",
                polygon_coordinates=[
                    (29.0, -95.0),
                    (95.0, -94.0),  # Lat 95.0 invalid
                    (29.5, -93.5),
                ],
                effective_until=datetime(2026, 10, 15, 12, 0, tzinfo=UTC),
                affected_corridor="Corridor-B",
            )

    def test_alert_enforces_effective_until_timezone_awareness(self):
        with pytest.raises(ValidationError) as exc:
            WeatherDisruptionAlert(
                alert_id="NWS-ALERT-NAIVE",
                severity=AlertSeverity.MEDIUM,
                event_type="Flash Flood",
                polygon_coordinates=[(29.0, -95.0), (30.0, -94.0), (29.5, -93.5)],
                effective_until=datetime(2026, 10, 15, 12, 0),  # Naive datetime
                affected_corridor="Corridor-A",
            )
        assert "effective_until must be timezone-aware" in str(exc.value)

    def test_alert_immutability(self):
        alert = WeatherDisruptionAlert(
            alert_id="ALERT-03",
            severity=AlertSeverity.LOW,
            event_type="Dense Fog",
            polygon_coordinates=[(28.0, -95.0), (29.0, -95.0), (28.5, -94.0)],
            effective_until=datetime(2026, 10, 15, 12, 0, tzinfo=UTC),
            affected_corridor="Corridor-C",
        )
        with pytest.raises(ValidationError):
            alert.severity = AlertSeverity.HIGH  # type: ignore


class TestRerouteAlternativeAndDecision:
    """Test reroute alternatives, decisions and approval logic."""

    def test_valid_reroute_alternative(self, valid_alternative):
        assert valid_alternative.route_id == "ALT-RAIL-01"
        assert valid_alternative.mode == TransportMode.RAIL
        assert valid_alternative.distance_km == 485.5
        assert valid_alternative.additional_cost_usd == 1200.0

    @pytest.mark.parametrize("invalid_mode", ["AIR", "DRONE", "UNKNOWN"])
    def test_invalid_transport_mode(self, invalid_mode):
        with pytest.raises(ValidationError):
            RerouteAlternative(
                route_id="ALT-02",
                mode=invalid_mode,  # type: ignore
                distance_km=100.0,
                estimated_transit_hours=5.0,
                additional_cost_usd=500.0,
                risk_mitigated="Bypasses high winds",
            )

    def test_alternative_immutability(self, valid_alternative):
        with pytest.raises(ValidationError):
            valid_alternative.distance_km = 600.0  # type: ignore

    def test_reroute_decision_creation(self, valid_alternative):
        decision = RerouteDecision(
            shipment_id="SHP-2026-X99",
            selected_route=valid_alternative,
            requires_human_approval=True,
            estimated_penalty_saved=7000.0,
            rationale="Rail diversion avoids storm surge bottleneck; saves 2 days of demurrage ($7,000)",
        )
        assert decision.shipment_id == "SHP-2026-X99"
        assert decision.selected_route.mode == TransportMode.RAIL
        assert decision.requires_human_approval is True
        assert decision.estimated_penalty_saved == 7000.0

    def test_decision_immutability(self, valid_alternative):
        decision = RerouteDecision(
            shipment_id="SHP-2026-X99",
            selected_route=valid_alternative,
            requires_human_approval=False,
            estimated_penalty_saved=1500.0,
            rationale="Minor highway detour",
        )
        with pytest.raises(ValidationError):
            decision.requires_human_approval = True  # type: ignore

    def test_negative_additional_cost_rejected(self):
        with pytest.raises(ValidationError) as exc:
            RerouteAlternative(
                route_id="ALT-NEG",
                mode=TransportMode.ROAD,
                distance_km=100.0,
                estimated_transit_hours=2.0,
                additional_cost_usd=-50.0,
                risk_mitigated="None",
            )
        assert "additional_cost_usd" in str(exc.value)


class TestShipmentTrackingState:
    """Test dynamic runtime operational tracking state evolved across LangGraph nodes."""

    def test_valid_tracking_state(self, valid_manifest_payload, valid_destination_data):
        manifest = ShipmentManifest.model_validate(valid_manifest_payload)
        current_coords = Coordinates(**valid_destination_data)

        state = ShipmentTrackingState(
            manifest=manifest,
            current_location=current_coords,
            status=ShipmentStatus.IN_TRANSIT,
            accumulated_delay_hours=3.5,
        )
        assert state.manifest.shipment_id == "SHP-2026-X99"
        assert state.accumulated_delay_hours == 3.5
        assert state.active_reroute is None

    def test_tracking_state_negative_delay_rejected(
        self, valid_manifest_payload, valid_destination_data
    ):
        manifest = ShipmentManifest.model_validate(valid_manifest_payload)
        current_coords = Coordinates(**valid_destination_data)

        with pytest.raises(ValidationError):
            ShipmentTrackingState(
                manifest=manifest,
                current_location=current_coords,
                accumulated_delay_hours=-1.0,
            )

    def test_tracking_state_immutability(self, valid_manifest_payload, valid_destination_data):
        manifest = ShipmentManifest.model_validate(valid_manifest_payload)
        current_coords = Coordinates(**valid_destination_data)

        state = ShipmentTrackingState(
            manifest=manifest,
            current_location=current_coords,
        )
        with pytest.raises(ValidationError):
            state.status = ShipmentStatus.DELAYED  # type: ignore
