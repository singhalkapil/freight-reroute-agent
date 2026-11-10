"""Unit tests for FMCSA Hours-of-Service and Cold-Chain Constraint Validator."""

from __future__ import annotations

from datetime import UTC, datetime
import pytest

from freight_agent.constraints.fmcsa_validator import (
    FMCSAConstraintValidator,
    ViolationSeverity,
)
from freight_agent.models.domain import (
    Coordinates,
    PortLocation,
    RerouteAlternative,
    ShipmentManifest,
    TransportMode,
)


@pytest.fixture
def base_manifest() -> ShipmentManifest:
    return ShipmentManifest(
        shipment_id="SHP-TEST-001",
        origin=PortLocation(latitude=29.9511, longitude=-90.0715, port_code="USMSY"),
        destination=Coordinates(latitude=41.8781, longitude=-87.6298),
        cargo_type="Standard Dry Freight",
        perishability_hours=None,
        demurrage_rate_per_day=3600.0,
        sla_delivery_deadline=datetime(2026, 11, 15, 12, 0, tzinfo=UTC),
    )


class TestFMCSAConstraintValidator:
    """Test regulatory and physical constraint validation."""

    def test_short_road_trip_is_fully_compliant(self, base_manifest: ShipmentManifest) -> None:
        """Road detour under 8 hours requires no mandatory rest and is 100% compliant."""
        alternative = RerouteAlternative(
            route_id="ALT-ROAD-01",
            mode=TransportMode.ROAD,
            distance_km=450.0,
            estimated_transit_hours=6.5,
            additional_cost_usd=1200.0,
            risk_mitigated="Bypasses corridor disruption",
        )
        res = FMCSAConstraintValidator.validate_reroute(base_manifest, alternative)
        assert res.is_compliant is True
        assert res.compliance_score == 1.0
        assert len(res.violations) == 0
        assert res.mandatory_rest_stops_hours == 0.0

    def test_road_trip_between_8_and_11_hours_requires_break(self, base_manifest: ShipmentManifest) -> None:
        """Road detour between 8 and 11 hours adds 30 min break under 49 CFR § 395.3."""
        alternative = RerouteAlternative(
            route_id="ALT-ROAD-02",
            mode=TransportMode.ROAD,
            distance_km=850.0,
            estimated_transit_hours=10.0,
            additional_cost_usd=2200.0,
            risk_mitigated="Bypasses corridor disruption",
        )
        res = FMCSAConstraintValidator.validate_reroute(base_manifest, alternative)
        assert res.is_compliant is True
        assert res.mandatory_rest_stops_hours == 0.5
        assert res.total_effective_transit_hours == 10.5

    def test_solo_drive_over_11_hours_triggers_team_drivers_warning(self, base_manifest: ShipmentManifest) -> None:
        """Road detour over 11 hours requires team drivers (warning, not hard block)."""
        alternative = RerouteAlternative(
            route_id="ALT-ROAD-03",
            mode=TransportMode.ROAD,
            distance_km=1300.0,
            estimated_transit_hours=16.0,
            additional_cost_usd=3800.0,
            risk_mitigated="Bypasses corridor disruption",
        )
        res = FMCSAConstraintValidator.validate_reroute(base_manifest, alternative)
        assert res.is_compliant is True  # Team drivers can drive up to 22h
        assert res.requires_team_drivers is True
        assert any(v.rule_name == "FMCSA_11_HOUR_DRIVE_EXCEEDED" for v in res.violations)

    def test_team_limit_exceeded_triggers_hard_block(self, base_manifest: ShipmentManifest) -> None:
        """Road detour over 22 hours triggers a HARD_BLOCK."""
        alternative = RerouteAlternative(
            route_id="ALT-ROAD-04",
            mode=TransportMode.ROAD,
            distance_km=2500.0,
            estimated_transit_hours=26.0,
            additional_cost_usd=6000.0,
            risk_mitigated="Bypasses corridor disruption",
        )
        res = FMCSAConstraintValidator.validate_reroute(base_manifest, alternative)
        assert res.is_compliant is False
        assert any(v.severity == ViolationSeverity.HARD_BLOCK for v in res.violations)

    def test_perishable_cold_chain_spoilage_triggers_hard_block(self) -> None:
        """Refrigerated seafood exceeding cold chain buffer triggers HARD_BLOCK."""
        perishable_manifest = ShipmentManifest(
            shipment_id="SHP-FISH-001",
            origin=PortLocation(latitude=29.9511, longitude=-90.0715, port_code="USMSY"),
            destination=Coordinates(latitude=41.8781, longitude=-87.6298),
            cargo_type="Fresh Perishable Seafood (Cold-Chain)",
            perishability_hours=48.0,
            demurrage_rate_per_day=7200.0,
            sla_delivery_deadline=datetime(2026, 11, 12, 12, 0, tzinfo=UTC),
        )
        # Detour takes 52 hours via barge
        alternative = RerouteAlternative(
            route_id="ALT-BARGE-01",
            mode=TransportMode.BARGE,
            distance_km=1500.0,
            estimated_transit_hours=52.0,
            additional_cost_usd=1500.0,
            risk_mitigated="Bypasses corridor disruption",
        )
        res = FMCSAConstraintValidator.validate_reroute(
            perishable_manifest, alternative, max_cold_chain_hours=48.0
        )
        assert res.is_compliant is False
        assert any(v.rule_name == "COLD_CHAIN_SPOILAGE_RISK" for v in res.violations)

    def test_tariff_ceiling_breach_triggers_hard_block(self, base_manifest: ShipmentManifest) -> None:
        """Detour additional cost exceeding tariff ceiling triggers HARD_BLOCK."""
        alternative = RerouteAlternative(
            route_id="ALT-ROAD-EXPENSIVE",
            mode=TransportMode.ROAD,
            distance_km=600.0,
            estimated_transit_hours=8.0,
            additional_cost_usd=9500.0,
            risk_mitigated="Bypasses corridor disruption",
        )
        res = FMCSAConstraintValidator.validate_reroute(
            base_manifest, alternative, tariff_ceiling_usd=7500.0
        )
        assert res.is_compliant is False
        assert any(v.rule_name == "TARIFF_CEILING_BREACH" for v in res.violations)
