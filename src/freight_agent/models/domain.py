"""Domain models and data contracts for freight rerouting agent.

Strictly typed, immutable Pydantic v2 domain models enforcing enterprise logistics invariants.
"""

from datetime import datetime
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TransportMode(StrEnum):
    """Supported intermodal freight transportation modes."""

    ROAD = "ROAD"
    RAIL = "RAIL"
    BARGE = "BARGE"


class AlertSeverity(StrEnum):
    """NOAA weather and environmental disruption alert severity levels."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ShipmentStatus(StrEnum):
    """Lifecycle and operational status of a freight shipment."""

    PLANNED = "PLANNED"
    IN_TRANSIT = "IN_TRANSIT"
    DELAYED = "DELAYED"
    DIVERTED = "DIVERTED"
    REROUTED = "REROUTED"
    DELIVERED = "DELIVERED"
    CANCELLED = "CANCELLED"


# Latitudes must be between -90.0 and 90.0, Longitudes between -180.0 and 180.0
Latitude = Annotated[float, Field(ge=-90.0, le=90.0, description="Latitude in decimal degrees")]
Longitude = Annotated[float, Field(ge=-180.0, le=180.0, description="Longitude in decimal degrees")]


class Coordinates(BaseModel):
    """Geographic coordinate pair (Latitude, Longitude)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    latitude: Latitude
    longitude: Longitude

    @property
    def as_tuple(self) -> tuple[float, float]:
        """Return coordinates as (lat, lon) tuple."""
        return (self.latitude, self.longitude)


class PortLocation(BaseModel):
    """Port or intermodal terminal location combining coordinates with UN/LOCODE port identifier."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    latitude: Latitude
    longitude: Longitude
    port_code: str = Field(
        ...,
        min_length=3,
        max_length=10,
        pattern=r"^[A-Z0-9]{3,10}$",
        description="Port code (e.g., UN/LOCODE such as USLAX, NLRTM, SGSIN)",
    )

    @property
    def as_tuple(self) -> tuple[float, float]:
        """Return coordinates as (lat, lon) tuple."""
        return (self.latitude, self.longitude)


class ShipmentManifest(BaseModel):
    """Immutable manifest representing an active or planned cargo shipment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    shipment_id: str = Field(..., min_length=1, description="Unique enterprise shipment identifier")
    origin: PortLocation = Field(..., description="Origin port facility with coordinates and port code")
    destination: Coordinates = Field(..., description="Destination coordinates")
    cargo_type: str = Field(..., min_length=1, description="Commodity or cargo classification")
    perishability_hours: float | None = Field(
        default=None,
        gt=0,
        description="Maximum cold chain / perishability shelf life in transit hours",
    )
    demurrage_rate_per_day: float = Field(
        ...,
        ge=0.0,
        description="Financial demurrage penalty accrual rate per delayed day in USD",
    )
    sla_delivery_deadline: datetime = Field(
        ...,
        description="Contractual SLA delivery deadline with mandatory timezone/UTC awareness",
    )
    current_status: ShipmentStatus = Field(
        default=ShipmentStatus.IN_TRANSIT,
        description="Current operational status of shipment",
    )

    @field_validator("sla_delivery_deadline")
    @classmethod
    def validate_deadline_has_timezone(cls, value: datetime) -> datetime:
        """Enforce timezone-aware SLA delivery deadlines to avoid ambiguous temporal comparisons."""
        if value.tzinfo is None:
            raise ValueError("sla_delivery_deadline must be timezone-aware (e.g. UTC or explicit tz offset)")
        return value


class WeatherDisruptionAlert(BaseModel):
    """NOAA or regional meteorological disruption polygon alert affecting freight corridors."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    alert_id: str = Field(..., min_length=1, description="Unique alert identifier")
    severity: AlertSeverity = Field(..., description="Disruption severity level")
    event_type: str = Field(..., min_length=1, description="Weather event classification (e.g., Hurricane, Flood)")
    polygon_coordinates: list[tuple[Latitude, Longitude]] = Field(
        ...,
        min_length=3,
        description="Geo polygon perimeter vertices as list of (lat, lon) coordinate tuples",
    )
    effective_until: datetime = Field(
        ...,
        description="Expiration / clearance timestamp of disruption alert",
    )
    affected_corridor: str = Field(..., min_length=1, description="Name or identifier of affected transit corridor")

    @field_validator("polygon_coordinates")
    @classmethod
    def validate_polygon_vertices(cls, coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """Validate each coordinate tuple in the disruption polygon."""
        if len(coords) < 3:
            raise ValueError("Polygon must contain at least 3 coordinate vertices")
        for idx, pt in enumerate(coords):
            if len(pt) != 2:
                raise ValueError(f"Vertex at index {idx} must be a (lat, lon) 2-tuple")
            lat, lon = pt
            if not (-90.0 <= lat <= 90.0):
                raise ValueError(f"Vertex at index {idx} has invalid latitude: {lat} (must be -90.0 to 90.0)")
            if not (-180.0 <= lon <= 180.0):
                raise ValueError(f"Vertex at index {idx} has invalid longitude: {lon} (must be -180.0 to 180.0)")
        return coords


class RerouteAlternative(BaseModel):
    """Evaluated intermodal alternative routing option avoiding corridor disruptions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    route_id: str = Field(..., min_length=1, description="Unique alternative route identifier")
    mode: TransportMode = Field(..., description="Intermodal transit mode (ROAD, RAIL, BARGE)")
    distance_km: float = Field(..., gt=0.0, description="Total transit distance in kilometers")
    estimated_transit_hours: float = Field(..., gt=0.0, description="Estimated total transit duration in hours")
    additional_cost_usd: float = Field(
        ...,
        ge=0.0,
        description="Incremental freight cost in USD compared to primary scheduled route",
    )
    risk_mitigated: str = Field(
        ...,
        min_length=1,
        description="Description or indicator of disruption risks bypassed by this alternative",
    )


class RerouteDecision(BaseModel):
    """Deterministic routing agent decision for shipment rerouting with human-in-the-loop gating."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    shipment_id: str = Field(..., min_length=1, description="Shipment identifier being evaluated")
    selected_route: RerouteAlternative = Field(..., description="Selected optimal reroute alternative")
    requires_human_approval: bool = Field(
        ...,
        description="Flag indicating if cost/delay threshold triggers mandatory human sign-off",
    )
    estimated_penalty_saved: float = Field(
        ...,
        ge=0.0,
        description="Estimated demurrage penalty saved in USD by taking reroute",
    )
    rationale: str = Field(..., min_length=1, description="Audit trail and reasoning behind decision")

