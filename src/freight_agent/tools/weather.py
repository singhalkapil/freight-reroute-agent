"""NOAA National Weather Service (NWS) API client and corridor disruption analyzer."""

import logging
from datetime import UTC, datetime
from typing import Any

import httpx
from shapely.geometry import LineString, MultiPolygon, Polygon, shape

from freight_agent.models.domain import AlertSeverity, WeatherDisruptionAlert

logger = logging.getLogger(__name__)

# Severity mapping from NOAA NWS terminology to domain AlertSeverity
NOAA_SEVERITY_MAP: dict[str, AlertSeverity] = {
    "EXTREME": AlertSeverity.CRITICAL,
    "SEVERE": AlertSeverity.HIGH,
    "MODERATE": AlertSeverity.MEDIUM,
    "MINOR": AlertSeverity.LOW,
    "UNKNOWN": AlertSeverity.LOW,
}


class NOAAWeatherClient:
    """Client for retrieving and analyzing weather disruptions from the National Weather Service."""

    DEFAULT_BASE_URL: str = "https://api.weather.gov"
    DEFAULT_USER_AGENT: str = "freight-reroute-agent/0.1.0 (engineering@freight-agent.internal)"

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = 15.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.user_agent = user_agent
        self.timeout = timeout
        self._headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/geo+json, application/json",
        }
        self._client = client or httpx.Client(timeout=self.timeout)

    def close(self) -> None:
        """Close the underlying HTTP client session."""
        self._client.close()

    def __enter__(self) -> "NOAAWeatherClient":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def fetch_active_alerts(self, status: str = "actual") -> list[dict[str, Any]]:
        """Fetch active meteorological alerts from the NWS API."""
        url = f"{self.base_url}/alerts/active"
        params = {"status": status}
        response = self._client.get(url, headers=self._headers, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get("features", [])

    def parse_alert_feature(self, feature: dict[str, Any]) -> tuple[WeatherDisruptionAlert, Polygon | MultiPolygon] | None:
        """Parse a GeoJSON alert feature into a domain model and Shapely geometry."""
        props = feature.get("properties", {})
        geometry_data = feature.get("geometry")

        if not geometry_data:
            return None

        # Build Shapely geometry for spatial intersection
        geom = shape(geometry_data)
        if geom.is_empty:
            return None

        alert_id = str(props.get("id") or feature.get("id") or "UNKNOWN")
        raw_sev = str(props.get("severity", "UNKNOWN")).upper()
        severity = NOAA_SEVERITY_MAP.get(raw_sev, AlertSeverity.MEDIUM)
        event_type = props.get("event") or props.get("headline") or "Meteorological Disruption"
        affected_corridor = props.get("areaDesc") or "Unspecified Corridor"

        # Resolve timestamp with guaranteed timezone awareness
        raw_expiry = props.get("ends") or props.get("expires") or props.get("effective")
        if raw_expiry:
            try:
                effective_until = datetime.fromisoformat(raw_expiry.replace("Z", "+00:00"))
            except ValueError:
                effective_until = datetime.now(UTC)
        else:
            effective_until = datetime.now(UTC)

        if effective_until.tzinfo is None:
            effective_until = effective_until.replace(tzinfo=UTC)

        # Extract polygon coordinates as list of (lat, lon)
        poly_coords: list[tuple[float, float]] = []
        if isinstance(geom, Polygon):
            poly_coords = [(float(lat), float(lon)) for lon, lat in geom.exterior.coords]
        elif isinstance(geom, MultiPolygon):
            # Take the exterior coordinates of the largest polygon component
            largest_poly = max(geom.geoms, key=lambda p: p.area)
            poly_coords = [(float(lat), float(lon)) for lon, lat in largest_poly.exterior.coords]

        if len(poly_coords) < 3:
            return None

        alert = WeatherDisruptionAlert(
            alert_id=alert_id,
            severity=severity,
            event_type=event_type,
            polygon_coordinates=poly_coords,
            effective_until=effective_until,
            affected_corridor=affected_corridor,
        )
        return alert, geom

    def check_corridor_disruptions(
        self,
        corridor_linestring: LineString | list[tuple[float, float]],
        status: str = "actual",
    ) -> list[WeatherDisruptionAlert]:
        """Test geometric intersection between a freight corridor line and active NOAA alert polygons."""
        # Convert corridor coordinates into a Shapely LineString (in lon, lat for GeoJSON consistency)
        if isinstance(corridor_linestring, LineString):
            corridor_geom = corridor_linestring
        elif isinstance(corridor_linestring, list):
            if not corridor_linestring or len(corridor_linestring) < 2:
                raise ValueError("corridor_linestring must contain at least 2 points")
            # If coordinates are provided as (lon, lat) tuples
            corridor_geom = LineString(corridor_linestring)
        else:
            raise TypeError("corridor_linestring must be a Shapely LineString or list of coordinate tuples")

        features = self.fetch_active_alerts(status=status)
        intersecting_alerts: list[WeatherDisruptionAlert] = []

        for feat in features:
            parsed = self.parse_alert_feature(feat)
            if not parsed:
                continue

            alert_model, alert_geom = parsed
            if alert_geom.intersects(corridor_geom):
                intersecting_alerts.append(alert_model)

        return intersecting_alerts
