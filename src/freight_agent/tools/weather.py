"""NOAA National Weather Service (NWS) API client and corridor disruption analyzer.

Features robust defensive recovery:
- Handles HTTP 503 Gateway Timeouts and 5xx transient server errors with exponential backoff.
- Handles HTTP 429 Rate Limits respecting Retry-After headers.
- Handles null, empty, malformed, and non-polygon GeoJSON geometries gracefully.
- Matches corridor linestrings with alert polygons using Shapely geometric intersection.
"""

import logging
import time
from datetime import UTC, datetime
from typing import Any

import httpx
from shapely.errors import GEOSException
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


class NOAAAPIError(Exception):
    """Base exception for NOAA NWS API communication failures."""


class NOAAServiceUnavailableError(NOAAAPIError):
    """Raised when NOAA NWS API returns persistent 503 Gateway Timeout or 5xx errors."""


class NOAARateLimitError(NOAAAPIError):
    """Raised when NOAA NWS API rate limit is exceeded after retries."""


class NOAAWeatherClient:
    """Client for retrieving and analyzing weather disruptions from the National Weather Service."""

    DEFAULT_BASE_URL: str = "https://api.weather.gov"
    DEFAULT_USER_AGENT: str = "freight-reroute-agent/0.1.0 (engineering@freight-agent.internal)"

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = 15.0,
        max_retries: int = 3,
        initial_backoff: float = 0.5,
        backoff_factor: float = 2.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.user_agent = user_agent
        self.timeout = timeout
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.backoff_factor = backoff_factor
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

    def _execute_request_with_retry(
        self,
        url: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute HTTP GET request with exponential backoff for 503, 5xx, 429, and network errors."""
        retries = 0
        backoff = self.initial_backoff

        while True:
            try:
                response = self._client.get(url, headers=self._headers, params=params)

                # Handle 429 Rate Limits
                if response.status_code == 429:
                    if retries >= self.max_retries:
                        raise NOAARateLimitError("NOAA NWS API rate limit (429) exceeded maximum retries")
                    retry_after = response.headers.get("Retry-After")
                    sleep_time = float(retry_after) if retry_after and retry_after.isdigit() else backoff
                    logger.warning("HTTP 429 Rate Limit from NOAA API. Backing off for %.2fs", sleep_time)
                    time.sleep(sleep_time)
                    retries += 1
                    backoff *= self.backoff_factor
                    continue

                # Handle 503 Gateway Timeout and 5xx Server Errors
                if response.status_code in (500, 502, 503, 504):
                    if retries >= self.max_retries:
                        raise NOAAServiceUnavailableError(
                            f"NOAA NWS API error (HTTP {response.status_code}) persisted after {retries} retries: {response.text[:200]}"
                        )
                    logger.warning(
                        "HTTP %d from NOAA API (attempt %d/%d). Retrying in %.2fs...",
                        response.status_code,
                        retries + 1,
                        self.max_retries,
                        backoff,
                    )
                    time.sleep(backoff)
                    retries += 1
                    backoff *= self.backoff_factor
                    continue

                response.raise_for_status()
                return response.json()

            except (httpx.TransportError, httpx.TimeoutException) as err:
                if retries >= self.max_retries:
                    raise NOAAServiceUnavailableError(
                        f"Network transport error connecting to NOAA API after {retries} retries: {err}"
                    ) from err
                logger.warning(
                    "Network error (%s) contacting NOAA API (attempt %d/%d). Retrying in %.2fs...",
                    err.__class__.__name__,
                    retries + 1,
                    self.max_retries,
                    backoff,
                )
                time.sleep(backoff)
                retries += 1
                backoff *= self.backoff_factor

    def fetch_active_alerts(self, status: str = "actual") -> list[dict[str, Any]]:
        """Fetch active meteorological alerts from the NWS API with defensive retry handling."""
        url = f"{self.base_url}/alerts/active"
        params = {"status": status}
        data = self._execute_request_with_retry(url, params=params)
        return data.get("features", [])

    def parse_alert_feature(
        self,
        feature: dict[str, Any],
    ) -> tuple[WeatherDisruptionAlert, Polygon | MultiPolygon] | None:
        """Parse a GeoJSON alert feature defensively, discarding empty or malformed geometries."""
        if not isinstance(feature, dict):
            return None

        props = feature.get("properties") or {}
        geometry_data = feature.get("geometry")

        # Defensive recovery: Handle missing, null, or empty geometry dictionaries
        if not geometry_data or not isinstance(geometry_data, dict):
            logger.debug("Skipping alert feature %s: geometry is null or invalid", props.get("id"))
            return None

        geom_type = geometry_data.get("type")
        raw_coords = geometry_data.get("coordinates")
        if not geom_type or not raw_coords:
            logger.debug("Skipping alert feature %s: empty geometry type or coordinates", props.get("id"))
            return None

        # Build Shapely geometry with exception safeguarding
        try:
            geom = shape(geometry_data)
        except (GEOSException, ValueError, TypeError) as err:
            logger.warning("Failed to construct Shapely geometry for alert %s: %s", props.get("id"), err)
            return None

        if geom.is_empty or not geom.is_valid:
            logger.debug("Skipping alert feature %s: geometry is empty or invalid", props.get("id"))
            return None

        # Extract polygon coordinates as list of (lat, lon)
        poly_coords: list[tuple[float, float]] = []
        if isinstance(geom, Polygon):
            poly_coords = [(float(lat), float(lon)) for lon, lat in geom.exterior.coords]
        elif isinstance(geom, MultiPolygon):
            if not geom.geoms:
                return None
            largest_poly = max(geom.geoms, key=lambda p: p.area)
            poly_coords = [(float(lat), float(lon)) for lon, lat in largest_poly.exterior.coords]
        else:
            logger.debug("Skipping alert %s: geometry is %s (not a polygon)", props.get("id"), geom.geom_type)
            return None

        if len(poly_coords) < 3:
            logger.debug("Skipping alert %s: polygon has fewer than 3 vertices", props.get("id"))
            return None

        alert_id = str(props.get("id") or feature.get("id") or f"ALERT-{int(time.time())}")
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

        try:
            alert = WeatherDisruptionAlert(
                alert_id=alert_id,
                severity=severity,
                event_type=event_type,
                polygon_coordinates=poly_coords,
                effective_until=effective_until,
                affected_corridor=affected_corridor,
            )
        except (ValueError, TypeError) as val_err:
            logger.warning("Alert model validation failed for %s: %s", alert_id, val_err)
            return None

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
            try:
                if alert_geom.intersects(corridor_geom):
                    intersecting_alerts.append(alert_model)
            except GEOSException as err:
                logger.warning("Intersection evaluation failed for alert %s: %s", alert_model.alert_id, err)
                continue

        return intersecting_alerts
