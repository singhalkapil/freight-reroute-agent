"""NOAA National Weather Service (NWS) API client and corridor disruption analyzer.

Features robust defensive recovery:
- Handles HTTP 503 Gateway Timeouts and 5xx transient server errors with exponential backoff.
- Handles HTTP 429 Rate Limits with capped Retry-After intervals and jitter.
- Repairs self-intersecting / invalid GeoJSON polygons using shapely.make_valid.
- Automatically normalizes (lat, lon) and (lon, lat) corridor coordinate ordering for spatial intersection.
"""

import logging
import random
import time
from datetime import UTC, datetime
from typing import Any

import httpx
from shapely import make_valid
from shapely.errors import GEOSException
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Polygon, shape

from freight_agent.models.domain import (
    AlertSeverity,
    Coordinates,
    PortLocation,
    WeatherDisruptionAlert,
)

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


def normalize_to_geojson_lon_lat(
    coord_point: tuple[float, float] | Coordinates | PortLocation,
) -> tuple[float, float]:
    """Normalize a coordinate point into GeoJSON (longitude, latitude) format for GIS spatial testing.

    In North America / Continental US:
    - Latitude is positive (e.g., 24.0° to 72.0° N).
    - Longitude is negative (e.g., -180.0° to -50.0° W).
    """
    if isinstance(coord_point, (Coordinates, PortLocation)):
        return (coord_point.longitude, coord_point.latitude)

    if isinstance(coord_point, (tuple, list)) and len(coord_point) == 2:
        val1, val2 = float(coord_point[0]), float(coord_point[1])
        # If val1 is positive latitude and val2 is negative longitude -> swap to (lon, lat)
        if val1 > 0 and val2 < 0:
            return (val2, val1)
        # If val1 is negative longitude and val2 is positive latitude -> already (lon, lat)
        return (val1, val2)

    raise TypeError(f"Invalid coordinate representation: {coord_point}")


class NOAAWeatherClient:
    """Client for retrieving and analyzing weather disruptions from the National Weather Service."""

    DEFAULT_BASE_URL: str = "https://api.weather.gov"
    DEFAULT_USER_AGENT: str = "freight-reroute-agent/0.1.0 (engineering@freight-agent.internal)"
    DEFAULT_MAX_RETRY_AFTER: float = 60.0

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = 15.0,
        max_retries: int = 3,
        initial_backoff: float = 0.5,
        backoff_factor: float = 2.0,
        max_retry_after: float = DEFAULT_MAX_RETRY_AFTER,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.user_agent = user_agent
        self.timeout = timeout
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.backoff_factor = backoff_factor
        self.max_retry_after = max_retry_after
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
                        raise NOAARateLimitError(
                            "NOAA NWS API rate limit (429) exceeded maximum retries"
                        )
                    retry_after = response.headers.get("Retry-After")
                    raw_sleep = (
                        float(retry_after) if retry_after and retry_after.isdigit() else backoff
                    )
                    # Cap sleep to prevent thread lockup and add small randomized jitter
                    sleep_time = min(raw_sleep, self.max_retry_after) + random.uniform(0.05, 0.20)
                    logger.warning(
                        "HTTP 429 Rate Limit from NOAA API. Backing off for %.2fs", sleep_time
                    )
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
        """Parse a GeoJSON alert feature defensively, repairing invalid geometries with make_valid."""
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
            logger.debug(
                "Skipping alert feature %s: empty geometry type or coordinates", props.get("id")
            )
            return None

        # Build Shapely geometry with exception safeguarding
        try:
            geom = shape(geometry_data)
        except (GEOSException, ValueError, TypeError) as err:
            logger.warning(
                "Failed to construct Shapely geometry for alert %s: %s", props.get("id"), err
            )
            return None

        if geom.is_empty:
            logger.debug("Skipping alert feature %s: geometry is empty", props.get("id"))
            return None

        # Repair invalid geometries (e.g., self-intersecting bow-tie polygons from NWS)
        if not geom.is_valid:
            try:
                geom = make_valid(geom)
            except Exception as err:
                logger.warning(
                    "Failed to repair invalid geometry for alert %s: %s", props.get("id"), err
                )
                return None

        # Resolve polygon or multi-polygon from repaired geometry
        if isinstance(geom, GeometryCollection):
            poly_elements = [g for g in geom.geoms if isinstance(g, (Polygon, MultiPolygon))]
            if not poly_elements:
                return None
            geom = max(poly_elements, key=lambda p: p.area)

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
            logger.debug(
                "Skipping alert %s: geometry is %s (not a polygon)", props.get("id"), geom.geom_type
            )
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
        corridor_linestring: LineString
        | list[tuple[float, float]]
        | list[Coordinates | PortLocation],
        status: str = "actual",
    ) -> list[WeatherDisruptionAlert]:
        """Test geometric intersection between a freight corridor and active NOAA alert polygons.

        Accepts Shapely LineString, list of (lat, lon) / (lon, lat) tuples, or list of Coordinates.
        Automatically normalizes coordinate ordering to GeoJSON (lon, lat) for GIS testing.
        """
        if isinstance(corridor_linestring, LineString):
            raw_pts = list(corridor_linestring.coords)
            normalized_pts = [normalize_to_geojson_lon_lat(p) for p in raw_pts]
            corridor_geom = LineString(normalized_pts)
        elif isinstance(corridor_linestring, list):
            if len(corridor_linestring) < 2:
                raise ValueError("corridor_linestring must contain at least 2 points")
            normalized_pts = [normalize_to_geojson_lon_lat(p) for p in corridor_linestring]
            corridor_geom = LineString(normalized_pts)
        else:
            raise TypeError(
                "corridor_linestring must be a Shapely LineString or list of coordinates"
            )

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
                logger.warning(
                    "Intersection evaluation failed for alert %s: %s", alert_model.alert_id, err
                )
                continue

        return intersecting_alerts
