"""Intermodal routing engine with OpenStreetMap/Overpass integration and offline fallback.

Calculates deterministic multimodal route metrics (ROAD, RAIL, BARGE):
- Transit time and distance using Overpass API or Haversine with empirical intermodal multipliers.
- Fuel and operational surcharges.
- Perishability and contractual SLA deadline demurrage penalties.
- In-memory LRU caching to eliminate redundant network queries.
"""

import logging
import math
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Any

import httpx

from freight_agent.models.domain import (
    Coordinates,
    PortLocation,
    RerouteAlternative,
    ShipmentManifest,
    TransportMode,
)

logger = logging.getLogger(__name__)

# Mean Earth radius in kilometers
EARTH_RADIUS_KM: float = 6371.0

# Multimodal routing parameters: speed, network curvature multiplier, and base cost per km
MODE_PARAMETERS: dict[TransportMode, dict[str, float]] = {
    TransportMode.ROAD: {
        "speed_kmh": 75.0,
        "distance_multiplier": 1.25,
        "cost_per_km": 2.20,
        "fuel_burn_per_km": 0.35,  # Liters of diesel per km
        "transfer_surcharge_usd": 150.0,
        "handling_hours": 1.5,
    },
    TransportMode.RAIL: {
        "speed_kmh": 45.0,
        "distance_multiplier": 1.35,
        "cost_per_km": 1.40,
        "fuel_burn_per_km": 0.18,  # Rail equivalent diesel per ton-km
        "transfer_surcharge_usd": 400.0,
        "handling_hours": 4.0,
    },
    TransportMode.BARGE: {
        "speed_kmh": 15.0,
        "distance_multiplier": 1.60,
        "cost_per_km": 0.85,
        "fuel_burn_per_km": 0.10,  # Waterway fuel burn
        "transfer_surcharge_usd": 650.0,
        "handling_hours": 8.0,
    },
}

DIESEL_PRICE_PER_LITER_USD: float = 1.15


def calculate_haversine_distance_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Compute great-circle distance between two coordinate pairs using the Haversine formula."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c


@lru_cache(maxsize=512)
def _cached_query_overpass(
    overpass_url: str,
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    mode: TransportMode,
    timeout: float = 5.0,
) -> float | None:
    """Cached network query to Overpass API to determine infrastructure availability."""
    if mode == TransportMode.RAIL:
        tag_filter = '["railway"~"rail"]'
    elif mode == TransportMode.BARGE:
        tag_filter = '["waterway"~"canal|river"]'
    else:
        tag_filter = '["highway"~"motorway|trunk|primary"]'

    min_lat = min(origin_lat, dest_lat) - 0.5
    max_lat = max(origin_lat, dest_lat) + 0.5
    min_lon = min(origin_lon, dest_lon) - 0.5
    max_lon = max(origin_lon, dest_lon) + 0.5

    query = (
        f"[out:json][timeout:{int(timeout)}];"
        f"(way{tag_filter}({min_lat},{min_lon},{max_lat},{max_lon}););"
        f"out count;"
    )

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                overpass_url,
                data={"data": query},
                headers={"User-Agent": "freight-reroute-agent/0.1.0"},
            )
            if response.status_code == 200:
                data = response.json()
                count = int(data.get("elements", [{}])[0].get("tags", {}).get("ways", 0))
                if count > 0:
                    direct_dist = calculate_haversine_distance_km(
                        origin_lat, origin_lon, dest_lat, dest_lon
                    )
                    multiplier = MODE_PARAMETERS[mode]["distance_multiplier"]
                    return round(direct_dist * multiplier, 2)
    except Exception as err:
        logger.debug("Overpass API query failed: %s; falling back to Haversine", err)

    return None


@lru_cache(maxsize=512)
def compute_intermodal_metrics_cached(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    mode: TransportMode,
    overpass_url: str | None = None,
    enable_online: bool = True,
) -> tuple[float, float, float]:
    """Compute (distance_km, transit_hours, fuel_cost_usd) for a mode with LRU caching.

    Performs network query only on cache miss; subsequent calls return instantly from memory.
    """
    params = MODE_PARAMETERS[mode]
    distance_km: float | None = None

    if enable_online and overpass_url:
        distance_km = _cached_query_overpass(
            overpass_url, origin_lat, origin_lon, dest_lat, dest_lon, mode
        )

    if distance_km is None:
        direct_dist = calculate_haversine_distance_km(
            origin_lat, origin_lon, dest_lat, dest_lon
        )
        distance_km = round(direct_dist * params["distance_multiplier"], 2)

    travel_hours = distance_km / params["speed_kmh"]
    transit_hours = round(travel_hours + params["handling_hours"], 2)

    fuel_liters = distance_km * params["fuel_burn_per_km"]
    fuel_cost = round(fuel_liters * DIESEL_PRICE_PER_LITER_USD, 2)

    return distance_km, transit_hours, fuel_cost


class IntermodalRouteSolver:
    """Solves and compares intermodal freight rerouting alternatives with Overpass and fallback routing."""

    DEFAULT_OVERPASS_URL: str = "https://overpass-api.de/api/interpreter"

    def __init__(
        self,
        overpass_url: str = DEFAULT_OVERPASS_URL,
        timeout: float = 10.0,
        client: httpx.Client | None = None,
        enable_online_lookup: bool = True,
    ) -> None:
        self.overpass_url = overpass_url
        self.timeout = timeout
        self.enable_online_lookup = enable_online_lookup
        self._client = client or httpx.Client(timeout=self.timeout)

    def close(self) -> None:
        """Close the internal HTTP client."""
        self._client.close()

    def __enter__(self) -> "IntermodalRouteSolver":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def compute_route_metrics(
        self,
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
        mode: TransportMode,
    ) -> tuple[float, float, float]:
        """Compute (distance_km, transit_hours, fuel_cost_usd) for a specific mode with LRU caching."""
        return compute_intermodal_metrics_cached(
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            dest_lat=dest_lat,
            dest_lon=dest_lon,
            mode=mode,
            overpass_url=self.overpass_url,
            enable_online=self.enable_online_lookup,
        )

    @staticmethod
    def cache_info():
        """Retrieve metrics LRU cache performance statistics."""
        return compute_intermodal_metrics_cached.cache_info()

    @staticmethod
    def cache_clear():
        """Clear all entries in the metrics and overpass LRU caches."""
        compute_intermodal_metrics_cached.cache_clear()
        _cached_query_overpass.cache_clear()

    def calculate_demurrage_penalty(
        self,
        manifest: ShipmentManifest,
        transit_hours: float,
        departure_time: datetime | None = None,
    ) -> float:
        """Calculate estimated demurrage penalties based on contractual SLA deadlines and cargo perishability."""
        penalties = 0.0

        # 1. Contractual SLA deadline breach penalty
        if departure_time is None:
            departure_time = datetime.now(UTC)
        elif departure_time.tzinfo is None:
            departure_time = departure_time.replace(tzinfo=UTC)

        estimated_arrival = departure_time + timedelta(hours=transit_hours)
        if estimated_arrival > manifest.sla_delivery_deadline:
            excess_seconds = (estimated_arrival - manifest.sla_delivery_deadline).total_seconds()
            excess_days = excess_seconds / 86400.0
            sla_penalty = excess_days * manifest.demurrage_rate_per_day
            penalties += sla_penalty

        # 2. Perishability spoilage penalty: if transit time exceeds cold chain threshold
        if manifest.perishability_hours is not None:
            if transit_hours > manifest.perishability_hours:
                hours_spoiled = transit_hours - manifest.perishability_hours
                # Surcharge for perished freight (3x standard daily rate for each excess day)
                spoilage_penalty = (hours_spoiled / 24.0) * (manifest.demurrage_rate_per_day * 3.0)
                penalties += spoilage_penalty

        return round(penalties, 2)

    def evaluate_intermodal_alternative(
        self,
        manifest: ShipmentManifest,
        mode: TransportMode,
        risk_mitigated_desc: str = "Bypasses disrupted corridor",
        origin_override: Coordinates | PortLocation | None = None,
        dest_override: Coordinates | None = None,
    ) -> RerouteAlternative:
        """Evaluate and build a RerouteAlternative for a shipment under a specified mode."""
        origin = origin_override or manifest.origin
        dest = dest_override or manifest.destination

        distance_km, transit_hours, fuel_cost = self.compute_route_metrics(
            origin.latitude,
            origin.longitude,
            dest.latitude,
            dest.longitude,
            mode,
        )

        params = MODE_PARAMETERS[mode]
        # Total additional cost = (base distance cost + fuel + intermodal transfer fee)
        base_cost = distance_km * params["cost_per_km"]
        total_additional_cost = round(
            base_cost + fuel_cost + params["transfer_surcharge_usd"], 2
        )

        route_id = f"REROUTE-{mode.value}-{manifest.shipment_id}"

        return RerouteAlternative(
            route_id=route_id,
            mode=mode,
            distance_km=distance_km,
            estimated_transit_hours=transit_hours,
            additional_cost_usd=total_additional_cost,
            risk_mitigated=risk_mitigated_desc,
        )

    def evaluate_all_modes(
        self,
        manifest: ShipmentManifest,
        risk_mitigated_desc: str = "Avoids active weather bottleneck",
    ) -> list[RerouteAlternative]:
        """Evaluate all supported intermodal transportation modes (ROAD, RAIL, BARGE)."""
        alternatives: list[RerouteAlternative] = []
        for mode in [TransportMode.ROAD, TransportMode.RAIL, TransportMode.BARGE]:
            alt = self.evaluate_intermodal_alternative(
                manifest=manifest,
                mode=mode,
                risk_mitigated_desc=risk_mitigated_desc,
            )
            alternatives.append(alt)
        return alternatives
