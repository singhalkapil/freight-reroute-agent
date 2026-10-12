"""Unit and resilience tests for weather and routing integration tools."""

import httpx
import pytest
import respx
from shapely.geometry import LineString

from freight_agent.models.domain import (
    AlertSeverity,
    Coordinates,
    PortLocation,
    ShipmentManifest,
    ShipmentStatus,
    TransportMode,
)
from freight_agent.tools.routing import (
    IntermodalRouteSolver,
    calculate_haversine_distance_km,
)
from freight_agent.tools.weather import (
    NOAARateLimitError,
    NOAAServiceUnavailableError,
    NOAAWeatherClient,
)


@pytest.fixture
def sample_manifest() -> ShipmentManifest:
    return ShipmentManifest(
        shipment_id="SHP-HOU-DAL-01",
        origin=PortLocation(latitude=29.3013, longitude=-94.7977, port_code="USGAL"),
        destination=Coordinates(latitude=32.7767, longitude=-96.7970),
        cargo_type="Fresh Produce",
        perishability_hours=18.0,
        demurrage_rate_per_day=4800.0,
        sla_delivery_deadline="2026-10-20T12:00:00Z",
        current_status=ShipmentStatus.IN_TRANSIT,
    )


@pytest.fixture
def mock_polygon_feature() -> dict:
    """GeoJSON alert feature over Galveston/Houston corridor."""
    return {
        "id": "https://api.weather.gov/alerts/NWS-ALERT-HOU-01",
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [-95.5, 29.0],
                    [-94.5, 29.0],
                    [-94.5, 30.0],
                    [-95.5, 30.0],
                    [-95.5, 29.0],
                ]
            ],
        },
        "properties": {
            "id": "NWS-ALERT-HOU-01",
            "severity": "Severe",
            "event": "Flash Flood Warning",
            "areaDesc": "Greater Houston Transit Corridor",
            "ends": "2026-10-18T18:00:00Z",
        },
    }


# =========================================================================
# NOAAWeatherClient Tests
# =========================================================================


class TestNOAAWeatherClient:
    """Tests for NOAA NWS client, polygon matching, and defensive recovery."""

    @respx.mock
    def test_fetch_active_alerts_normal_response(self, mock_polygon_feature):
        respx.get("https://api.weather.gov/alerts/active").mock(
            return_value=httpx.Response(
                200,
                json={"features": [mock_polygon_feature]},
            )
        )

        with NOAAWeatherClient() as client:
            alerts = client.fetch_active_alerts()

        assert len(alerts) == 1
        assert alerts[0]["properties"]["id"] == "NWS-ALERT-HOU-01"

    @respx.mock
    def test_custom_user_agent_enforced(self, mock_polygon_feature):
        route = respx.get("https://api.weather.gov/alerts/active").mock(
            return_value=httpx.Response(200, json={"features": [mock_polygon_feature]})
        )

        custom_ua = "my-custom-logistics-bot/2.0"
        with NOAAWeatherClient(user_agent=custom_ua) as client:
            client.fetch_active_alerts()

        assert route.called
        sent_headers = route.calls.last.request.headers
        assert sent_headers["User-Agent"] == custom_ua
        assert "application/geo+json" in sent_headers["Accept"]

    @respx.mock
    def test_check_corridor_disruptions_intersecting(self, mock_polygon_feature):
        respx.get("https://api.weather.gov/alerts/active").mock(
            return_value=httpx.Response(200, json={"features": [mock_polygon_feature]})
        )

        client = NOAAWeatherClient()

        # Corridor cutting directly through polygon (-95.5 to -94.5, 29.0 to 30.0)
        corridor = LineString([(-95.0, 28.5), (-95.0, 30.5)])
        disruptions = client.check_corridor_disruptions(corridor)

        assert len(disruptions) == 1
        assert disruptions[0].alert_id == "NWS-ALERT-HOU-01"
        assert disruptions[0].severity == AlertSeverity.HIGH
        assert disruptions[0].event_type == "Flash Flood Warning"

    @respx.mock
    def test_check_corridor_disruptions_non_intersecting(self, mock_polygon_feature):
        respx.get("https://api.weather.gov/alerts/active").mock(
            return_value=httpx.Response(200, json={"features": [mock_polygon_feature]})
        )

        client = NOAAWeatherClient()

        # Distant corridor far north in Oklahoma/North Texas
        corridor = LineString([(-97.0, 33.0), (-97.0, 35.0)])
        disruptions = client.check_corridor_disruptions(corridor)

        assert len(disruptions) == 0

    @respx.mock
    @pytest.mark.parametrize(
        "empty_geometry",
        [
            None,
            {},
            {"type": "Polygon", "coordinates": []},
            {"type": "Point", "coordinates": [-95.0, 29.0]},
            {"type": "Polygon", "coordinates": [[[-95.0, 29.0], [-94.5, 29.0]]]},  # only 2 points
        ],
    )
    def test_empty_and_malformed_geometries_handled_defensively(self, empty_geometry):
        feature = {
            "id": "NWS-EMPTY-01",
            "type": "Feature",
            "geometry": empty_geometry,
            "properties": {"id": "NWS-EMPTY-01", "severity": "Minor"},
        }
        respx.get("https://api.weather.gov/alerts/active").mock(
            return_value=httpx.Response(200, json={"features": [feature]})
        )

        client = NOAAWeatherClient()
        corridor = LineString([(-95.0, 28.0), (-95.0, 31.0)])
        disruptions = client.check_corridor_disruptions(corridor)

        # Must not raise exception, gracefully skip empty geometries
        assert len(disruptions) == 0

    @respx.mock
    def test_gateway_timeout_503_retry_recovery(self, mock_polygon_feature):
        """Simulate two consecutive 503 Gateway Timeouts followed by a 200 recovery."""
        route = respx.get("https://api.weather.gov/alerts/active")
        route.side_effect = [
            httpx.Response(503, text="Service Unavailable"),
            httpx.Response(503, text="Service Unavailable"),
            httpx.Response(200, json={"features": [mock_polygon_feature]}),
        ]

        client = NOAAWeatherClient(initial_backoff=0.001, max_retries=3)
        alerts = client.fetch_active_alerts()

        assert len(alerts) == 1
        assert route.call_count == 3

    @respx.mock
    def test_gateway_timeout_503_exhaustion_raises_error(self):
        """Persistent 503 should exhaust retries and raise NOAAServiceUnavailableError."""
        respx.get("https://api.weather.gov/alerts/active").mock(
            return_value=httpx.Response(503, text="Gateway Timeout")
        )

        client = NOAAWeatherClient(initial_backoff=0.001, max_retries=2)
        with pytest.raises(NOAAServiceUnavailableError) as exc_info:
            client.fetch_active_alerts()

        assert "HTTP 503" in str(exc_info.value)

    @respx.mock
    def test_rate_limit_429_retry_recovery(self, mock_polygon_feature):
        """Simulate HTTP 429 Rate Limit followed by success."""
        route = respx.get("https://api.weather.gov/alerts/active")
        route.side_effect = [
            httpx.Response(429, headers={"Retry-After": "0"}),
            httpx.Response(200, json={"features": [mock_polygon_feature]}),
        ]

        client = NOAAWeatherClient(initial_backoff=0.001, max_retries=2)
        alerts = client.fetch_active_alerts()

        assert len(alerts) == 1
        assert route.call_count == 2

    @respx.mock
    def test_rate_limit_429_exhaustion_raises_error(self):
        """Persistent HTTP 429 should exhaust retries and raise NOAARateLimitError."""
        respx.get("https://api.weather.gov/alerts/active").mock(
            return_value=httpx.Response(429, headers={"Retry-After": "0"})
        )

        client = NOAAWeatherClient(initial_backoff=0.001, max_retries=1)
        with pytest.raises(NOAARateLimitError):
            client.fetch_active_alerts()


# =========================================================================
# IntermodalRouteSolver Tests
# =========================================================================


class TestIntermodalRouteSolver:
    """Tests for intermodal routing solver, Haversine fallback, perishability, and LRU cache."""

    def test_haversine_distance_computation(self):
        # Same coordinate must have 0 distance
        assert calculate_haversine_distance_km(29.3, -94.7, 29.3, -94.7) == 0.0

        # Known distance: Galveston (29.3013, -94.7977) to Dallas (32.7767, -96.7970)
        dist = calculate_haversine_distance_km(29.3013, -94.7977, 32.7767, -96.7970)
        assert 420.0 < dist < 460.0

    @respx.mock
    def test_overpass_online_query_success(self):
        respx.post("https://overpass-api.de/api/interpreter").mock(
            return_value=httpx.Response(
                200,
                json={"elements": [{"tags": {"ways": "42"}}]},
            )
        )

        solver = IntermodalRouteSolver(enable_online_lookup=True)
        dist, hours, cost = solver.compute_route_metrics(
            29.3013, -94.7977, 32.7767, -96.7970, TransportMode.ROAD
        )

        assert dist > 400.0
        assert hours > 0.0
        assert cost > 0.0

    def test_offline_haversine_fallback(self):
        # Offline mode explicitly disabled Overpass lookup
        solver = IntermodalRouteSolver(enable_online_lookup=False)
        dist, hours, cost = solver.compute_route_metrics(
            29.3013, -94.7977, 32.7767, -96.7970, TransportMode.RAIL
        )

        # Great-circle is ~435 km, RAIL multiplier is 1.35
        expected_min = 420.0 * 1.35
        expected_max = 460.0 * 1.35
        assert expected_min <= dist <= expected_max
        assert hours > 0.0
        assert cost > 0.0

    def test_multimodal_metrics_comparison(self):
        solver = IntermodalRouteSolver(enable_online_lookup=False)

        _, road_hours, _ = solver.compute_route_metrics(
            29.0, -95.0, 32.0, -96.0, TransportMode.ROAD
        )
        _, rail_hours, _ = solver.compute_route_metrics(
            29.0, -95.0, 32.0, -96.0, TransportMode.RAIL
        )
        _, barge_hours, _ = solver.compute_route_metrics(
            29.0, -95.0, 32.0, -96.0, TransportMode.BARGE
        )

        # ROAD (75 km/h) should be faster than RAIL (45 km/h), which is faster than BARGE (15 km/h)
        assert road_hours < rail_hours < barge_hours

    def test_demurrage_penalty_with_perishability(self, sample_manifest):
        solver = IntermodalRouteSolver(enable_online_lookup=False)

        # Perishability limit is 18.0 hours
        # Transit time 15 hours -> no penalty
        penalty_within = solver.calculate_demurrage_penalty(sample_manifest, transit_hours=15.0)
        assert penalty_within == 0.0

        # Transit time 26 hours -> 8 hours spoiled
        penalty_exceeded = solver.calculate_demurrage_penalty(sample_manifest, transit_hours=26.0)
        # Expected: (8 / 24) * (4800 * 3) = 4800.0
        assert penalty_exceeded == 4800.0

    def test_evaluate_all_modes_returns_alternatives(self, sample_manifest):
        solver = IntermodalRouteSolver(enable_online_lookup=False)
        alternatives = solver.evaluate_all_modes(
            sample_manifest, risk_mitigated_desc="Avoids Hurricane Corridor"
        )

        assert len(alternatives) == 3
        modes = {alt.mode for alt in alternatives}
        assert modes == {TransportMode.ROAD, TransportMode.RAIL, TransportMode.BARGE}

        for alt in alternatives:
            assert alt.distance_km > 0.0
            assert alt.estimated_transit_hours > 0.0
            assert alt.additional_cost_usd > 0.0
            assert alt.risk_mitigated == "Avoids Hurricane Corridor"

    def test_in_memory_lru_cache_behavior(self):
        IntermodalRouteSolver.cache_clear()
        solver = IntermodalRouteSolver(enable_online_lookup=False)

        # Initial call -> cache miss
        solver.compute_route_metrics(29.0, -95.0, 32.0, -96.0, TransportMode.ROAD)
        info1 = IntermodalRouteSolver.cache_info()
        assert info1.misses >= 1

        # Identical call -> cache hit
        solver.compute_route_metrics(29.0, -95.0, 32.0, -96.0, TransportMode.ROAD)
        info2 = IntermodalRouteSolver.cache_info()
        assert info2.hits == info1.hits + 1

        # Clear cache
        IntermodalRouteSolver.cache_clear()
        info3 = IntermodalRouteSolver.cache_info()
        assert info3.currsize == 0
