"""Tests for the quantitative evaluation suite and disruption benchmark dataset."""

from pathlib import Path

import pytest
from evals.evaluate import (
    ScenarioWeatherClient,
    calculate_metrics_summary,
    load_benchmark_scenarios,
    run_scenario,
)

from freight_agent.models.domain import ShipmentManifest
from freight_agent.tools.routing import IntermodalRouteSolver


@pytest.fixture
def scenarios_file() -> Path:
    return Path("evals/disruption_scenarios.json")


@pytest.fixture
def benchmark_scenarios(scenarios_file: Path) -> list[dict]:
    return load_benchmark_scenarios(scenarios_file)


class TestBenchmarkDataset:
    """Validation test suite for evals/disruption_scenarios.json dataset integrity."""

    def test_scenario_counts_and_categories(self, benchmark_scenarios: list[dict]):
        """Ensure benchmark dataset contains exactly 30 test cases with required distribution."""
        assert len(benchmark_scenarios) == 30

        categories = [s["category"] for s in benchmark_scenarios]
        assert categories.count("CLEAR_WEATHER") == 10
        assert categories.count("SEVERE_DISRUPTION") == 10
        assert categories.count("HIGH_COST_ESCALATION") == 5
        assert categories.count("EDGE_CASE") == 5

    def test_scenario_schemas_and_models(self, benchmark_scenarios: list[dict]):
        """Ensure all 30 scenarios have valid Pydantic ShipmentManifest configurations."""
        seen_ids = set()
        for s in benchmark_scenarios:
            assert "scenario_id" in s
            assert s["scenario_id"] not in seen_ids
            seen_ids.add(s["scenario_id"])

            ship_data = s["shipment"]
            # Validate through Pydantic model
            manifest = ShipmentManifest.model_validate(ship_data)
            assert manifest.shipment_id.startswith("SHP-")

            gt = s["ground_truth"]
            assert isinstance(gt["is_disrupted"], bool)
            assert isinstance(gt["expected_detour"], bool)
            assert isinstance(gt["requires_human_approval"], bool)

    def test_scenario_weather_client_parsing(self, benchmark_scenarios: list[dict]):
        """Verify ScenarioWeatherClient correctly parses GeoJSON features into disruption alerts."""
        disrupted_scenarios = [s for s in benchmark_scenarios if s["ground_truth"]["is_disrupted"]]
        for sc in disrupted_scenarios:
            client = ScenarioWeatherClient(sc["active_alerts"])
            features = client.fetch_active_alerts()
            assert len(features) > 0
            alert_tuple = client.parse_alert_feature(features[0])
            assert alert_tuple is not None
            alert_model, alert_geom = alert_tuple
            assert alert_model.alert_id is not None
            assert alert_geom.is_valid


class TestBenchmarkHarness:
    """Tests for the benchmark evaluation harness, scoring math, and report generation."""

    def test_sample_scenario_execution(self, benchmark_scenarios: list[dict]):
        """Execute a subset of scenarios and verify ScenarioEvaluationRecord attributes."""
        solver = IntermodalRouteSolver(enable_online_lookup=False)

        # 1. Clear weather scenario
        cw_sc = next(s for s in benchmark_scenarios if s["scenario_id"] == "CW-01")
        cw_rec = run_scenario(cw_sc, solver)
        assert cw_rec.detected_disruption is False
        assert cw_rec.halted_for_approval is False
        assert cw_rec.detection_match is True
        assert cw_rec.approval_compliant is True
        assert cw_rec.edi_generated is True

        # 2. Severe disruption scenario
        sd_sc = next(s for s in benchmark_scenarios if s["scenario_id"] == "SD-01")
        sd_rec = run_scenario(sd_sc, solver)
        assert sd_rec.detected_disruption is True
        assert sd_rec.halted_for_approval is False
        assert sd_rec.detection_match is True
        assert sd_rec.edi_generated is True
        assert sd_rec.additional_cost_usd > 0

        # 3. High cost escalation scenario
        hc_sc = next(s for s in benchmark_scenarios if s["scenario_id"] == "HC-01")
        hc_rec = run_scenario(hc_sc, solver)
        assert hc_rec.detected_disruption is True
        assert hc_rec.halted_for_approval is True
        assert hc_rec.detection_match is True
        assert hc_rec.approval_compliant is True
        assert hc_rec.edi_generated is True
        assert hc_rec.additional_cost_usd > 5000.0

    def test_metrics_summary_calculation(self, benchmark_scenarios: list[dict]):
        """Verify mathematical correctness of Precision, Recall, Optimality, and Compliance."""
        solver = IntermodalRouteSolver(enable_online_lookup=False)
        # Run 4 representative scenarios
        test_ids = ["CW-01", "SD-01", "HC-01", "EC-01"]
        records = [
            run_scenario(s, solver) for s in benchmark_scenarios if s["scenario_id"] in test_ids
        ]

        summary = calculate_metrics_summary(records)
        assert summary.total_scenarios == 4
        assert summary.successful_runs == 4
        assert summary.failed_runs == 0
        assert summary.detection_precision_pct == 100.0
        assert summary.detection_recall_pct == 100.0
        assert summary.detection_f1_pct == 100.0
        assert summary.approval_compliance_pct == 100.0
        assert summary.latency_mean_ms > 0

    def test_unclamped_net_savings_calculation(self, benchmark_scenarios: list[dict]):
        """Verify that detour cost exceeding avoided demurrage correctly yields negative net savings."""
        solver = IntermodalRouteSolver(enable_online_lookup=False)
        # Take an existing disrupted scenario and set demurrage below detour cost
        sd_sc = next(s for s in benchmark_scenarios if s["scenario_id"] == "SD-01").copy()
        sd_sc["shipment"] = sd_sc["shipment"].copy()
        sd_sc["shipment"]["demurrage_rate_per_day"] = 100.0

        rec = run_scenario(sd_sc, solver)
        assert rec.detected_disruption is True
        assert rec.blind_demurrage_usd == 200.0
        assert rec.additional_cost_usd > 200.0
        # Net saved must be negative (unclamped)
        expected_deficit = round(200.0 - rec.additional_cost_usd, 2)
        assert rec.net_cost_saved_usd == expected_deficit
        assert rec.net_cost_saved_usd < 0.0
        assert rec.cost_optimality_pct < 0.0
