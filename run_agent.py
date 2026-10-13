"""CLI entry point for running the Freight Reroute Agent with dynamic inputs.

Supports interactive prompts, command-line arguments, scenario presets,
and loading shipment manifests directly from JSON files.
"""

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Ensure src/ is on sys.path for direct CLI execution without PYTHONPATH=src
_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from freight_agent.graph.runner import FreightAgentRunner  # noqa: E402
from freight_agent.models.domain import (  # noqa: E402
    Coordinates,
    PortLocation,
    ShipmentManifest,
    ShipmentStatus,
)

PRESETS: dict[str, dict] = {
    "clear": {
        "shipment_id": "SHP-CLEAR-ROUTE",
        "origin": {"port_code": "USSAV", "latitude": 32.0809, "longitude": -81.0912},
        "destination": {"latitude": 33.7490, "longitude": -84.3880},
        "cargo_type": "Automotive Parts",
        "perishability_hours": None,
        "demurrage_rate_per_day": 4000.0,
        "sla_hours": 48.0,
    },
    "hurricane": {
        "shipment_id": "SHP-GULF-HURRICANE",
        "origin": {"port_code": "USGAL", "latitude": 29.3013, "longitude": -94.7977},
        "destination": {"latitude": 32.7767, "longitude": -96.7970},
        "cargo_type": "Industrial Machinery",
        "perishability_hours": None,
        "demurrage_rate_per_day": 6000.0,
        "sla_hours": 36.0,
    },
    "cold-chain": {
        "shipment_id": "SHP-VACCINES-COLD",
        "origin": {"port_code": "USTPA", "latitude": 27.9506, "longitude": -82.4572},
        "destination": {"latitude": 28.5383, "longitude": -81.3792},
        "cargo_type": "Refrigerated Vaccines",
        "perishability_hours": 16.0,
        "demurrage_rate_per_day": 8500.0,
        "sla_hours": 24.0,
    },
    "high-cost": {
        "shipment_id": "SHP-TRANSCON-TURBINE",
        "origin": {"port_code": "USLAX", "latitude": 33.7701, "longitude": -118.1937},
        "destination": {"latitude": 41.8781, "longitude": -87.6298},
        "cargo_type": "Power Generation Turbines",
        "perishability_hours": None,
        "demurrage_rate_per_day": 12000.0,
        "sla_hours": 72.0,
    },
}


def interactive_prompt() -> ShipmentManifest:
    """Prompt the user interactively in the terminal for shipment parameters."""
    print("\n--- Enter Shipment Details (Press Enter to accept defaults) ---")
    shipment_id = input("Shipment ID [SHP-CUSTOM-01]: ").strip() or "SHP-CUSTOM-01"
    port_code = input("Origin Port Code [USGAL]: ").strip() or "USGAL"
    orig_lat = float(input("Origin Latitude [29.3013]: ").strip() or 29.3013)
    orig_lon = float(input("Origin Longitude [-94.7977]: ").strip() or -94.7977)
    dest_lat = float(input("Destination Latitude [32.7767]: ").strip() or 32.7767)
    dest_lon = float(input("Destination Longitude [-96.7970]: ").strip() or -96.7970)
    cargo = input("Cargo Classification [Precision Electronics]: ").strip() or "Precision Electronics"
    perish_in = input("Cold Chain Perishability (hours, or leave blank): ").strip()
    perish = float(perish_in) if perish_in else None
    demurrage = float(input("Demurrage Rate ($/day) [5000.0]: ").strip() or 5000.0)
    sla_h = float(input("SLA Deadline (hours from now) [36.0]: ").strip() or 36.0)

    return ShipmentManifest(
        shipment_id=shipment_id,
        origin=PortLocation(latitude=orig_lat, longitude=orig_lon, port_code=port_code),
        destination=Coordinates(latitude=dest_lat, longitude=dest_lon),
        cargo_type=cargo,
        perishability_hours=perish,
        demurrage_rate_per_day=demurrage,
        sla_delivery_deadline=datetime.now(UTC) + timedelta(hours=sla_h),
        current_status=ShipmentStatus.IN_TRANSIT,
    )


def main() -> None:
    """Main CLI execution flow."""
    parser = argparse.ArgumentParser(
        description="Run Freight Reroute Agent on custom or preset shipments."
    )
    parser.add_argument(
        "-i", "--interactive", action="store_true", help="Prompt interactively for shipment details"
    )
    parser.add_argument(
        "-p", "--preset", choices=list(PRESETS.keys()), help="Run a pre-configured operational scenario"
    )
    parser.add_argument("-f", "--file", type=Path, help="Load shipment manifest from a JSON file")
    parser.add_argument("--shipment-id", default="SHP-CLI-01", help="Custom shipment ID")
    parser.add_argument(
        "--port", default="USGAL", help="Origin port UN/LOCODE (e.g., USGAL, USLAX, USSAV)"
    )
    parser.add_argument(
        "--origin",
        nargs=2,
        type=float,
        metavar=("LAT", "LON"),
        default=[29.3013, -94.7977],
        help="Origin coordinates as latitude and longitude",
    )
    parser.add_argument(
        "--dest",
        nargs=2,
        type=float,
        metavar=("LAT", "LON"),
        default=[32.7767, -96.7970],
        help="Destination coordinates as latitude and longitude",
    )
    parser.add_argument(
        "--cargo", default="Precision Electronics", help="Cargo commodity classification"
    )
    parser.add_argument(
        "--perishability",
        type=float,
        default=None,
        help="Cold-chain perishability shelf life in hours",
    )
    parser.add_argument(
        "--demurrage",
        type=float,
        default=5000.0,
        help="Demurrage penalty rate per day in USD",
    )
    parser.add_argument(
        "--sla-hours",
        type=float,
        default=36.0,
        help="Contractual SLA delivery buffer in hours",
    )

    args = parser.parse_args()

    # 1. Resolve Manifest Source
    if args.interactive:
        manifest = interactive_prompt()
    elif args.file:
        with args.file.open(encoding="utf-8") as f:
            data = json.load(f)
        manifest = ShipmentManifest.model_validate(data)
    elif args.preset:
        p = PRESETS[args.preset]
        manifest = ShipmentManifest(
            shipment_id=p["shipment_id"],
            origin=PortLocation(**p["origin"]),
            destination=Coordinates(**p["destination"]),
            cargo_type=p["cargo_type"],
            perishability_hours=p["perishability_hours"],
            demurrage_rate_per_day=p["demurrage_rate_per_day"],
            sla_delivery_deadline=datetime.now(UTC) + timedelta(hours=p["sla_hours"]),
            current_status=ShipmentStatus.IN_TRANSIT,
        )
    else:
        manifest = ShipmentManifest(
            shipment_id=args.shipment_id,
            origin=PortLocation(
                latitude=args.origin[0], longitude=args.origin[1], port_code=args.port
            ),
            destination=Coordinates(latitude=args.dest[0], longitude=args.dest[1]),
            cargo_type=args.cargo,
            perishability_hours=args.perishability,
            demurrage_rate_per_day=args.demurrage,
            sla_delivery_deadline=datetime.now(UTC) + timedelta(hours=args.sla_hours),
            current_status=ShipmentStatus.IN_TRANSIT,
        )

    # 2. Display Execution Banner
    print("\n" + "=" * 65)
    print(f" Executing Freight Reroute Agent for: {manifest.shipment_id}")
    print(
        f" Route   : {manifest.origin.port_code} ({manifest.origin.latitude}, {manifest.origin.longitude}) "
        f"-> ({manifest.destination.latitude}, {manifest.destination.longitude})"
    )
    print(f" Cargo   : {manifest.cargo_type} | Perishability: {manifest.perishability_hours or 'N/A'}h")
    print(
        f" Demurrage Rate : ${manifest.demurrage_rate_per_day:,.2f}/day | "
        f"SLA Deadline: {manifest.sla_delivery_deadline.strftime('%Y-%m-%d %H:%M UTC')}"
    )
    print("=" * 65 + "\n")

    runner = FreightAgentRunner()
    thread_id = f"cli-{manifest.shipment_id}"

    # 3. Stream real-time node execution
    for node_name, update in runner.stream_run(manifest, thread_id=thread_id):
        print(f"[NODE] {node_name}")
        for log in update.get("audit_log", []):
            print(f"       -> {log}")

    result = runner.run(manifest, thread_id=thread_id)

    # 4. Handle Human Approval Gate if interrupted
    if result.get("requires_human_review"):
        decision = result["decision"]
        print("\n" + "!" * 65)
        print(" [HUMAN REVIEW TRIGGERED]")
        print(f" Detour Reason: {decision.rationale}")
        print(f" Additional Surcharge: ${decision.selected_route.additional_cost_usd:,.2f}")
        print("!" * 65)

        choice = input("\nApprove detour tender? [y/N]: ").strip().lower()
        if choice in ("y", "yes"):
            result = runner.resume_human_review(
                thread_id=thread_id, approved=True, comments="Approved via CLI console"
            )
            print("\n[SUCCESS] Detour approved and dispatched.")
        else:
            result = runner.resume_human_review(
                thread_id=thread_id, approved=False, comments="Rejected via CLI console"
            )
            print("\n[ABORTED] Detour rejected. Tender cancelled.")

    # 5. Display Summary Output
    print("\n" + "=" * 65)
    print(" Execution Completed!")
    print(f" Disruption Detected : {result.get('current_corridor_disrupted')}")
    if result.get("decision"):
        d = result["decision"]
        print(f" Selected Route      : {d.selected_route.route_id} ({d.selected_route.mode.value})")
        print(f" Additional Cost     : ${d.selected_route.additional_cost_usd:,.2f}")
        print(f" Net Demurrage Saved : ${d.estimated_penalty_saved:,.2f}")

    if result.get("edi_tender_payload"):
        print("\n--- Generated ANSI X12 EDI 204 Transaction Payload ---")
        print(result["edi_tender_payload"].edi_raw)
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
