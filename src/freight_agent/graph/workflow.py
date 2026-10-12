"""Core LangGraph state machine workflow for freight reroute agent.

Implements sequential inspection, disruption detection, intermodal detour evaluation,
human-in-the-loop approval gating, and EDI 204 dispatch tender generation.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from freight_agent.graph.state import FreightAgentState
from freight_agent.models.domain import (
    RerouteAlternative,
    RerouteDecision,
    ShipmentManifest,
)
from freight_agent.tools.routing import IntermodalRouteSolver
from freight_agent.tools.weather import NOAAWeatherClient

logger = logging.getLogger(__name__)

# Enterprise operational thresholds for human-in-the-loop escalation
COST_APPROVAL_THRESHOLD_USD: float = 5000.0
SLA_BREACH_APPROVAL_THRESHOLD_HOURS: float = 24.0


def inspect_manifest_node(state: FreightAgentState) -> dict[str, Any]:
    """Node 1: Inspects cargo perishability and remaining contractual SLA delivery buffer."""
    shipment: ShipmentManifest = state["shipment"]
    now = datetime.now(UTC)

    # Compute remaining time to SLA delivery deadline
    sla_buffer_seconds = (shipment.sla_delivery_deadline - now).total_seconds()
    sla_buffer_hours = max(0.0, round(sla_buffer_seconds / 3600.0, 2))

    findings = [
        f"INSPECT_MANIFEST: Validated shipment {shipment.shipment_id} "
        f"[Commodity='{shipment.cargo_type}', SLA Buffer={sla_buffer_hours}h, "
        f"Perishability={shipment.perishability_hours or 'N/A'}h, "
        f"DemurrageRate=${shipment.demurrage_rate_per_day:,.2f}/day]"
    ]

    if shipment.perishability_hours and sla_buffer_hours < shipment.perishability_hours:
        findings.append(
            f"INSPECT_MANIFEST: WARNING - Perishability shelf life ({shipment.perishability_hours}h) "
            f"exceeds remaining SLA buffer ({sla_buffer_hours}h)."
        )

    return {
        "audit_log": findings,
    }


def make_detect_bottlenecks_node(weather_client: NOAAWeatherClient | None = None):
    """Create Node 2: Checks active freight corridor against NOAA weather alerts."""
    client = weather_client or NOAAWeatherClient()

    def detect_bottlenecks_node(state: FreightAgentState) -> dict[str, Any]:
        shipment = state["shipment"]
        # Construct corridor line between origin port and inland destination
        corridor_path = [shipment.origin, shipment.destination]

        try:
            alerts = client.check_corridor_disruptions(corridor_path)
        except Exception as err:
            logger.warning("Weather check failed during bottleneck detection: %s", err)
            alerts = []

        is_disrupted = len(alerts) > 0
        if is_disrupted:
            alert_descs = [f"{a.severity} {a.event_type} on '{a.affected_corridor}'" for a in alerts]
            log_entry = (
                f"DETECT_BOTTLENECKS: Detected {len(alerts)} active corridor disruption(s): "
                + "; ".join(alert_descs)
            )
        else:
            log_entry = (
                f"DETECT_BOTTLENECKS: Corridor {shipment.origin.port_code} -> Destination clear. "
                "No active NOAA weather disruptions detected."
            )

        return {
            "active_alerts": alerts,
            "current_corridor_disrupted": is_disrupted,
            "audit_log": [log_entry],
        }

    return detect_bottlenecks_node


def make_calculate_alternatives_node(route_solver: IntermodalRouteSolver | None = None):
    """Create Node 3: Computes intermodal detour alternatives when corridor disruption is present."""
    solver = route_solver or IntermodalRouteSolver(enable_online_lookup=False)

    def calculate_alternatives_node(state: FreightAgentState) -> dict[str, Any]:
        shipment = state["shipment"]
        alerts = state.get("active_alerts", [])

        if not alerts:
            return {
                "candidate_routes": [],
                "decision": None,
                "requires_human_review": False,
                "audit_log": ["CALCULATE_ALTERNATIVES: Skipped detour calculation; corridor is clear."],
            }

        risk_desc = f"Avoids {alerts[0].event_type} disruption ({alerts[0].affected_corridor})"
        alternatives: list[RerouteAlternative] = solver.evaluate_all_modes(
            manifest=shipment,
            risk_mitigated_desc=risk_desc,
        )

        # Select optimal route: filter candidates that meet perishability, then sort by additional cost
        viable_routes = [
            r for r in alternatives
            if (shipment.perishability_hours is None or r.estimated_transit_hours <= shipment.perishability_hours)
        ]
        # Fall back to fastest route if all exceed perishability
        selected_route = (
            min(viable_routes, key=lambda r: r.additional_cost_usd)
            if viable_routes
            else min(alternatives, key=lambda r: r.estimated_transit_hours)
        )

        # Evaluate SLA delay breach for the selected detour
        now = datetime.now(UTC)
        estimated_arrival = now + timedelta(hours=selected_route.estimated_transit_hours)
        sla_delay_seconds = max(0.0, (estimated_arrival - shipment.sla_delivery_deadline).total_seconds())
        sla_delay_hours = sla_delay_seconds / 3600.0

        # Calculate demurrage penalty saved by rerouting (avoiding 2-day corridor shutdown)
        estimated_penalty_saved = round(shipment.demurrage_rate_per_day * 2.0, 2)

        # Conditional gating: Escalates if cost > $5,000 OR SLA breaches by > 24 hours
        cost_exceeded = selected_route.additional_cost_usd > COST_APPROVAL_THRESHOLD_USD
        sla_exceeded = sla_delay_hours > SLA_BREACH_APPROVAL_THRESHOLD_HOURS
        requires_approval = cost_exceeded or sla_exceeded

        reasons = []
        if cost_exceeded:
            reasons.append(f"surcharge ${selected_route.additional_cost_usd:,.2f} exceeds ${COST_APPROVAL_THRESHOLD_USD:,.2f}")
        if sla_exceeded:
            reasons.append(f"estimated delay {sla_delay_hours:.1f}h exceeds {SLA_BREACH_APPROVAL_THRESHOLD_HOURS}h threshold")
        rationale = (
            f"Intermodal reroute via {selected_route.mode.value} avoids {risk_desc}. "
            + (f"Mandatory human review triggered: {'; '.join(reasons)}." if requires_approval else "Approved automatically.")
        )

        decision = RerouteDecision(
            shipment_id=shipment.shipment_id,
            selected_route=selected_route,
            requires_human_approval=requires_approval,
            estimated_penalty_saved=estimated_penalty_saved,
            rationale=rationale,
        )

        log_entry = (
            f"CALCULATE_ALTERNATIVES: Evaluated {len(alternatives)} routes. "
            f"Selected {selected_route.mode.value} (Route ID: {selected_route.route_id}, "
            f"Cost: +${selected_route.additional_cost_usd:,.2f}, "
            f"Transit: {selected_route.estimated_transit_hours:.1f}h, "
            f"HumanApprovalRequired={requires_approval})"
        )

        return {
            "candidate_routes": alternatives,
            "decision": decision,
            "requires_human_review": requires_approval,
            "audit_log": [log_entry],
        }

    return calculate_alternatives_node


def human_review_node(state: FreightAgentState) -> dict[str, Any]:
    """Node 4: Sets a LangGraph interrupt breakpoint awaiting human dispatcher sign-off."""
    decision = state.get("decision")
    if not decision:
        return {
            "human_approved": False,
            "audit_log": ["HUMAN_REVIEW: No decision available to review."],
        }

    # Interrupt execution and yield interrupt payload to dispatcher
    interrupt_payload = {
        "shipment_id": state["shipment"].shipment_id,
        "route_id": decision.selected_route.route_id,
        "mode": decision.selected_route.mode.value,
        "additional_cost_usd": decision.selected_route.additional_cost_usd,
        "rationale": decision.rationale,
        "prompt": "Approval required: detour exceeds financial threshold ($5,000) or SLA buffer (24h).",
    }

    approval_input = interrupt(interrupt_payload)

    # Process resume input from human dispatcher
    if isinstance(approval_input, dict):
        is_approved = bool(approval_input.get("approved", True))
        comments = approval_input.get("comments", "Manual review submitted via dispatcher console")
    elif isinstance(approval_input, bool):
        is_approved = approval_input
        comments = "Direct boolean approval response"
    else:
        is_approved = True
        comments = "Approved"

    status_str = "APPROVED" if is_approved else "REJECTED"
    log_msg = f"HUMAN_REVIEW: Dispatcher {status_str} detour {decision.selected_route.route_id}. Comments: '{comments}'"

    return {
        "human_approved": is_approved,
        "audit_log": [log_msg],
    }


def dispatch_tender_node(state: FreightAgentState) -> dict[str, Any]:
    """Node 5: Generates a synthetic ANSI X12 EDI 204 Motor Carrier Load Tender booking payload."""
    shipment = state["shipment"]
    decision = state.get("decision")
    human_approved = state.get("human_approved")

    # If rerouted and human review rejected -> abort tender
    if decision and human_approved is False:
        return {
            "edi_tender_payload": None,
            "audit_log": ["DISPATCH_TENDER: Aborted EDI 204 booking generation due to dispatcher rejection."],
        }

    assigned_mode = decision.selected_route.mode.value if decision else "MOTOR"
    route_id = decision.selected_route.route_id if decision else "PRIMARY-SCHEDULE"
    timestamp_str = datetime.now(UTC).strftime("%y%m%d*%H%M")
    date_str = datetime.now(UTC).strftime("%Y%m%d")

    # Standard synthetic ANSI X12 EDI 204 envelope
    edi_lines = [
        f"ISA*00*          *00*          *ZZ*FREIGHTAGENT   *ZZ*CARRIERSYS     *{timestamp_str}*U*00401*000000001*0*P*>~",
        f"GS*SM*FREIGHTAGENT*CARRIERSYS*{date_str}*1200*1*X*004010~",
        "ST*204*0001~",
        f"B2**LOGS*{shipment.shipment_id}**PP~",
        "B2A*00*LT~",  # Load Tender
        f"MS3*{assigned_mode}*B~",
        f"N1*SH*ORIGIN TERMINAL*10*{shipment.origin.port_code}~",
        f"N3*LAT {shipment.origin.latitude:.4f} LON {shipment.origin.longitude:.4f}~",
        "N1*CN*DESTINATION FACILITY~",
        f"N3*LAT {shipment.destination.latitude:.4f} LON {shipment.destination.longitude:.4f}~",
        f"OID*1*{shipment.shipment_id}***CTN*1*L*42000~",
        f"G62*10*{shipment.sla_delivery_deadline.strftime('%Y%m%d')}~",
        f"NTE*GEN*REROUTE ASSIGNMENT: {route_id}~",
        "SE*13*0001~",
        "GE*1*1~",
        "IEA*1*000000001~",
    ]
    edi_payload = "\n".join(edi_lines)

    log_entry = (
        f"DISPATCH_TENDER: Successfully generated EDI 204 booking tender "
        f"[Mode={assigned_mode}, Route={route_id}, Segments=16]"
    )

    return {
        "edi_tender_payload": edi_payload,
        "audit_log": [log_entry],
    }


def route_after_detection(state: FreightAgentState) -> str:
    """Conditional router: branches to calculate_alternatives if disrupted, else dispatch_tender."""
    if state.get("current_corridor_disrupted", False):
        return "calculate_alternatives"
    return "dispatch_tender"


def route_after_alternatives(state: FreightAgentState) -> str:
    """Conditional router: branches to human_review if cost or delay thresholds are breached."""
    if state.get("requires_human_review", False):
        return "human_review"
    return "dispatch_tender"


def route_after_human_review(state: FreightAgentState) -> str:
    """Conditional router: proceeds to tender if approved, ends if rejected."""
    if state.get("human_approved", False):
        return "dispatch_tender"
    return END


def create_freight_agent_graph(
    weather_client: NOAAWeatherClient | None = None,
    route_solver: IntermodalRouteSolver | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> StateGraph:
    """Construct the compiled LangGraph workflow state machine."""
    workflow = StateGraph(FreightAgentState)

    # 1. Add Workflow Nodes
    workflow.add_node("inspect_manifest", inspect_manifest_node)
    workflow.add_node("detect_bottlenecks", make_detect_bottlenecks_node(weather_client))
    workflow.add_node("calculate_alternatives", make_calculate_alternatives_node(route_solver))
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("dispatch_tender", dispatch_tender_node)

    # 2. Add Workflow Edges
    workflow.add_edge(START, "inspect_manifest")
    workflow.add_edge("inspect_manifest", "detect_bottlenecks")

    workflow.add_conditional_edges(
        "detect_bottlenecks",
        route_after_detection,
        {
            "calculate_alternatives": "calculate_alternatives",
            "dispatch_tender": "dispatch_tender",
        },
    )

    workflow.add_conditional_edges(
        "calculate_alternatives",
        route_after_alternatives,
        {
            "human_review": "human_review",
            "dispatch_tender": "dispatch_tender",
        },
    )

    workflow.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {
            "dispatch_tender": "dispatch_tender",
            END: END,
        },
    )

    workflow.add_edge("dispatch_tender", END)

    # 3. Compile with Checkpointer (default MemorySaver for interrupt/resume recovery)
    active_checkpointer = checkpointer if checkpointer is not None else MemorySaver()
    return workflow.compile(checkpointer=active_checkpointer)
