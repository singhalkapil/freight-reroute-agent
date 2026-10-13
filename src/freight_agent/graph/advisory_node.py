"""LangGraph advisory node synthesizing executive incident briefs and driver advisories.

Ensures strict mathematical and financial grounding:
Demurrage savings, detour surcharges, transit hours, and SLA breach buffers are injected
directly from deterministic state calculations into the prompt and output models,
preventing LLM hallucination of arithmetic values.
"""

import logging
from datetime import UTC, datetime
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from freight_agent.graph.state import FreightAgentState
from freight_agent.models.domain import DriverAdvisory, IncidentAdvisoryBrief

logger = logging.getLogger(__name__)


class SyntheticLLMAdvisoryContent(BaseModel):
    """LLM generated narrative content (strictly qualitative; numbers are injected externally)."""

    driver_instructions: str = Field(
        description="Clear, actionable instructions for the freight driver describing route and corridor cautions."
    )
    weather_hazards: str = Field(
        description="Summary of weather hazards and NOAA conditions to watch for on transit path."
    )
    recommended_rest_stops: list[str] = Field(
        default_factory=list,
        description="Suggested staging areas, weigh stations, or safe rest havens.",
    )
    executive_rationale: str = Field(
        description="Concise executive narrative explaining why detour was chosen over waiting out disruption."
    )


ADVISORY_SYSTEM_PROMPT = """You are a senior intermodal freight operations dispatcher and safety officer.
Synthesize a structured incident brief and driver advisory for an active freight reroute.

CRITICAL GROUNDING RULES:
1. Do NOT calculate or alter any financial figures, transit hours, or demurrage penalties. Use the exact values provided in the operational context.
2. Focus your output on clear, professional driver routing instructions, NOAA weather hazard warnings, and executive rationale explaining the decision.
3. Be concise, actionable, and safety-focused.
"""

ADVISORY_HUMAN_PROMPT = """OPERATIONAL CONTEXT:
Shipment ID: {shipment_id}
Cargo Type: {cargo_type}
Origin Port: {origin_port}
Destination: {destination}
Assigned Route ID: {route_id}
Assigned Transit Mode: {mode}
Detour Distance: {distance_km} km
Deterministic Detour Cost: ${cost_usd:,.2f} USD
Deterministic Demurrage Saved: ${penalty_saved:,.2f} USD
Deterministic Estimated Transit Hours: {transit_hours:.1f} hours
Human Approval Required: {requires_human_approval}
Active Weather Disruption(s): {weather_disruptions}
Rationale Context: {rationale}

Generate the driver routing instructions, weather hazard precautions, recommended rest stops, and executive narrative for this incident.
"""


def make_generate_advisory_node(llm: BaseChatModel | Any | None = None):
    """Create a LangGraph node function for synthesizing executive briefs and driver advisories.

    Args:
        llm: Optional LangChain chat model instance supporting .with_structured_output().
             If None or if LLM invocation fails, a robust deterministic fallback is generated.
    """
    structured_llm = (
        llm.with_structured_output(SyntheticLLMAdvisoryContent) if llm is not None else None
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ADVISORY_SYSTEM_PROMPT),
            ("human", ADVISORY_HUMAN_PROMPT),
        ]
    )

    def generate_advisory_node(state: FreightAgentState) -> dict[str, Any]:
        """Node: Generates IncidentAdvisoryBrief with strict deterministic grounding."""
        shipment = state["shipment"]
        decision = state.get("decision")
        alerts = state.get("active_alerts", [])

        # Extract deterministic ground-truth values
        if decision:
            route = decision.selected_route
            route_id = route.route_id
            mode_val = route.mode.value
            distance_km = route.distance_km
            cost_usd = route.additional_cost_usd
            transit_hours = route.estimated_transit_hours
            penalty_saved = decision.estimated_penalty_saved
            requires_human = decision.requires_human_approval
            rationale_text = decision.rationale
        else:
            route_id = "PRIMARY-DIRECT"
            mode_val = "ROAD"
            distance_km = 0.0
            cost_usd = 0.0
            transit_hours = 0.0
            penalty_saved = 0.0
            requires_human = False
            rationale_text = "Standard primary transit schedule without active detour."

        weather_summary = (
            "; ".join(f"{a.severity} {a.event_type} on {a.affected_corridor}" for a in alerts)
            if alerts
            else "Corridor clear. No active severe NOAA weather alerts."
        )

        # Attempt structured LLM synthesis for narrative components
        llm_content: SyntheticLLMAdvisoryContent | None = None
        if structured_llm is not None:
            try:
                chain = prompt | structured_llm
                llm_content = chain.invoke(
                    {
                        "shipment_id": shipment.shipment_id,
                        "cargo_type": shipment.cargo_type,
                        "origin_port": shipment.origin.port_code,
                        "destination": f"({shipment.destination.latitude:.4f}, {shipment.destination.longitude:.4f})",
                        "route_id": route_id,
                        "mode": mode_val,
                        "distance_km": distance_km,
                        "cost_usd": cost_usd,
                        "penalty_saved": penalty_saved,
                        "transit_hours": transit_hours,
                        "requires_human_approval": requires_human,
                        "weather_disruptions": weather_summary,
                        "rationale": rationale_text,
                    }
                )
            except Exception as err:
                logger.warning(
                    "LLM advisory generation failed; falling back to deterministic template: %s",
                    err,
                )
                llm_content = None

        # Deterministic fallback narrative if LLM is omitted or fails
        if llm_content is None:
            if alerts:
                driver_text = (
                    f"REROUTE INSTRUCTIONS: Divert via {mode_val} route {route_id} to bypass "
                    f"{alerts[0].event_type} disruption along {alerts[0].affected_corridor}. "
                    f"Maintain safe spacing and monitor dispatch radio channel 19."
                )
                weather_text = (
                    f"HAZARD WARNING: {alerts[0].severity} {alerts[0].event_type} detected. "
                    f"Reduced visibility and potential road closures in affected zones."
                )
                rest_stops = [
                    f"Intermodal Staging Depot near {shipment.origin.port_code}",
                    "Regional Safe Haven Rest Stop Milepost 142",
                ]
                exec_text = (
                    f"Executive Rationale: Automated reroute via {mode_val} saves an estimated "
                    f"${penalty_saved:,.2f} in demurrage penalties against an additional surcharge of "
                    f"${cost_usd:,.2f}. Detour transit time is {transit_hours:.1f} hours."
                )
            else:
                driver_text = (
                    f"SCHEDULE INSTRUCTIONS: Proceed on primary {mode_val} corridor to destination. "
                    "Corridor conditions normal."
                )
                weather_text = "No severe NOAA weather alerts active along corridor."
                rest_stops = ["Designated Commercial Rest Area Corridor Interstate"]
                exec_text = (
                    "Executive Rationale: Primary corridor is clear of disruptions; shipment "
                    "proceeds on schedule with zero detour expense."
                )

            llm_content = SyntheticLLMAdvisoryContent(
                driver_instructions=driver_text,
                weather_hazards=weather_text,
                recommended_rest_stops=rest_stops,
                executive_rationale=exec_text,
            )

        driver_advisory = DriverAdvisory(
            driver_instructions=llm_content.driver_instructions,
            weather_hazards=llm_content.weather_hazards,
            recommended_rest_stops=llm_content.recommended_rest_stops,
        )

        # Build synthetic EDI 204 summary line
        edi_summary = (
            f"EDI-204 TENDER SUMMARY [Shipment={shipment.shipment_id}, Mode={mode_val}, "
            f"Route={route_id}, DetourCost=${cost_usd:,.2f}, SlaDeadline={shipment.sla_delivery_deadline.isoformat()}]"
        )

        # STRICT GROUNDING: Inject exact deterministic figures directly from state
        brief = IncidentAdvisoryBrief(
            shipment_id=shipment.shipment_id,
            driver_advisory=driver_advisory,
            executive_rationale=llm_content.executive_rationale,
            additional_cost_usd=cost_usd,
            estimated_penalty_saved=penalty_saved,
            estimated_transit_hours=transit_hours,
            requires_human_approval=requires_human,
            edi_tender_summary=edi_summary,
            generated_at=datetime.now(UTC),
        )

        log_entry = (
            f"GENERATE_ADVISORY: Created audit-ready incident brief and driver advisory "
            f"[Route={route_id}, Cost=+${cost_usd:,.2f}, DemurrageSaved=${penalty_saved:,.2f}]"
        )

        return {
            "incident_advisory": brief,
            "audit_log": [log_entry],
        }

    return generate_advisory_node
