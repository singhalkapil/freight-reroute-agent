"""Runner interface for Freight Reroute Agent supporting streaming logs and checkpoint recovery."""

import logging
import uuid
from collections.abc import Iterator
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from freight_agent.graph.state import FreightAgentState
from freight_agent.graph.workflow import create_freight_agent_graph
from freight_agent.models.domain import ShipmentManifest
from freight_agent.tools.routing import IntermodalRouteSolver
from freight_agent.tools.weather import NOAAWeatherClient

logger = logging.getLogger(__name__)


class FreightAgentRunner:
    """Execution runner for the Freight Reroute Agent with streaming logs and HITL checkpoint recovery."""

    def __init__(
        self,
        weather_client: NOAAWeatherClient | None = None,
        route_solver: IntermodalRouteSolver | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> None:
        self.checkpointer = checkpointer if checkpointer is not None else MemorySaver()
        self.graph = create_freight_agent_graph(
            weather_client=weather_client,
            route_solver=route_solver,
            checkpointer=self.checkpointer,
        )

    def run(
        self,
        manifest: ShipmentManifest,
        thread_id: str | None = None,
    ) -> FreightAgentState:
        """Run workflow synchronously for a given shipment manifest to completion or interrupt."""
        tid = thread_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": tid}}
        initial_state: FreightAgentState = {
            "shipment": manifest,
            "active_alerts": [],
            "candidate_routes": [],
            "decision": None,
            "human_approved": None,
            "audit_log": [],
            "edi_tender_payload": None,
            "current_corridor_disrupted": False,
            "requires_human_review": False,
        }

        result = self.graph.invoke(initial_state, config=config)
        return result

    def stream_run(
        self,
        manifest: ShipmentManifest,
        thread_id: str | None = None,
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        """Stream workflow execution yielding (node_name, state_update) pairs in real time."""
        tid = thread_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": tid}}
        initial_state: FreightAgentState = {
            "shipment": manifest,
            "active_alerts": [],
            "candidate_routes": [],
            "decision": None,
            "human_approved": None,
            "audit_log": [],
            "edi_tender_payload": None,
            "current_corridor_disrupted": False,
            "requires_human_review": False,
        }

        for event in self.graph.stream(initial_state, config=config):
            yield from event.items()

    def resume_human_review(
        self,
        thread_id: str,
        approved: bool = True,
        comments: str = "",
    ) -> FreightAgentState:
        """Resume an interrupted workflow from its checkpoint with a human dispatcher decision."""
        config = {"configurable": {"thread_id": thread_id}}
        resume_command = Command(
            resume={
                "approved": approved,
                "comments": comments or ("Approved by dispatcher" if approved else "Rejected by dispatcher"),
            }
        )
        return self.graph.invoke(resume_command, config=config)

    def get_state(self, thread_id: str) -> Any:
        """Retrieve current state snapshot and pending interrupts for a thread from checkpointer."""
        config = {"configurable": {"thread_id": thread_id}}
        return self.graph.get_state(config)
