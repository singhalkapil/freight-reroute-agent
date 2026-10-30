"""Unit tests for unstructured and noisy dispatch communication parsing."""

from __future__ import annotations

import pytest

from freight_agent.models.domain import AlertSeverity
from freight_agent.parser.manifest_parser import UnstructuredManifestParser


class TestNoisyDispatchParser:
    """Test parsing of informal dispatcher communications and event logs."""

    def test_parses_whatsapp_incident_feed(self) -> None:
        """Parses informal dispatcher message on I-80 corridor with delay."""
        parser = UnstructuredManifestParser()
        raw_msg = (
            "URGENT DISPATCH: Bridge collapsed near mile marker 42 on I-80 corridor! "
            "Corridor closure for next 36 hours. All inbound truckloads held."
        )
        event = parser.parse_noisy_dispatch_event(raw_msg)
        assert event.corridor_location == "I-80"
        assert event.estimated_blockage_hours == 36.0
        assert event.severity == AlertSeverity.CRITICAL
        assert "URGENT DISPATCH" in event.raw_message_snippet

    def test_parses_port_canal_lock_closure(self) -> None:
        """Parses canal lock maintenance downtime communication."""
        parser = UnstructuredManifestParser()
        raw_msg = (
            "NOTICE: Lock and Dam 26 maintenance causing 12 hours delay "
            "along Mississippi River Barge Corridor."
        )
        event = parser.parse_noisy_dispatch_event(raw_msg)
        assert event.estimated_blockage_hours == 12.0
        assert event.severity == AlertSeverity.HIGH
