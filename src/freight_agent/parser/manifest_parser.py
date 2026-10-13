"""Unstructured shipment manifest ingestion using LangChain structured output.

Ingests raw dispatch emails, carrier notes, and text-extracted Bills of Lading (BOL),
applying security guardrails (prompt injection sanitization), deterministic port geocoding
fallbacks for missing coordinates, and schema validation into immutable ShipmentManifest objects.
"""

import logging
import re
from datetime import UTC, datetime
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError

from freight_agent.models.domain import Coordinates, PortLocation, ShipmentManifest, ShipmentStatus

logger = logging.getLogger(__name__)


# Deterministic port geocoding fallback dictionary (UN/LOCODE and major cargo terminals)
KNOWN_PORT_COORDINATES: dict[str, tuple[float, float]] = {
    "USLAX": (33.74318, -118.26725),  # Port of Los Angeles, CA
    "USLGB": (33.75418, -118.21645),  # Port of Long Beach, CA
    "USNYC": (40.6698, -74.1569),  # Port of New York & New Jersey
    "USSAV": (32.1283, -81.1444),  # Port of Savannah, GA
    "USHOU": (29.7275, -95.2635),  # Port of Houston, TX
    "USGAL": (29.3108, -94.7925),  # Port of Galveston, TX
    "USSEA": (47.5855, -122.3551),  # Port of Seattle, WA
    "USTAC": (47.2667, -122.4167),  # Port of Tacoma, WA
    "USCHI": (41.8781, -87.6298),  # Chicago Intermodal Rail Terminal
    "USBAL": (39.2715, -76.5828),  # Port of Baltimore, MD
    "USORF": (36.9461, -76.3267),  # Port of Virginia / Norfolk, VA
    "USMIA": (25.7781, -80.1794),  # Port of Miami, FL
    "USOAK": (37.7952, -122.2797),  # Port of Oakland, CA
    "USNOL": (29.9322, -90.0439),  # Port of New Orleans, LA
    "USMEM": (35.1495, -90.0490),  # Memphis Intermodal Hub, TN
    "USKCY": (39.0997, -94.5786),  # Kansas City Intermodal Rail Hub
    "USDFW": (32.7767, -96.7970),  # Dallas/Fort Worth Intermodal Hub
    "USATL": (33.7490, -84.3880),  # Atlanta Freight Logistics Hub
}

# Suspicious instruction override tokens and prompt injection attack patterns
SUSPICIOUS_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.IGNORECASE),
    re.compile(r"system\s+prompt", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+a", re.IGNORECASE),
    re.compile(r"disregard\s+all", re.IGNORECASE),
    re.compile(r"new\s+role\s*:", re.IGNORECASE),
    re.compile(r"bypass\s+(safety|guardrails|security)", re.IGNORECASE),
    re.compile(r"output\s+only\s+the\s+following", re.IGNORECASE),
    re.compile(r"drop\s+database|exec\s+xp_|union\s+select", re.IGNORECASE),
]


class ManifestParserError(Exception):
    """Base exception for manifest parsing errors."""


class PromptInjectionDetectedError(ManifestParserError):
    """Raised when suspicious prompt injection tokens are detected in input text."""


class RawExtractedManifest(BaseModel):
    """Intermediate extraction schema matching natural language LLM parsing before coordinate resolution."""

    shipment_id: str = Field(description="Unique shipment ID, BOL number, or tracking number")
    origin_port_code: str = Field(
        description="Origin port UN/LOCODE (e.g. USLAX, USNYC, USHOU) or terminal code"
    )
    origin_latitude: float | None = Field(
        default=None, description="Origin latitude if explicitly specified, otherwise null"
    )
    origin_longitude: float | None = Field(
        default=None, description="Origin longitude if explicitly specified, otherwise null"
    )
    destination_latitude: float | None = Field(
        default=None, description="Destination latitude if explicitly specified, otherwise null"
    )
    destination_longitude: float | None = Field(
        default=None, description="Destination longitude if explicitly specified, otherwise null"
    )
    destination_name_or_code: str | None = Field(
        default=None,
        description="Destination city, port code, or facility name (e.g., Chicago, USCHI, Dallas)",
    )
    cargo_type: str = Field(
        description="Cargo commodity or description (e.g., Electronics, Fresh Produce)"
    )
    perishability_hours: float | None = Field(
        default=None, description="Shelf life or cold chain limit in hours if applicable"
    )
    demurrage_rate_per_day: float = Field(
        default=1200.0,
        description="Demurrage penalty rate per day in USD. Default to 1200.0 if not specified.",
    )
    sla_delivery_deadline: str = Field(
        description="Contractual delivery deadline in ISO-8601 format (e.g. 2026-10-20T18:00:00Z)"
    )
    current_status: str = Field(
        default="IN_TRANSIT",
        description="Operational status (e.g., IN_TRANSIT, PLANNED, DELAYED)",
    )


def sanitize_manifest_input(text: str, strict: bool = False) -> str:
    """Sanitize raw manifest text by detecting and defusing prompt injection patterns.

    Args:
        text: Raw document, email, or OCR string.
        strict: If True, raises PromptInjectionDetectedError upon detecting suspicious patterns.
                If False, strips out the suspicious statements and logs a warning.

    Returns:
        Cleaned input text safe for LLM ingestion.
    """
    cleaned = text.strip()

    for pattern in SUSPICIOUS_PATTERNS:
        match = pattern.search(cleaned)
        if match:
            if strict:
                raise PromptInjectionDetectedError(
                    f"Potential prompt injection pattern detected: '{match.group(0)}'"
                )
            logger.warning(
                "Sanitizing detected prompt injection attempt in manifest text: %s",
                match.group(0),
            )
            cleaned = pattern.sub("[REDACTED_INJECTION_ATTEMPT]", cleaned)

    return cleaned


SYSTEM_EXTRACTION_PROMPT = """You are an expert intermodal freight documentation parser.
Extract the structured shipment manifest details from the following raw text (carrier email, dispatch note, or Bill of Lading).

IMPORTANT INSTRUCTIONS:
- Identify the unique shipment ID / tracking number.
- Identify the origin port code (UN/LOCODE such as USLAX, USNYC, USHOU, USGAL, USSEA). If unknown, extract standard 5-character port code.
- If explicit latitude/longitude numbers are present in the text, extract them. If coordinates are absent, leave them null.
- Extract commodity/cargo type.
- Extract perishability limit in hours if mentioned (e.g. cold chain, refrigerated, pharmaceuticals).
- Extract demurrage penalty rate per day in USD. Default to 1200.0 if not stated.
- Extract contractual SLA delivery deadline in ISO-8601 UTC format (YYYY-MM-DDTHH:MM:SSZ).
"""


class UnstructuredManifestParser:
    """Enterprise parser for unstructured freight dispatch emails, notes, and documents.

    Uses LangChain `.with_structured_output()` to guarantee structured Pydantic model extraction,
    complemented by deterministic geocoding fallbacks and security sanitization.
    """

    def __init__(
        self,
        llm: BaseChatModel | Any,
        geocoding_lookup: dict[str, tuple[float, float]] | None = None,
        strict_guardrails: bool = False,
    ) -> None:
        """Initialize the manifest parser.

        Args:
            llm: LangChain chat model instance supporting .with_structured_output().
            geocoding_lookup: Custom or standard mapping of port codes to (lat, lon) coordinates.
            strict_guardrails: If True, raises PromptInjectionDetectedError on suspicious input.
        """
        self.llm = llm
        self.geocoding_lookup = (
            geocoding_lookup if geocoding_lookup is not None else KNOWN_PORT_COORDINATES
        )
        self.strict_guardrails = strict_guardrails

        # Configure structured output chain
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_EXTRACTION_PROMPT),
                ("human", "Raw manifest document text:\n\n{document_text}"),
            ]
        )
        self.structured_llm = self.llm.with_structured_output(RawExtractedManifest)

    def _resolve_coordinates(
        self,
        lat: float | None,
        lon: float | None,
        code_or_name: str | None,
        entity_name: str = "Location",
    ) -> tuple[float, float]:
        """Resolve geographic coordinates using explicit values or fallback geocoding lookup."""
        if lat is not None and lon is not None:
            # Validate within coordinate bounds
            if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                return (lat, lon)
            logger.warning(
                "%s coordinates (%s, %s) out of bounds, falling back to geocoding lookup",
                entity_name,
                lat,
                lon,
            )

        # Lookup by port code / name
        if code_or_name:
            normalized_key = code_or_name.strip().upper()
            if normalized_key in self.geocoding_lookup:
                return self.geocoding_lookup[normalized_key]

            # Try prefix / substring lookup in known dictionary (e.g. 'CHICAGO' -> 'USCHI')
            for key, coords in self.geocoding_lookup.items():
                if key in normalized_key or normalized_key in key:
                    return coords

        raise ManifestParserError(
            f"Cannot resolve coordinates for {entity_name} '{code_or_name}': "
            f"no valid explicit coordinates and not found in deterministic geocoding lookup."
        )

    def _parse_deadline(self, deadline_str: str) -> datetime:
        """Parse SLA deadline string into a timezone-aware UTC datetime."""
        cleaned = deadline_str.strip()
        # Handle trailing Z
        if cleaned.endswith("Z"):
            cleaned = cleaned[:-1] + "+00:00"

        try:
            dt = datetime.fromisoformat(cleaned)
        except ValueError:
            # Try date-only format
            try:
                dt = datetime.strptime(cleaned, "%Y-%m-%d").replace(tzinfo=UTC)
            except ValueError as err:
                raise ManifestParserError(
                    f"Failed to parse SLA delivery deadline '{deadline_str}': {err}"
                ) from err

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt

    def parse(self, raw_text: str) -> ShipmentManifest:
        """Parse raw manifest text into a validated, immutable ShipmentManifest.

        Args:
            raw_text: Raw unstructured document or email text.

        Returns:
            Validated immutable ShipmentManifest Pydantic model.

        Raises:
            PromptInjectionDetectedError: If prompt injection detected in strict mode.
            ManifestParserError: If extraction or validation fails.
        """
        if not raw_text or not raw_text.strip():
            raise ManifestParserError("Manifest input text is empty.")

        # Step 1: Security sanitization
        sanitized_text = sanitize_manifest_input(raw_text, strict=self.strict_guardrails)

        # Step 2: Invoke structured LLM extraction
        try:
            chain = self.prompt | self.structured_llm
            extracted: RawExtractedManifest = chain.invoke({"document_text": sanitized_text})
        except ValidationError as val_err:
            raise ManifestParserError(
                f"LLM structured output schema validation failed: {val_err}"
            ) from val_err
        except Exception as err:
            logger.error("Structured LLM extraction invocation failed: %s", err)
            raise ManifestParserError(f"LLM extraction error: {err}") from err

        # Step 3: Resolve origin port and coordinates (with fallback)
        origin_code = extracted.origin_port_code.strip().upper()
        # Ensure code is 3-10 alphanumeric
        sanitized_origin_code = re.sub(r"[^A-Z0-9]", "", origin_code) or "USLAX"

        origin_lat, origin_lon = self._resolve_coordinates(
            lat=extracted.origin_latitude,
            lon=extracted.origin_longitude,
            code_or_name=sanitized_origin_code,
            entity_name="Origin",
        )
        origin_location = PortLocation(
            latitude=origin_lat,
            longitude=origin_lon,
            port_code=sanitized_origin_code,
        )

        # Step 4: Resolve destination coordinates (with fallback)
        dest_lat, dest_lon = self._resolve_coordinates(
            lat=extracted.destination_latitude,
            lon=extracted.destination_longitude,
            code_or_name=extracted.destination_name_or_code,
            entity_name="Destination",
        )
        destination_location = Coordinates(
            latitude=dest_lat,
            longitude=dest_lon,
        )

        # Step 5: Parse SLA deadline
        deadline_dt = self._parse_deadline(extracted.sla_delivery_deadline)

        # Step 6: Determine operational status
        try:
            status = ShipmentStatus(extracted.current_status.strip().upper())
        except ValueError:
            status = ShipmentStatus.IN_TRANSIT

        # Step 7: Construct and validate immutable ShipmentManifest
        try:
            manifest = ShipmentManifest(
                shipment_id=extracted.shipment_id.strip(),
                origin=origin_location,
                destination=destination_location,
                cargo_type=extracted.cargo_type.strip(),
                perishability_hours=extracted.perishability_hours,
                demurrage_rate_per_day=max(0.0, float(extracted.demurrage_rate_per_day)),
                sla_delivery_deadline=deadline_dt,
                current_status=status,
            )
            return manifest
        except ValidationError as err:
            raise ManifestParserError(f"Failed to instantiate ShipmentManifest: {err}") from err
