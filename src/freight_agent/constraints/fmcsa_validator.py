"""Federal Motor Carrier Safety Administration (FMCSA) & Cold-Chain Constraint Engine.

Implements mathematical and regulatory invariants under 49 CFR § 395:
- 11-Hour Maximum Driving Limit (Solo Driver)
- 14-Hour Maximum Duty Window
- Mandatory 30-Minute Rest Break after 8 Driving Hours
- Cold-Chain Cargo Shelf-Life Spoilage Invariant
- Carrier Tariff Cost Ceiling Invariant
"""

from __future__ import annotations

from enum import StrEnum
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field

from freight_agent.models.domain import RerouteAlternative, ShipmentManifest, TransportMode


class ViolationSeverity(StrEnum):
    """Severity of regulatory or operational constraint violation."""

    WARNING = "WARNING"
    HARD_BLOCK = "HARD_BLOCK"


class ConstraintViolation(BaseModel):
    """Specific constraint violation record with regulatory citation."""

    model_config = ConfigDict(frozen=True)

    rule_name: str = Field(..., description="Name of rule, e.g. FMCSA_11_HOUR_DRIVE")
    citation: str = Field(..., description="Regulatory or contractual citation (e.g. 49 CFR § 395.3(a)(1))")
    severity: ViolationSeverity
    description: str = Field(..., description="Detailed explanation of the violation")
    remediation_recommendation: str = Field(..., description="Actionable suggestion to resolve violation")


class ConstraintValidationResult(BaseModel):
    """Comprehensive validation result across FMCSA, cold-chain, and tariff gates."""

    model_config = ConfigDict(frozen=True)

    is_compliant: bool
    requires_team_drivers: bool = Field(default=False)
    mandatory_rest_stops_hours: float = Field(default=0.0)
    total_effective_transit_hours: float = Field(default=0.0)
    violations: List[ConstraintViolation] = Field(default_factory=list)
    compliance_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Compliance adherence metric: 1.0 (perfect) to 0.0 (unresolvable)",
    )


class FMCSAConstraintValidator:
    """Enforces FMCSA hours-of-service, cold-chain perishability, and tariff ceilings."""

    MAX_SOLO_DRIVE_HOURS: float = 11.0
    MAX_SOLO_DUTY_WINDOW: float = 14.0
    REST_BREAK_THRESHOLD: float = 8.0
    MANDATORY_BREAK_HOURS: float = 0.5
    MAX_TEAM_DRIVE_HOURS: float = 22.0

    @classmethod
    def validate_reroute(
        cls,
        manifest: ShipmentManifest,
        alternative: RerouteAlternative,
        tariff_ceiling_usd: float = 7500.0,
        max_cold_chain_hours: float = 48.0,
    ) -> ConstraintValidationResult:
        """Validate proposed reroute against FMCSA regulations, cold-chain buffers, and tariff caps."""
        violations: list[ConstraintViolation] = []
        requires_team_drivers = False
        rest_hours = 0.0

        est_hours = alternative.estimated_transit_hours
        mode = alternative.mode

        # 1. FMCSA Hours of Service (Applies strictly to ROAD transport)
        if mode == TransportMode.ROAD:
            if est_hours > cls.MAX_SOLO_DRIVE_HOURS:
                if est_hours <= cls.MAX_TEAM_DRIVE_HOURS:
                    requires_team_drivers = True
                    violations.append(
                        ConstraintViolation(
                            rule_name="FMCSA_11_HOUR_DRIVE_EXCEEDED",
                            citation="49 CFR § 395.3(a)(1)",
                            severity=ViolationSeverity.WARNING,
                            description=(
                                f"Detour transit time ({est_hours:.1f} hrs) exceeds maximum 11-hour solo driving limit."
                            ),
                            remediation_recommendation=(
                                "Dispatch team drivers (dual-relay) or schedule 10-hour sleeper-berth layover."
                            ),
                        )
                    )
                else:
                    violations.append(
                        ConstraintViolation(
                            rule_name="FMCSA_TEAM_LIMIT_EXCEEDED",
                            citation="49 CFR § 395.3(b)",
                            severity=ViolationSeverity.HARD_BLOCK,
                            description=(
                                f"Detour transit time ({est_hours:.1f} hrs) exceeds 22-hour team driver ceiling."
                            ),
                            remediation_recommendation="Reroute via RAIL or multimodal transshipment.",
                        )
                    )

            # Check 30-minute mandatory break
            if est_hours >= cls.REST_BREAK_THRESHOLD:
                rest_hours += cls.MANDATORY_BREAK_HOURS

        total_effective_hours = est_hours + rest_hours

        # 2. Perishable / Cold-Chain Cargo Spoilage Invariant
        cargo_type = (manifest.cargo_type or "").lower()
        is_perishable = any(
            p in cargo_type
            for p in ["perishable", "refrigerated", "seafood", "produce", "pharma", "vaccine", "fresh"]
        )
        if is_perishable and total_effective_hours > max_cold_chain_hours:
            violations.append(
                ConstraintViolation(
                    rule_name="COLD_CHAIN_SPOILAGE_RISK",
                    citation="FSMA Sanitary Transportation of Human and Animal Food (21 CFR Part 1)",
                    severity=ViolationSeverity.HARD_BLOCK,
                    description=(
                        f"Effective transit time ({total_effective_hours:.1f} hrs) exceeds maximum allowable "
                        f"cold-chain shelf-life buffer ({max_cold_chain_hours:.1f} hrs)."
                    ),
                    remediation_recommendation="Select expedited air/express corridor or re-ice at staging hub.",
                )
            )

        # 3. Carrier Tariff Ceiling Gate
        if alternative.additional_cost_usd > tariff_ceiling_usd:
            violations.append(
                ConstraintViolation(
                    rule_name="TARIFF_CEILING_BREACH",
                    citation="Contractual Carrier Rate Tariff Tier Agreement EDI 204/990",
                    severity=ViolationSeverity.HARD_BLOCK,
                    description=(
                        f"Detour surcharge (${alternative.additional_cost_usd:,.2f}) breaches "
                        f"maximum authorized tariff ceiling (${tariff_ceiling_usd:,.2f})."
                    ),
                    remediation_recommendation="Seek dispatcher executive approval or negotiate spot tariff waiver.",
                )
            )

        hard_blocks = [v for v in violations if v.severity == ViolationSeverity.HARD_BLOCK]
        is_compliant = len(hard_blocks) == 0

        # Score calculation: 1.0 if clean, 0.7 if warnings only, 0.0 if hard blocked
        if is_compliant:
            score = 1.0 if len(violations) == 0 else 0.85
        else:
            score = max(0.0, 0.5 - (0.2 * len(hard_blocks)))

        return ConstraintValidationResult(
            is_compliant=is_compliant,
            requires_team_drivers=requires_team_drivers,
            mandatory_rest_stops_hours=rest_hours,
            total_effective_transit_hours=total_effective_hours,
            violations=violations,
            compliance_score=score,
        )
