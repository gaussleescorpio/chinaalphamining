from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceClaim:
    value: float
    source: str
    independently_reproduced: bool
    scope: str


@dataclass(frozen=True)
class ShadowGateInput:
    reproducible: bool
    point_in_time_audited: bool
    leakage_controls_passed: bool
    execution_clock_verified: bool
    development_evidence_passed: bool
    historical_pressure_continuity: bool
    cost_model_present: bool
    risk_limits_present: bool
    monitoring_present: bool
    external_continuation_probability: EvidenceClaim | None = None


@dataclass(frozen=True)
class ShadowQualification:
    qualified: bool
    status: str
    reasons: tuple[str, ...]
    claim_disclosure: str | None


def qualify_for_shadow(gate: ShadowGateInput) -> ShadowQualification:
    """Decide research-to-shadow eligibility without implying live approval."""

    required = {
        "REPRODUCIBLE": gate.reproducible,
        "PIT_AUDITED": gate.point_in_time_audited,
        "LEAKAGE_CONTROLS": gate.leakage_controls_passed,
        "EXECUTION_CLOCK": gate.execution_clock_verified,
        "DEVELOPMENT_EVIDENCE": gate.development_evidence_passed,
        "HISTORICAL_CONTINUITY": gate.historical_pressure_continuity,
        "COST_MODEL": gate.cost_model_present,
        "RISK_LIMITS": gate.risk_limits_present,
        "MONITORING": gate.monitoring_present,
    }
    failures = tuple(name for name, passed in required.items() if not passed)
    claim = gate.external_continuation_probability
    disclosure = None
    if claim is not None:
        disclosure = (
            f"External validation claim: {claim.value:.2%}; source={claim.source}; "
            f"scope={claim.scope}; independently_reproduced={claim.independently_reproduced}."
        )
    if failures:
        return ShadowQualification(False, "NOT_QUALIFIED", failures, disclosure)
    return ShadowQualification(
        True,
        "SHADOW_RESEARCH_QUALIFIED",
        ("PAPER_ONLY", "NO_LIVE_ORDER_AUTHORITY", "MONITORING_REQUIRED"),
        disclosure,
    )
