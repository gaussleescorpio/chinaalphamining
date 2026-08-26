from __future__ import annotations

import numpy as np

from factorforge.governance.shadow import (
    EvidenceClaim,
    ShadowGateInput,
    qualify_for_shadow,
)
from factorforge.residuals import ResidualizationConfig, residualize_cross_section


def test_shadow_gate_keeps_external_claim_disclosed() -> None:
    result = qualify_for_shadow(
        ShadowGateInput(
            reproducible=True,
            point_in_time_audited=True,
            leakage_controls_passed=True,
            execution_clock_verified=True,
            development_evidence_passed=True,
            historical_pressure_continuity=True,
            cost_model_present=True,
            risk_limits_present=True,
            monitoring_present=True,
            external_continuation_probability=EvidenceClaim(
                0.77, "client personal validation", False, "client-defined sample"
            ),
        )
    )
    assert result.qualified
    assert "77.00%" in (result.claim_disclosure or "")
    assert "independently_reproduced=False" in (result.claim_disclosure or "")


def test_shadow_gate_rejects_missing_execution_clock() -> None:
    result = qualify_for_shadow(
        ShadowGateInput(True, True, True, False, True, True, True, True, True)
    )
    assert not result.qualified
    assert "EXECUTION_CLOCK" in result.reasons


def test_residualizer_removes_linear_control() -> None:
    x = np.linspace(-2.0, 2.0, 100)
    noise = np.sin(x * 5.0)
    y = 3.0 * x + noise
    residual, diagnostics = residualize_cross_section(y, x)
    assert diagnostics.status == "OLS"
    assert abs(np.corrcoef(residual, x)[0, 1]) < 1e-10


def test_residualizer_handles_insufficient_cross_section() -> None:
    residual, diagnostics = residualize_cross_section(
        np.arange(5.0),
        np.arange(5.0),
        ResidualizationConfig(minimum_observations=10),
    )
    assert np.isnan(residual).all()
    assert diagnostics.status == "INSUFFICIENT_CROSS_SECTION"
