from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ResidualizationConfig:
    minimum_observations: int = 20
    maximum_condition_number: float = 1.0e6
    ridge_penalty: float = 1.0e-6
    fallback: str = "nan"


@dataclass(frozen=True)
class ResidualizationDiagnostics:
    observations: int
    controls: int
    condition_number: float
    used_ridge: bool
    status: str


def residualize_cross_section(
    target: np.ndarray,
    controls: np.ndarray,
    config: ResidualizationConfig = ResidualizationConfig(),
) -> tuple[np.ndarray, ResidualizationDiagnostics]:
    """Residualize one visible cross-section and report numerical failure modes.

    The function performs no lagging. Callers must supply point-in-time target and
    controls already aligned to the decision timestamp.
    """

    y = np.asarray(target, dtype=float)
    x = np.asarray(controls, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim != 1 or x.ndim != 2 or len(y) != len(x):
        raise ValueError("target and controls must describe one aligned cross-section")
    valid = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    minimum = max(config.minimum_observations, x.shape[1] + 3)
    output = np.full(y.shape, np.nan, dtype=float)
    if int(valid.sum()) < minimum:
        if config.fallback == "demean" and valid.any():
            output[valid] = y[valid] - np.mean(y[valid])
        return output, ResidualizationDiagnostics(
            int(valid.sum()), x.shape[1], float("inf"), False, "INSUFFICIENT_CROSS_SECTION"
        )
    design = np.column_stack((np.ones(int(valid.sum())), x[valid]))
    condition = float(np.linalg.cond(design))
    use_ridge = not np.isfinite(condition) or condition > config.maximum_condition_number
    if use_ridge:
        gram = design.T @ design
        penalty = np.eye(gram.shape[0]) * config.ridge_penalty
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(gram + penalty, design.T @ y[valid])
        status = "RIDGE_FALLBACK"
    else:
        coefficients, *_ = np.linalg.lstsq(design, y[valid], rcond=None)
        status = "OLS"
    output[valid] = y[valid] - design @ coefficients
    return output, ResidualizationDiagnostics(
        int(valid.sum()), x.shape[1], condition, use_ridge, status
    )
