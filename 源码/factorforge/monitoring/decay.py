from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DecayAssessment:
    state: str
    warnings: tuple[str, ...]
    recent_mean: float
    reference_mean: float
    retention_ratio: float
    recent_positive_fraction: float


def assess_decay(return_path: np.ndarray, recent_length: int = 63) -> DecayAssessment:
    """Three-state monitor. It diagnoses; it never changes a frozen factor."""

    values = np.asarray(return_path, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < recent_length * 2:
        return DecayAssessment("WATCH", ("INSUFFICIENT_HISTORY",), 0.0, 0.0, 0.0, 0.0)
    recent = values[-recent_length:]
    reference = values[:-recent_length]
    recent_mean = float(np.mean(recent))
    reference_mean = float(np.mean(reference))
    retention = recent_mean / reference_mean if abs(reference_mean) > 1e-12 else 0.0
    positive_fraction = float(np.mean(recent > 0.0))
    warnings = []
    if recent_mean <= 0.0:
        warnings.append("RECENT_MEAN_NONPOSITIVE")
    if reference_mean > 0.0 and retention < 0.30:
        warnings.append("RETENTION_BELOW_30_PERCENT")
    if positive_fraction < 0.40:
        warnings.append("RECENT_POSITIVE_FRACTION_LOW")
    state = "NORMAL" if not warnings else "PAUSE" if len(warnings) >= 2 else "WATCH"
    return DecayAssessment(
        state,
        tuple(warnings),
        recent_mean,
        reference_mean,
        retention,
        positive_fraction,
    )
