from __future__ import annotations

import numpy as np


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Return monotone Benjamini-Hochberg q-values for a fixed hypothesis set."""

    values = np.asarray(p_values, dtype=float)
    if (
        values.ndim != 1
        or np.any(~np.isfinite(values))
        or np.any((values < 0.0) | (values > 1.0))
    ):
        raise ValueError("p_values must be a finite one-dimensional array in [0, 1]")
    count = len(values)
    if count == 0:
        return values.copy()
    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    adjusted = ranked * count / np.arange(1, count + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output


def benjamini_hochberg_total(p_values: np.ndarray, total_hypotheses: int) -> np.ndarray:
    """BH q-values with unreported hypotheses conservatively treated as p=1."""

    values = np.asarray(p_values, dtype=float)
    if total_hypotheses < len(values):
        raise ValueError(
            "total_hypotheses cannot be smaller than the supplied p-value vector"
        )
    if (
        values.ndim != 1
        or np.any(~np.isfinite(values))
        or np.any((values < 0.0) | (values > 1.0))
    ):
        raise ValueError("p_values must be a finite one-dimensional array in [0, 1]")
    if not len(values):
        return values.copy()
    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    adjusted = ranked * total_hypotheses / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output
