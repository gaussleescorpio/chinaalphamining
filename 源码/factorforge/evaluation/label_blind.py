from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata


@dataclass(frozen=True)
class LabelBlindResult:
    accepted: bool
    reason_code: str
    coverage: float
    variance: float
    rank_path_sha256: str
    cross_sectional_variation_fraction: float


def orientation_free_rank_fingerprint(values: np.ndarray) -> str:
    """Hash cross-sectional rank paths, treating sign reversal as identical."""

    matrix = np.asarray(values, dtype=float)
    rows: list[bytes] = []
    for row in matrix:
        active = np.isfinite(row)
        codes = np.zeros(len(row), dtype=np.int32)
        if np.sum(active) >= 2:
            ranks = np.rint(2.0 * rankdata(row[active], method="average")).astype(
                np.int32
            )
            reverse = int(ranks.max() + ranks.min()) - ranks
            chosen = ranks if ranks.tobytes() <= reverse.tobytes() else reverse
            codes[active] = chosen
        rows.append(active.astype(np.uint8).tobytes() + codes.tobytes())
    return hashlib.sha256(b"".join(rows)).hexdigest()


def audit_values(values: np.ndarray, minimum_coverage: float) -> LabelBlindResult:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("candidate values must be a time-by-asset matrix")
    coverage = float(np.mean(np.isfinite(matrix)))
    variance = float(np.nanvar(matrix)) if np.any(np.isfinite(matrix)) else 0.0
    finite = np.isfinite(matrix)
    active_counts = np.sum(finite, axis=1)
    clean = np.where(finite, matrix, 0.0)
    row_mean = np.divide(
        clean.sum(axis=1),
        active_counts,
        out=np.zeros(matrix.shape[0], dtype=float),
        where=active_counts > 0,
    )
    squared_deviation = np.where(finite, (matrix - row_mean[:, None]) ** 2, 0.0)
    row_variance = np.divide(
        squared_deviation.sum(axis=1),
        active_counts,
        out=np.full(matrix.shape[0], np.nan, dtype=float),
        where=active_counts > 0,
    )
    cross_sectional_variation_fraction = float(
        np.mean(
            (active_counts >= 3) & np.isfinite(row_variance) & (row_variance > 1e-14)
        )
    )
    fingerprint = orientation_free_rank_fingerprint(matrix)
    if coverage < minimum_coverage:
        return LabelBlindResult(
            False,
            "INSUFFICIENT_COVERAGE",
            coverage,
            variance,
            fingerprint,
            cross_sectional_variation_fraction,
        )
    if not np.isfinite(variance) or variance <= 1e-14:
        return LabelBlindResult(
            False,
            "CONSTANT_OR_INVALID",
            coverage,
            variance,
            fingerprint,
            cross_sectional_variation_fraction,
        )
    if cross_sectional_variation_fraction < minimum_coverage:
        return LabelBlindResult(
            False,
            "INSUFFICIENT_CROSS_SECTIONAL_VARIATION",
            coverage,
            variance,
            fingerprint,
            cross_sectional_variation_fraction,
        )
    return LabelBlindResult(
        True,
        "LABEL_BLIND_PASS",
        coverage,
        variance,
        fingerprint,
        cross_sectional_variation_fraction,
    )
