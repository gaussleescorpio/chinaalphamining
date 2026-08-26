from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class HACMeanTest:
    observations: int
    mean: float
    standard_error: float
    t_statistic: float
    one_sided_p_value: float
    lag: int


def newey_west_mean_test(values: np.ndarray, lag: int | None = None) -> HACMeanTest:
    """One-sided mean test using a Bartlett-kernel Newey-West variance."""

    active = np.asarray(values, dtype=float)
    active = active[np.isfinite(active)]
    count = len(active)
    if count < 3:
        mean = float(np.mean(active)) if count else 0.0
        return HACMeanTest(count, mean, math.inf, 0.0, 1.0, 0)
    chosen_lag = min(
        count - 1,
        max(0, int(lag if lag is not None else 4 * (count / 100.0) ** (2.0 / 9.0))),
    )
    centered = active - np.mean(active)
    long_run_variance = float(centered @ centered / count)
    for offset in range(1, chosen_lag + 1):
        covariance = float(centered[offset:] @ centered[:-offset] / count)
        long_run_variance += 2.0 * (1.0 - offset / (chosen_lag + 1.0)) * covariance
    standard_error = math.sqrt(max(long_run_variance, 0.0) / count)
    statistic = (
        float(np.mean(active) / standard_error) if standard_error > 1e-15 else 0.0
    )
    p_value = float(norm.sf(statistic)) if standard_error > 1e-15 else 1.0
    return HACMeanTest(
        count, float(np.mean(active)), standard_error, statistic, p_value, chosen_lag
    )


def moving_block_bootstrap_mean_interval(
    values: np.ndarray,
    *,
    block_length: int,
    repetitions: int = 1_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Deterministic circular moving-block bootstrap interval for the mean."""

    active = np.asarray(values, dtype=float)
    active = active[np.isfinite(active)]
    if len(active) < 3:
        return (math.nan, math.nan)
    block = min(len(active), max(1, int(block_length)))
    blocks_per_draw = math.ceil(len(active) / block)
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=float)
    offsets = np.arange(block)
    for index in range(repetitions):
        starts = rng.integers(0, len(active), size=blocks_per_draw)
        indices = (starts[:, None] + offsets[None, :]) % len(active)
        sample = active[indices.reshape(-1)[: len(active)]]
        means[index] = float(np.mean(sample))
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return float(low), float(high)


def stationary_bootstrap_mean_interval(
    values: np.ndarray,
    *,
    mean_block_length: int,
    repetitions: int = 1_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Stationary-bootstrap interval preserving dependence without fixed block edges."""

    active = np.asarray(values, dtype=float)
    active = active[np.isfinite(active)]
    if len(active) < 3:
        return (math.nan, math.nan)
    average_block = min(len(active), max(1, int(mean_block_length)))
    restart_probability = 1.0 / average_block
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=float)
    for draw in range(repetitions):
        indices = np.empty(len(active), dtype=int)
        indices[0] = int(rng.integers(0, len(active)))
        for position in range(1, len(active)):
            if rng.random() < restart_probability:
                indices[position] = int(rng.integers(0, len(active)))
            else:
                indices[position] = (indices[position - 1] + 1) % len(active)
        means[draw] = float(np.mean(active[indices]))
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return float(low), float(high)


def block_sign_flip_mean_p_value(
    values: np.ndarray,
    *,
    block_length: int,
    repetitions: int = 2_000,
    seed: int = 0,
) -> float:
    """One-sided randomization p-value using a common sign within each time block."""

    active = np.asarray(values, dtype=float)
    active = active[np.isfinite(active)]
    if len(active) < 3:
        return 1.0
    block = min(len(active), max(1, int(block_length)))
    block_ids = np.arange(len(active)) // block
    block_count = int(block_ids[-1] + 1)
    observed = float(np.mean(active))
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(repetitions):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=block_count)
        permuted = float(np.mean(active * signs[block_ids]))
        exceedances += int(permuted >= observed)
    return float((exceedances + 1) / (repetitions + 1))


def autocorrelation_effective_sample_size(
    values: np.ndarray, maximum_lag: int | None = None
) -> float:
    """Estimate effective observations, stopping when serial correlation turns negative."""

    active = np.asarray(values, dtype=float)
    active = active[np.isfinite(active)]
    count = len(active)
    if count < 3 or float(np.std(active)) <= 1e-15:
        return float(count)
    limit = min(count - 1, maximum_lag or max(1, int(math.sqrt(count))))
    centered = active - np.mean(active)
    denominator = float(centered @ centered)
    correlation_sum = 0.0
    for lag in range(1, limit + 1):
        rho = float(centered[lag:] @ centered[:-lag] / denominator)
        if rho <= 0.0:
            break
        correlation_sum += rho
    return float(max(1.0, min(count, count / (1.0 + 2.0 * correlation_sum))))


def profit_concentration(values: np.ndarray, fraction: float = 0.05) -> float:
    active = np.asarray(values, dtype=float)
    positive = np.sort(active[np.isfinite(active) & (active > 0.0)])[::-1]
    total = float(np.sum(positive))
    if total <= 0.0:
        return 1.0
    count = max(1, int(math.ceil(len(positive) * fraction)))
    return float(np.sum(positive[:count]) / total)


def leave_best_event_mean(values: np.ndarray) -> float:
    active = np.asarray(values, dtype=float)
    active = active[np.isfinite(active)]
    if len(active) < 2:
        return 0.0
    return float(np.mean(np.delete(active, int(np.argmax(active)))))
