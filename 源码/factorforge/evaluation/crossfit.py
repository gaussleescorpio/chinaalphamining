from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from factorforge.evaluation.shape import payoff_shape
from factorforge.evaluation.statistics import (
    autocorrelation_effective_sample_size,
    block_sign_flip_mean_p_value,
    leave_best_event_mean,
    moving_block_bootstrap_mean_interval,
    newey_west_mean_test,
    profit_concentration,
    stationary_bootstrap_mean_interval,
)


@dataclass(frozen=True)
class CandidateEvidence:
    candidate_id: str
    final_direction: int
    raw_values: np.ndarray
    rank_values: np.ndarray
    score_values: np.ndarray
    oof_return_path: np.ndarray
    metrics: dict[str, Any]
    p_value: float


def evaluate_frozen_oos(
    evidence: CandidateEvidence,
    future_return: np.ndarray,
    timestamps: pd.DatetimeIndex,
    oos_start: str,
    oos_end: str,
    periods_per_year: int,
    one_way_cost_bps: float,
    holding_period: int = 1,
) -> tuple[np.ndarray, dict[str, float]]:
    """Evaluate a frozen formula and direction without changing either."""

    start = pd.Timestamp(oos_start, tz="UTC")
    end = pd.Timestamp(oos_end, tz="UTC")
    mask = (timestamps >= start) & (timestamps <= end)
    score = evidence.score_values[mask]
    future = np.asarray(future_return, dtype=float)[mask]
    weights = _weights(score)
    net = cohort_portfolio_path(weights, future, holding_period, one_way_cost_bps)
    active = net[np.isfinite(net)]
    if len(active) < 2:
        return net, {
            "observations": int(len(active)),
            "annual_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
        }
    annual = float(
        np.expm1(np.mean(np.log1p(np.clip(active, -0.999999, None))) * periods_per_year)
    )
    std = float(np.std(active, ddof=1))
    drawdown = _max_drawdown(active)
    return net, {
        "observations": int(len(active)),
        "annual_return": annual,
        "sharpe": (
            float(np.mean(active) / std * math.sqrt(periods_per_year))
            if std > 0.0
            else 0.0
        ),
        "max_drawdown": drawdown,
        "calmar": annual / drawdown if drawdown > 0.0 else 0.0,
    }


def cross_sectional_ranks(values: np.ndarray) -> np.ndarray:
    output = np.full_like(values, np.nan, dtype=float)
    for index, row in enumerate(values):
        active = np.isfinite(row)
        if np.sum(active) < 2:
            continue
        ranks = rankdata(row[active], method="average")
        output[index, active] = (ranks - 1.0) / (len(ranks) - 1.0) - 0.5
    return output


def _daily_rank_ic(score: np.ndarray, future: np.ndarray) -> np.ndarray:
    result = np.full(len(score), np.nan)
    for index, (left, right) in enumerate(zip(score, future, strict=True)):
        active = np.isfinite(left) & np.isfinite(right)
        if np.sum(active) >= 3:
            left_rank = rankdata(left[active])
            right_rank = rankdata(right[active])
            if np.std(left_rank) > 1e-12 and np.std(right_rank) > 1e-12:
                result[index] = np.corrcoef(left_rank, right_rank)[0, 1]
    return result


def _weights(score: np.ndarray) -> np.ndarray:
    finite = np.isfinite(score)
    active_count = finite.sum(axis=1, keepdims=True)
    row_mean = np.divide(
        np.where(finite, score, 0.0).sum(axis=1, keepdims=True),
        active_count,
        out=np.zeros((score.shape[0], 1), dtype=float),
        where=active_count > 0,
    )
    centered = np.where(finite, score - row_mean, np.nan)
    denominator = np.nansum(np.abs(centered), axis=1, keepdims=True)
    numerator = np.where(finite, centered, 0.0)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(centered),
        where=(denominator > 1e-12) & finite,
    )


def cohort_portfolio_path(
    weights: np.ndarray,
    forward_return: np.ndarray,
    holding_period: int,
    one_way_cost_bps: float,
) -> np.ndarray:
    """Build a daily capital path from overlapping fixed-horizon labels.

    A one-period strategy is charged on the full traded notional. For a longer
    horizon, one H-th of capital starts a new cohort each day; every cohort pays
    entry and exit cost and realizes its return only when it exits.
    """

    weight = np.asarray(weights, dtype=float)
    future = np.asarray(forward_return, dtype=float)
    if weight.shape != future.shape:
        raise ValueError("weights and forward_return must have identical geometry")
    horizon = max(1, int(holding_period))
    cost = float(one_way_cost_bps) / 10_000.0
    valid = np.any(np.isfinite(weight) & np.isfinite(future), axis=1)
    gross = np.nansum(weight * future, axis=1)
    output = np.full(len(weight), np.nan)
    if horizon == 1:
        previous = np.vstack((np.zeros((1, weight.shape[1])), weight[:-1]))
        traded_notional = np.sum(np.abs(weight - previous), axis=1)
        output[valid] = gross[valid] - cost * traded_notional[valid]
        return output

    gross_exposure = np.sum(np.abs(weight), axis=1)
    cohort_net = (gross - 2.0 * cost * gross_exposure) / horizon
    source = np.flatnonzero(valid & (np.arange(len(weight)) + horizon < len(weight)))
    output[source + horizon] = cohort_net[source]
    return output


def _max_drawdown(path: np.ndarray) -> float:
    wealth = np.cumprod(1.0 + np.clip(np.nan_to_num(path), -0.999999, None))
    peak = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
    return float(np.max(1.0 - wealth / peak)) if len(wealth) else 0.0


def _time_folds(
    indices: np.ndarray,
    minimum_train_fraction: float = 0.40,
    folds: int = 4,
    purge_steps: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if len(indices) < 20:
        raise ValueError("development interval is too short for cross-fitting")
    first = max(int(len(indices) * minimum_train_fraction), 5)
    boundaries = np.linspace(first, len(indices), folds + 1, dtype=int)
    result = []
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        if stop > start:
            train_stop = max(0, start - purge_steps)
            if train_stop:
                result.append((indices[:train_stop], indices[start:stop]))
    return result


def evaluate_candidate(
    candidate_id: str,
    values: np.ndarray,
    future_return: np.ndarray,
    timestamps: pd.DatetimeIndex,
    development_end: str,
    periods_per_year: int,
    one_way_cost_bps: float,
    quantiles: int,
    purge_steps: int = 0,
) -> CandidateEvidence:
    raw = np.asarray(values, dtype=float)
    future = np.asarray(future_return, dtype=float)
    if raw.shape != future.shape or raw.shape[0] != len(timestamps):
        raise ValueError("candidate, label and timestamp geometry must match")
    rank = cross_sectional_ranks(raw)
    development = np.flatnonzero(timestamps <= pd.Timestamp(development_end, tz="UTC"))
    if purge_steps:
        development = development[:-purge_steps]
    folds = _time_folds(development, purge_steps=purge_steps)
    oof_score = np.full_like(rank, np.nan)
    oof_path = np.full(len(rank), np.nan)
    fold_directions: list[int] = []
    fold_means: list[float] = []
    for train, evaluation in folds:
        train_ic = _daily_rank_ic(rank[train], future[train])
        finite_train_ic = train_ic[np.isfinite(train_ic)]
        if not len(finite_train_ic):
            raise ValueError("candidate has no finite training-fold rank IC")
        direction = 1 if np.mean(finite_train_ic) >= 0.0 else -1
        fold_directions.append(direction)
        signed = direction * rank[evaluation]
        oof_score[evaluation] = signed
        weights = _weights(signed)
        net = cohort_portfolio_path(
            weights,
            future[evaluation],
            max(1, purge_steps),
            one_way_cost_bps,
        )
        oof_path[evaluation] = net
        fold_means.append(float(np.mean(net)))
    full_ic = _daily_rank_ic(rank[development], future[development])
    finite_full_ic = full_ic[np.isfinite(full_ic)]
    if not len(finite_full_ic):
        raise ValueError("candidate has no finite development rank IC")
    final_direction = 1 if np.mean(finite_full_ic) >= 0.0 else -1
    score = final_direction * rank
    active_path = oof_path[np.isfinite(oof_path)]
    if not len(active_path):
        raise ValueError("candidate has no finite out-of-fold return path")
    oof_daily_ic = _daily_rank_ic(oof_score, future)
    finite_oof_ic = oof_daily_ic[np.isfinite(oof_daily_ic)]
    if not len(finite_oof_ic):
        raise ValueError("candidate has no finite out-of-fold rank IC")
    mean = float(np.mean(active_path))
    standard_deviation = (
        float(np.std(active_path, ddof=1)) if len(active_path) > 1 else 0.0
    )
    annual_return = float(
        np.expm1(
            np.mean(np.log1p(np.clip(active_path, -0.999999, None))) * periods_per_year
        )
    )
    sharpe = (
        mean / standard_deviation * math.sqrt(periods_per_year)
        if standard_deviation > 0.0
        else 0.0
    )
    drawdown = _max_drawdown(active_path)
    calmar = annual_return / drawdown if drawdown > 0.0 else 0.0
    hac = newey_west_mean_test(active_path, lag=max(1, purge_steps))
    p_value = hac.one_sided_p_value
    bootstrap_low, bootstrap_high = moving_block_bootstrap_mean_interval(
        active_path,
        block_length=max(2, purge_steps),
        repetitions=500,
        seed=int(hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:8], 16),
    )
    robustness_seed = int(
        hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:8], 16
    )
    stationary_low, stationary_high = stationary_bootstrap_mean_interval(
        active_path,
        mean_block_length=max(2, purge_steps),
        repetitions=500,
        seed=robustness_seed,
    )
    sign_flip_p = block_sign_flip_mean_p_value(
        active_path,
        block_length=max(2, purge_steps),
        repetitions=1_000,
        seed=robustness_seed,
    )
    shape = payoff_shape(oof_score, future, quantiles)
    metrics = {
        "oof_observations": int(len(active_path)),
        "direction": final_direction,
        "direction_fold_agreement": float(
            np.mean(np.asarray(fold_directions) == final_direction)
        ),
        "rank_ic_mean": float(np.mean(finite_oof_ic)),
        "annual_return": annual_return,
        "sharpe": sharpe,
        "max_drawdown": drawdown,
        "calmar": calmar,
        "positive_fold_fraction": float(np.mean(np.asarray(fold_means) > 0.0)),
        "worst_fold_mean": float(np.min(fold_means)),
        "purge_steps": int(purge_steps),
        "hac_mean_t": hac.t_statistic,
        "hac_mean_standard_error": hac.standard_error,
        "hac_lag": hac.lag,
        "block_bootstrap_mean_low": bootstrap_low,
        "block_bootstrap_mean_high": bootstrap_high,
        "stationary_bootstrap_mean_low": stationary_low,
        "stationary_bootstrap_mean_high": stationary_high,
        "block_sign_flip_p_value": sign_flip_p,
        "effective_sample_size": autocorrelation_effective_sample_size(active_path),
        "top_5_percent_profit_concentration": profit_concentration(active_path),
        "mean_after_removing_best_event": leave_best_event_mean(active_path),
        "payoff_shape": shape,
    }
    return CandidateEvidence(
        candidate_id, final_direction, raw, rank, score, oof_path, metrics, p_value
    )
