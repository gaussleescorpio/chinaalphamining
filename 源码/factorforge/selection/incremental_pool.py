from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from factorforge.evaluation.crossfit import CandidateEvidence


@dataclass(frozen=True)
class PoolPolicy:
    target_size: int
    periods_per_year: int
    minimum_positive_fold_fraction: float
    minimum_rank_ic: float
    maximum_return_correlation: float
    fdr_alpha: float
    fdr_gate_enabled: bool
    confirmation_fraction: float
    maximum_value_rank_correlation: float
    maximum_reconstruction_r2: float
    development_tail_selection_enabled: bool = False

    def validate(self) -> None:
        """Fail early when a pool configuration has impossible boundaries."""

        if self.target_size < 1:
            raise ValueError("target_size must be positive")
        if self.periods_per_year < 1:
            raise ValueError("periods_per_year must be positive")
        bounded = {
            "minimum_positive_fold_fraction": self.minimum_positive_fold_fraction,
            "fdr_alpha": self.fdr_alpha,
            "confirmation_fraction": self.confirmation_fraction,
            "maximum_return_correlation": self.maximum_return_correlation,
            "maximum_value_rank_correlation": self.maximum_value_rank_correlation,
        }
        for name, value in bounded.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")


@dataclass(frozen=True)
class PoolDecision:
    candidate_id: str
    accepted: bool
    reason_code: str
    step: int
    increments: Mapping[str, float]


def _drawdown(path: np.ndarray) -> float:
    wealth = np.cumprod(1.0 + np.clip(np.nan_to_num(path), -0.999999, None))
    peak = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
    return float(np.max(1.0 - wealth / peak)) if len(wealth) else 0.0


def _path_metrics(path: np.ndarray, periods_per_year: int) -> dict[str, float]:
    active = np.asarray(path, dtype=float)
    active = active[np.isfinite(active)]
    if len(active) < 3:
        return {
            "annual_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "worst_block": 0.0,
        }
    annual = float(
        np.expm1(np.mean(np.log1p(np.clip(active, -0.999999, None))) * periods_per_year)
    )
    std = float(np.std(active, ddof=1))
    sharpe = (
        float(np.mean(active) / std * math.sqrt(periods_per_year)) if std > 0.0 else 0.0
    )
    drawdown = _drawdown(active)
    blocks = [block for block in np.array_split(active, 4) if len(block)]
    return {
        "annual_return": annual,
        "sharpe": sharpe,
        "max_drawdown": drawdown,
        "calmar": annual / drawdown if drawdown > 0.0 else 0.0,
        "worst_block": float(min(np.mean(block) for block in blocks)),
    }


def _combine(paths: Sequence[np.ndarray]) -> np.ndarray:
    if not paths:
        return np.array([], dtype=float)
    matrix = np.column_stack(paths)
    finite = np.isfinite(matrix)
    count = np.sum(finite, axis=1)
    total = np.nansum(matrix, axis=1)
    return np.divide(total, count, out=np.full(len(matrix), np.nan), where=count > 0)


def _split_path(
    path: np.ndarray, confirmation_fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    active = np.flatnonzero(np.isfinite(path))
    cut = max(1, min(len(active) - 1, int(len(active) * (1.0 - confirmation_fraction))))
    selection = np.full(len(path), np.nan)
    confirmation = np.full(len(path), np.nan)
    selection[active[:cut]] = path[active[:cut]]
    confirmation[active[cut:]] = path[active[cut:]]
    return selection, confirmation


def _finite_mean(path: np.ndarray) -> float:
    """Mean of finite observations, or zero for an empty evidence segment."""

    active = np.asarray(path, dtype=float)
    active = active[np.isfinite(active)]
    return float(np.mean(active)) if len(active) else 0.0


def _positive_economic_dimensions(
    increments: Mapping[str, float], include_development_tail: bool = False
) -> int:
    economic_metrics = [
        "annual_return",
        "sharpe",
        "calmar",
        "drawdown_reduction",
        "worst_block",
    ]
    if include_development_tail:
        economic_metrics.extend(("confirmation_mean", "confirmation_sharpe"))
    return sum(increments[name] > 0.0 for name in economic_metrics)


def _rank_novelty(
    candidate: CandidateEvidence, selected: Sequence[CandidateEvidence]
) -> tuple[float, float]:
    if not selected:
        return 0.0, 0.0
    time_mask = np.isfinite(candidate.oof_return_path)
    y = np.nan_to_num(candidate.rank_values[time_mask], nan=0.0).reshape(-1)
    columns = [
        np.nan_to_num(item.rank_values[time_mask], nan=0.0).reshape(-1)
        for item in selected
    ]
    correlations = []
    for column in columns:
        correlations.append(
            abs(float(np.corrcoef(y, column)[0, 1]))
            if np.std(column) > 1e-12 and np.std(y) > 1e-12
            else 1.0
        )
    matrix = np.column_stack(columns)
    cut = max(10, int(len(y) * 0.60))
    if cut >= len(y) - 2:
        return max(correlations), 1.0
    train_x, test_x = matrix[:cut], matrix[cut:]
    train_y, test_y = y[:cut], y[cut:]
    gram = train_x.T @ train_x + 1e-4 * np.eye(train_x.shape[1])
    coefficient = np.linalg.solve(gram, train_x.T @ train_y)
    prediction = test_x @ coefficient
    denominator = float(np.sum((test_y - np.mean(test_y)) ** 2))
    r2 = (
        1.0 - float(np.sum((test_y - prediction) ** 2)) / denominator
        if denominator > 1e-12
        else 1.0
    )
    return max(correlations), r2


class IncrementalPoolSelector:
    """Build an empty pool one factor at a time using fixed incremental objectives."""

    def __init__(self, policy: PoolPolicy):
        policy.validate()
        self.policy = policy

    def select(
        self,
        candidates: Sequence[CandidateEvidence],
        q_values: Mapping[str, float],
    ) -> tuple[list[CandidateEvidence], list[PoolDecision]]:
        """Select an incremental pool using development cross-fit evidence only."""

        eligible: list[CandidateEvidence] = []
        decisions: list[PoolDecision] = []
        for evidence in candidates:
            metrics = evidence.metrics
            failures = []
            if (
                metrics["positive_fold_fraction"]
                < self.policy.minimum_positive_fold_fraction
            ):
                failures.append("UNSTABLE_TIME_BLOCKS")
            if metrics["rank_ic_mean"] < self.policy.minimum_rank_ic:
                failures.append("INSUFFICIENT_RANK_INFORMATION")
            if (
                self.policy.fdr_gate_enabled
                and q_values[evidence.candidate_id] > self.policy.fdr_alpha
            ):
                failures.append("FDR_NOT_PASSED")
            if failures:
                decisions.append(
                    PoolDecision(
                        evidence.candidate_id, False, "+".join(failures), 0, {}
                    )
                )
            else:
                eligible.append(evidence)

        selected: list[CandidateEvidence] = []
        remaining = list(eligible)
        while remaining and len(selected) < self.policy.target_size:
            baseline_path = _combine([item.oof_return_path for item in selected])
            baseline_selection, baseline_confirmation = (
                _split_path(baseline_path, self.policy.confirmation_fraction)
                if selected
                else (baseline_path, baseline_path)
            )
            baseline = _path_metrics(baseline_selection, self.policy.periods_per_year)
            baseline_confirm = _path_metrics(
                baseline_confirmation, self.policy.periods_per_year
            )
            scored: list[
                tuple[tuple[float, ...], CandidateEvidence, dict[str, float]]
            ] = []
            for candidate in remaining:
                if selected:
                    correlations = []
                    for current in selected:
                        active = np.isfinite(candidate.oof_return_path) & np.isfinite(
                            current.oof_return_path
                        )
                        left = candidate.oof_return_path[active]
                        right = current.oof_return_path[active]
                        if (
                            np.sum(active) < 3
                            or np.std(left) <= 1e-12
                            or np.std(right) <= 1e-12
                        ):
                            correlations.append(1.0)
                        else:
                            correlations.append(
                                abs(float(np.corrcoef(left, right)[0, 1]))
                            )
                    if max(correlations) > self.policy.maximum_return_correlation:
                        continue
                    value_correlation, reconstruction_r2 = _rank_novelty(
                        candidate, selected
                    )
                    if value_correlation > self.policy.maximum_value_rank_correlation:
                        continue
                    if reconstruction_r2 > self.policy.maximum_reconstruction_r2:
                        continue
                else:
                    value_correlation, reconstruction_r2 = 0.0, 0.0
                new_path = _combine(
                    [
                        *(item.oof_return_path for item in selected),
                        candidate.oof_return_path,
                    ]
                )
                new_selection, new_confirmation = _split_path(
                    new_path, self.policy.confirmation_fraction
                )
                updated = _path_metrics(new_selection, self.policy.periods_per_year)
                confirmed = _path_metrics(
                    new_confirmation, self.policy.periods_per_year
                )
                increments = {
                    "annual_return": updated["annual_return"]
                    - baseline["annual_return"],
                    "sharpe": updated["sharpe"] - baseline["sharpe"],
                    "calmar": updated["calmar"] - baseline["calmar"],
                    "drawdown_reduction": baseline["max_drawdown"]
                    - updated["max_drawdown"],
                    "worst_block": updated["worst_block"] - baseline["worst_block"],
                    "confirmation_mean": (
                        _finite_mean(new_confirmation)
                        - _finite_mean(baseline_confirmation)
                        if selected
                        else _finite_mean(new_confirmation)
                    ),
                    "confirmation_sharpe": confirmed["sharpe"]
                    - baseline_confirm["sharpe"],
                    "maximum_value_rank_correlation": value_correlation,
                    "reconstruction_r2": reconstruction_r2,
                }
                positive_dimensions = _positive_economic_dimensions(
                    increments, self.policy.development_tail_selection_enabled
                )
                key = (
                    float(positive_dimensions),
                    (
                        increments["confirmation_mean"]
                        if self.policy.development_tail_selection_enabled
                        else 0.0
                    ),
                    increments["worst_block"],
                    increments["sharpe"],
                    increments["calmar"],
                    increments["annual_return"],
                    -candidate.metrics["max_drawdown"],
                )
                scored.append((key, candidate, increments))
            if not scored:
                break
            scored.sort(key=lambda row: (row[0], row[1].candidate_id), reverse=True)
            key, winner, increments = scored[0]
            if (
                self.policy.development_tail_selection_enabled
                and increments["confirmation_mean"] <= 0.0
            ):
                break
            if selected and (
                key[0] < 3
                or increments["worst_block"] < 0.0
                ):
                break
            selected.append(winner)
            remaining.remove(winner)
            decisions.append(
                PoolDecision(
                    winner.candidate_id,
                    True,
                    "INCREMENTAL_POOL_ACCEPT",
                    len(selected),
                    increments,
                )
            )

        selected_ids = {item.candidate_id for item in selected}
        decided_ids = {item.candidate_id for item in decisions}
        for candidate in eligible:
            if (
                candidate.candidate_id not in selected_ids
                and candidate.candidate_id not in decided_ids
            ):
                decisions.append(
                    PoolDecision(
                        candidate.candidate_id,
                        False,
                        "NO_POSITIVE_POOL_INCREMENT",
                        0,
                        {},
                    )
                )
        return selected, decisions
