from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

from factorforge.evaluation.crossfit import (
    _daily_rank_ic,
    _weights,
    cohort_portfolio_path,
    cross_sectional_ranks,
)


@dataclass(frozen=True)
class ScreeningEvidence:
    candidate_id: str
    direction: int
    metrics: dict[str, float | int]
    p_value: float

    @property
    def selection_key(self) -> tuple[float, ...]:
        return (
            float(self.metrics["positive_block_fraction"]),
            float(self.metrics["worst_block_mean"]),
            float(self.metrics["rank_ic_mean"]),
            float(self.metrics["tail_spread"]),
            float(self.metrics["mean_net_return"]),
        )


def screen_candidate(
    candidate_id: str,
    values: np.ndarray,
    future_return: np.ndarray,
    timestamps: pd.DatetimeIndex,
    development_end: str,
    purge_steps: int,
    stride: int,
    one_way_cost_bps: float,
) -> ScreeningEvidence:
    development = np.flatnonzero(timestamps <= pd.Timestamp(development_end, tz="UTC"))
    if purge_steps:
        development = development[:-purge_steps]
    sampled = development[::stride]
    if len(sampled) < 24:
        raise ValueError("development interval is too short for stage-one screening")
    ranks = cross_sectional_ranks(np.asarray(values, dtype=float)[sampled])
    future = np.asarray(future_return, dtype=float)[sampled]
    cut = max(8, int(len(sampled) * 0.60))
    train_ic = _daily_rank_ic(ranks[:cut], future[:cut])
    finite_train = train_ic[np.isfinite(train_ic)]
    if not len(finite_train):
        raise ValueError("candidate has no finite stage-one training IC")
    direction = 1 if np.mean(finite_train) >= 0.0 else -1
    score = direction * ranks[cut:]
    evaluation_future = future[cut:]
    daily_ic = _daily_rank_ic(score, evaluation_future)
    finite_ic = daily_ic[np.isfinite(daily_ic)]
    if len(finite_ic) < 4:
        raise ValueError("candidate has insufficient stage-one evaluation IC")
    weights = _weights(score)
    net = cohort_portfolio_path(
        weights,
        evaluation_future,
        max(1, purge_steps),
        one_way_cost_bps,
    )
    blocks = [block for block in np.array_split(net[np.isfinite(net)], 3) if len(block)]
    if len(blocks) < 3:
        raise ValueError("candidate has insufficient stage-one time blocks")
    active = np.isfinite(score) & np.isfinite(evaluation_future)
    flat_score = score[active]
    flat_future = evaluation_future[active]
    if len(flat_score) < 20:
        raise ValueError(
            "candidate has insufficient stage-one cross-sectional observations"
        )
    low, high = np.quantile(flat_score, [0.10, 0.90])
    tail_spread = float(
        np.mean(flat_future[flat_score >= high])
        - np.mean(flat_future[flat_score <= low])
    )
    test = ttest_1samp(finite_ic, 0.0, alternative="greater")
    p_value = float(test.pvalue) if np.isfinite(test.pvalue) else 1.0
    metrics = {
        "direction": direction,
        "sampled_timestamps": int(len(sampled)),
        "evaluation_timestamps": int(len(score)),
        "rank_ic_mean": float(np.mean(finite_ic)),
        "mean_net_return": float(np.mean(net)),
        "positive_block_fraction": float(
            np.mean([np.mean(block) > 0.0 for block in blocks])
        ),
        "worst_block_mean": float(min(np.mean(block) for block in blocks)),
        "tail_spread": tail_spread,
        "screening_p_value": p_value,
        "stride": int(stride),
        "purge_steps": int(purge_steps),
    }
    return ScreeningEvidence(candidate_id, direction, metrics, p_value)
