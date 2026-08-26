from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from factorforge.evaluation.label_blind import audit_values
from factorforge.evaluation.multiple_testing import benjamini_hochberg
from factorforge.monitoring import assess_decay
from factorforge.evaluation.crossfit import (
    _time_folds,
    _weights,
    cohort_portfolio_path,
    evaluate_candidate,
)


def test_label_blind_numeric_checks_do_not_need_returns() -> None:
    values = np.arange(60, dtype=float).reshape(10, 6)
    result = audit_values(values, 0.90)
    assert result.accepted
    assert len(result.rank_path_sha256) == 64


def test_label_blind_empty_rows_are_rejected_without_runtime_warning() -> None:
    values = np.full((4, 3), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = audit_values(values, 0.90)
    assert not result.accepted
    assert result.reason_code == "INSUFFICIENT_COVERAGE"


def test_bh_is_monotone_in_pvalue_order() -> None:
    p = np.array([0.04, 0.001, 0.03, 0.20])
    q = benjamini_hochberg(p)
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12)
    assert np.all((q >= 0.0) & (q <= 1.0))


def test_decay_monitor_has_three_clear_states() -> None:
    normal = assess_decay(np.r_[np.full(150, 0.001), np.full(63, 0.0008)])
    paused = assess_decay(np.r_[np.full(150, 0.001), np.full(63, -0.001)])
    assert normal.state == "NORMAL"
    assert paused.state == "PAUSE"


def test_time_folds_purge_label_overlap() -> None:
    folds = _time_folds(np.arange(100), purge_steps=5)
    assert folds
    assert all(train[-1] <= evaluation[0] - 6 for train, evaluation in folds)


def test_candidate_records_purge_rule() -> None:
    rng = np.random.default_rng(4)
    values = rng.normal(size=(120, 6))
    future = 0.01 * values + rng.normal(scale=0.001, size=values.shape)
    timestamps = pd.date_range("2020-01-01", periods=120, freq="D", tz="UTC")
    evidence = evaluate_candidate(
        "x", values, future, timestamps, "2020-04-30", 252, 1.0, 5, purge_steps=3
    )
    assert evidence.metrics["purge_steps"] == 3


def test_multi_period_labels_use_funded_cohorts_and_round_trip_cost() -> None:
    weights = np.tile(np.array([[0.5, -0.5]]), (12, 1))
    forward = np.tile(np.array([[0.10, 0.00]]), (12, 1))
    path = cohort_portfolio_path(weights, forward, 5, 10.0)
    active = path[np.isfinite(path)]
    # Gross cohort contribution is 5% / 5; entry and exit each cost 10bp.
    assert np.allclose(active, (0.05 - 0.002) / 5)


def test_one_period_cost_uses_full_traded_notional() -> None:
    weights = np.array([[0.5, -0.5], [-0.5, 0.5]])
    future = np.zeros_like(weights)
    path = cohort_portfolio_path(weights, future, 1, 10.0)
    assert np.allclose(path, [-0.001, -0.002])


def test_sparse_assets_have_zero_weight_and_finite_oof_path() -> None:
    rng = np.random.default_rng(24)
    values = rng.normal(size=(120, 8))
    values[::3, 0] = np.nan
    values[1::4, 3] = np.nan
    future = 0.01 * np.nan_to_num(values) + rng.normal(
        scale=0.001, size=values.shape
    )
    future[~np.isfinite(values)] = np.nan
    timestamps = pd.date_range("2020-01-01", periods=120, freq="D", tz="UTC")

    weights = _weights(values)
    assert np.isfinite(weights).all()
    assert np.all(weights[~np.isfinite(values)] == 0.0)
    active_rows = np.isfinite(values).sum(axis=1) >= 2
    assert np.allclose(np.sum(np.abs(weights[active_rows]), axis=1), 1.0)
    turnover = 0.5 * np.sum(
        np.abs(weights - np.vstack((np.zeros((1, 8)), weights[:-1]))), axis=1
    )
    assert np.isfinite(turnover).all()

    evidence = evaluate_candidate(
        "sparse",
        values,
        future,
        timestamps,
        "2020-04-30",
        252,
        1.0,
        5,
        purge_steps=3,
    )
    assert np.isfinite(evidence.oof_return_path).any()
