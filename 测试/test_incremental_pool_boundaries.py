from __future__ import annotations

import numpy as np
import pytest

from factorforge.selection.incremental_pool import (
    PoolPolicy,
    _finite_mean,
    _positive_economic_dimensions,
)


def _policy(**overrides: float | int | bool) -> PoolPolicy:
    values: dict[str, float | int | bool] = {
        "target_size": 10,
        "periods_per_year": 252,
        "minimum_positive_fold_fraction": 0.75,
        "minimum_rank_ic": 0.0,
        "maximum_return_correlation": 0.7,
        "fdr_alpha": 0.1,
        "fdr_gate_enabled": False,
        "confirmation_fraction": 0.25,
        "maximum_value_rank_correlation": 0.8,
        "maximum_reconstruction_r2": 0.8,
    }
    values.update(overrides)
    return PoolPolicy(**values)  # type: ignore[arg-type]


def test_finite_mean_handles_empty_evidence_without_warning() -> None:
    assert _finite_mean(np.array([np.nan, np.inf, -np.inf])) == 0.0
    assert _finite_mean(np.array([1.0, np.nan, 3.0])) == 2.0


def test_redundancy_diagnostics_do_not_count_as_positive_pool_increments() -> None:
    increments = {
        "annual_return": -0.01,
        "sharpe": -0.1,
        "calmar": -0.1,
        "drawdown_reduction": -0.01,
        "worst_block": -0.001,
        "confirmation_mean": -0.001,
        "confirmation_sharpe": -0.1,
        "maximum_value_rank_correlation": 0.7,
        "reconstruction_r2": 0.7,
    }
    assert _positive_economic_dimensions(increments) == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target_size", 0),
        ("periods_per_year", 0),
        ("confirmation_fraction", 1.1),
        ("maximum_return_correlation", -0.1),
    ],
)
def test_pool_policy_rejects_invalid_boundaries(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        _policy(**{field: value}).validate()
