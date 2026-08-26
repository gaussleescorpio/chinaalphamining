from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import rankdata


def payoff_shape(
    score: np.ndarray, future_return: np.ndarray, quantiles: int = 10
) -> dict[str, Any]:
    x = np.asarray(score, dtype=float).reshape(-1)
    y = np.asarray(future_return, dtype=float).reshape(-1)
    active = np.isfinite(x) & np.isfinite(y)
    if np.sum(active) < quantiles * 4:
        return {"shape": "insufficient", "quantile_means": [], "monotonicity": None}
    ranks = rankdata(x[active], method="average")
    buckets = np.minimum(
        (ranks * quantiles / (len(ranks) + 1)).astype(int), quantiles - 1
    )
    means = np.asarray(
        [
            np.mean(y[active][buckets == index]) if np.any(buckets == index) else np.nan
            for index in range(quantiles)
        ]
    )
    finite = np.isfinite(means)
    monotonicity = (
        float(np.corrcoef(np.arange(quantiles)[finite], means[finite])[0, 1])
        if np.sum(finite) >= 3
        else 0.0
    )
    edge = float(0.5 * (np.nan_to_num(means[-1]) - np.nan_to_num(means[0])))
    middle_values = means[quantiles // 3 : -quantiles // 3 or None]
    middle = (
        float(np.nanmean(middle_values)) if np.any(np.isfinite(middle_values)) else 0.0
    )
    if abs(monotonicity) >= 0.70:
        shape = "monotonic"
    elif abs(edge) > 2.0 * abs(middle):
        shape = "tail"
    elif means[0] > middle and means[-1] > middle:
        shape = "u_shape"
    else:
        shape = "irregular"
    return {
        "shape": shape,
        "quantile_means": [
            float(value) if np.isfinite(value) else None for value in means
        ],
        "monotonicity": monotonicity,
        "long_short_edge": float(np.nan_to_num(means[-1]) - np.nan_to_num(means[0])),
    }
