from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from factorforge.evaluation.crossfit import CandidateEvidence, evaluate_candidate


def _block_permute(values: np.ndarray, block_length: int, seed: int) -> np.ndarray:
    source = np.asarray(values, dtype=float)
    blocks = [
        np.arange(start, min(start + block_length, len(source)))
        for start in range(0, len(source), block_length)
    ]
    order = np.random.default_rng(seed).permutation(len(blocks))
    return np.concatenate([source[blocks[index]] for index in order], axis=0)[
        : len(source)
    ]


def selected_factor_negative_controls(
    evidence: CandidateEvidence,
    future_return: np.ndarray,
    timestamps: pd.DatetimeIndex,
    development_end: str,
    periods_per_year: int,
    one_way_cost_bps: float,
    quantiles: int,
    purge_steps: int,
) -> list[dict[str, object]]:
    """Run deterministic label, time and asset-assignment controls."""

    seed = int(
        hashlib.sha256(evidence.candidate_id.encode("utf-8")).hexdigest()[:8], 16
    )
    lag = max(1, purge_steps * 3)
    delayed = np.full_like(evidence.raw_values, np.nan)
    delayed[lag:] = evidence.raw_values[:-lag]
    controls = {
        "TRAINING_LABEL_BLOCK_PERMUTATION": (
            evidence.raw_values,
            _block_permute(future_return, max(5, purge_steps), seed),
        ),
        "FEATURE_DELAY": (delayed, future_return),
        "ASSET_EVENT_MISMATCH": (
            np.roll(evidence.raw_values, 1, axis=1),
            future_return,
        ),
    }
    rows: list[dict[str, object]] = []
    for name, (values, labels) in controls.items():
        try:
            result = evaluate_candidate(
                evidence.candidate_id,
                values,
                labels,
                timestamps,
                development_end,
                periods_per_year,
                one_way_cost_bps,
                quantiles,
                purge_steps,
            )
            rows.append(
                {
                    "candidate_id": evidence.candidate_id,
                    "control": name,
                    "rank_ic_mean": result.metrics["rank_ic_mean"],
                    "annual_return": result.metrics["annual_return"],
                    "hac_p_value": result.p_value,
                    "status": "COMPUTED",
                }
            )
        except ValueError as error:
            rows.append(
                {
                    "candidate_id": evidence.candidate_id,
                    "control": name,
                    "rank_ic_mean": np.nan,
                    "annual_return": np.nan,
                    "hac_p_value": 1.0,
                    "status": f"INVALID:{error}",
                }
            )
    return rows
