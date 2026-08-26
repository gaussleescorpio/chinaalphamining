from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataQualityReport:
    rows: int
    assets: int
    timestamps: int
    duplicate_keys: int
    non_finite_values: int
    missing_fraction: float
    non_monotonic_assets: int
    inactive_rows: int
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def audit_long_frame(
    frame: pd.DataFrame,
    timestamp_column: str,
    asset_column: str,
    field_columns: Sequence[str],
    membership_column: str | None = None,
) -> DataQualityReport:
    required = {timestamp_column, asset_column, *field_columns}
    if membership_column:
        required.add(membership_column)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"input frame is missing fields: {missing}")
    local = frame.loc[:, list(required)].copy()
    time = pd.to_datetime(local[timestamp_column], utc=True, errors="coerce")
    duplicates = int(local.duplicated([timestamp_column, asset_column]).sum())
    numeric = local.loc[:, list(field_columns)].apply(pd.to_numeric, errors="coerce")
    array = numeric.to_numpy(dtype=float)
    non_finite = int(np.isinf(array).sum())
    missing_fraction = float(np.mean(~np.isfinite(array))) if array.size else 1.0
    non_monotonic = 0
    for _, group in local.assign(__time=time).groupby(asset_column, sort=False):
        series = group["__time"].dropna()
        non_monotonic += int(not series.is_monotonic_increasing)
    inactive = (
        int((~local[membership_column].fillna(False).astype(bool)).sum())
        if membership_column
        else 0
    )
    passed = (
        not duplicates
        and not non_finite
        and time.notna().all()
        and missing_fraction < 0.50
    )
    return DataQualityReport(
        rows=len(local),
        assets=int(local[asset_column].nunique()),
        timestamps=int(time.nunique()),
        duplicate_keys=duplicates,
        non_finite_values=non_finite,
        missing_fraction=missing_fraction,
        non_monotonic_assets=non_monotonic,
        inactive_rows=inactive,
        passed=bool(passed),
    )


def clean_long_frame(
    frame: pd.DataFrame,
    timestamp_column: str,
    asset_column: str,
    field_columns: Sequence[str],
) -> pd.DataFrame:
    """Apply deterministic, non-imputing sanitation; structural errors still fail closed."""

    local = frame.copy()
    local[timestamp_column] = pd.to_datetime(
        local[timestamp_column], utc=True, errors="raise"
    )
    if local.duplicated([timestamp_column, asset_column]).any():
        raise ValueError(
            "duplicate timestamp/asset keys require an explicit source-level resolution"
        )
    for field in field_columns:
        local[field] = pd.to_numeric(local[field], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
    return local.sort_values(
        [timestamp_column, asset_column], kind="stable"
    ).reset_index(drop=True)
