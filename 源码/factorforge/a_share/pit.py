from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd


@dataclass(frozen=True)
class PitJoinReport:
    rows: int
    matched_rows: int
    unmatched_rows: int
    future_rows: int


def point_in_time_join(
    decisions: pd.DataFrame,
    observations: pd.DataFrame,
    value_columns: Sequence[str],
    *,
    symbol_column: str = "symbol",
    decision_time_column: str = "decision_time",
    available_time_column: str = "available_at",
    observation_time_column: str | None = None,
) -> tuple[pd.DataFrame, PitJoinReport]:
    """Attach only the latest observation that was available at each decision."""

    required_left = {symbol_column, decision_time_column}
    required_right = {symbol_column, available_time_column, *value_columns}
    missing_left = sorted(required_left - set(decisions))
    missing_right = sorted(required_right - set(observations))
    if missing_left or missing_right:
        raise ValueError(
            f"PIT join missing fields: decisions={missing_left}, "
            f"observations={missing_right}"
        )
    left = decisions.copy()
    right = observations.copy()
    left[decision_time_column] = pd.to_datetime(left[decision_time_column], utc=True)
    right[available_time_column] = pd.to_datetime(right[available_time_column], utc=True)
    if observation_time_column:
        right[observation_time_column] = pd.to_datetime(
            right[observation_time_column], utc=True
        )
        invalid = right[available_time_column] < right[observation_time_column]
        if invalid.any():
            raise ValueError("observation available_at precedes its event time")
    left["__row_order"] = range(len(left))
    left = left.sort_values([decision_time_column, symbol_column], kind="stable")
    right = right.sort_values([available_time_column, symbol_column], kind="stable")
    keep = [symbol_column, available_time_column, *value_columns]
    joined = pd.merge_asof(
        left,
        right.loc[:, keep],
        left_on=decision_time_column,
        right_on=available_time_column,
        by=symbol_column,
        direction="backward",
        allow_exact_matches=True,
    ).sort_values("__row_order")
    future = joined[available_time_column] > joined[decision_time_column]
    future_count = int(future.fillna(False).sum())
    if future_count:
        raise AssertionError("PIT join produced future observations")
    matched = int(joined[available_time_column].notna().sum())
    joined = joined.drop(columns="__row_order").reset_index(drop=True)
    return joined, PitJoinReport(len(joined), matched, len(joined) - matched, future_count)


def resolve_symbol_history(
    frame: pd.DataFrame,
    security_master: pd.DataFrame,
    *,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    """Map changing tickers to a permanent id using effective intervals."""

    required = {"symbol", "permanent_id", "effective_from"}
    missing = sorted(required - set(security_master))
    if missing:
        raise ValueError(f"security master missing fields: {missing}")
    local = frame.copy()
    local[timestamp_column] = pd.to_datetime(local[timestamp_column], utc=True)
    master = security_master.copy()
    master["effective_from"] = pd.to_datetime(master["effective_from"], utc=True)
    if "effective_to" not in master:
        master["effective_to"] = pd.NaT
    master["effective_to"] = pd.to_datetime(master["effective_to"], utc=True)
    if "available_at" in master:
        master = master.rename(columns={"available_at": "__master_available_at"})
    merged = local.merge(master, on="symbol", how="left", suffixes=("", "_master"))
    active = (merged[timestamp_column] >= merged["effective_from"]) & (
        merged["effective_to"].isna() | (merged[timestamp_column] < merged["effective_to"])
    )
    if "__master_available_at" in merged:
        merged["__master_available_at"] = pd.to_datetime(
            merged["__master_available_at"], utc=True
        )
        active &= merged["__master_available_at"] <= merged[timestamp_column]
    if "listing_at" in merged:
        merged["listing_at"] = pd.to_datetime(merged["listing_at"], utc=True)
        active &= merged[timestamp_column] >= merged["listing_at"]
    if "delisting_at" in merged:
        merged["delisting_at"] = pd.to_datetime(merged["delisting_at"], utc=True)
        active &= merged["delisting_at"].isna() | (
            merged[timestamp_column] <= merged["delisting_at"]
        )
    valid = merged[active].copy()
    duplicates = valid.duplicated([timestamp_column, "symbol"], keep=False)
    if duplicates.any():
        raise ValueError("overlapping symbol-master effective intervals")
    original_keys = local[[timestamp_column, "symbol"]].drop_duplicates()
    mapped_keys = valid[[timestamp_column, "symbol"]].drop_duplicates()
    if len(mapped_keys) != len(original_keys):
        raise ValueError("one or more rows have no active permanent-id mapping")
    return valid.reset_index(drop=True)
