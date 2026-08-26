from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from .contracts import AShareResidualConfig


@dataclass(frozen=True)
class CrossFitResult:
    predictions: pd.Series
    residuals: pd.Series
    fold_table: pd.DataFrame


def build_cross_fitted_residual_correction(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    baseline_prediction_column: str,
    date_column: str = "date",
    config: AShareResidualConfig | None = None,
) -> CrossFitResult:
    """Fit residual corrections on earlier out-of-time baseline errors only."""
    config = config or AShareResidualConfig()
    config.validate()
    columns = [date_column, *feature_columns, target_column, baseline_prediction_column]
    work = frame.loc[:, columns].copy()
    work[date_column] = pd.to_datetime(work[date_column]).dt.normalize()
    work = work.sort_values(date_column, kind="stable")
    predictions = pd.Series(np.nan, index=work.index, dtype=float, name="corrected_prediction")
    fold_rows: list[dict[str, object]] = []
    blocks = _date_blocks(work[date_column], count=6)
    for fold, validation_dates in enumerate(blocks[2:], start=2):
        validation_start = validation_dates.min()
        earlier = pd.DatetimeIndex(sorted(work.loc[work[date_column] < validation_start, date_column].unique()))
        if len(earlier) <= config.purge_sessions:
            continue
        training_dates = earlier[:-config.purge_sessions]
        train = work[date_column].isin(training_dates)
        valid = work[date_column].isin(validation_dates)
        train &= work[[target_column, baseline_prediction_column]].notna().all(axis=1)
        valid &= work[baseline_prediction_column].notna()
        if train.sum() < 200 or valid.sum() == 0:
            continue
        residual_target = work.loc[train, target_column] - work.loc[train, baseline_prediction_column]
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", RobustScaler(quantile_range=(10.0, 90.0))),
            ("ridge", Ridge(alpha=config.ridge_alpha)),
        ])
        model.fit(work.loc[train, feature_columns], residual_target)
        correction = model.predict(work.loc[valid, feature_columns])
        predictions.loc[valid] = work.loc[valid, baseline_prediction_column].to_numpy() + correction
        fold_rows.append({
            "fold": fold,
            "train_start": training_dates.min(), "train_end": training_dates.max(),
            "validation_start": validation_dates.min(), "validation_end": validation_dates.max(),
            "train_rows": int(train.sum()), "validation_rows": int(valid.sum()),
            "purge_sessions": config.purge_sessions,
        })
    residuals = (work[target_column] - predictions).rename("corrected_cross_fitted_residual")
    return CrossFitResult(predictions, residuals, pd.DataFrame(fold_rows))


def _date_blocks(dates: pd.Series, count: int = 5) -> list[pd.DatetimeIndex]:
    unique = pd.DatetimeIndex(sorted(pd.to_datetime(dates).dt.normalize().unique()))
    return [pd.DatetimeIndex(block) for block in np.array_split(unique, count) if len(block)]


def build_cross_fitted_residual_target(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    date_column: str = "date",
    config: AShareResidualConfig | None = None,
) -> CrossFitResult:
    """Create out-of-time residuals; no row is scored by a model trained on it.

    The expanding folds use only earlier dates.  A purge equal to entry offset
    plus holding period separates training labels from validation decisions.
    """

    config = config or AShareResidualConfig()
    config.validate()
    required = {date_column, target_column, *feature_columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"residual frame missing columns: {missing}")
    work = frame.loc[:, [date_column, *feature_columns, target_column]].copy()
    work[date_column] = pd.to_datetime(work[date_column]).dt.normalize()
    work = work.sort_values(date_column, kind="stable")
    predictions = pd.Series(np.nan, index=work.index, dtype=float, name="baseline_prediction")
    fold_rows: list[dict[str, object]] = []
    blocks = _date_blocks(work[date_column], count=6)
    for fold, validation_dates in enumerate(blocks[1:], start=1):
        validation_start = validation_dates.min()
        earlier = pd.DatetimeIndex(sorted(work.loc[work[date_column] < validation_start, date_column].unique()))
        if len(earlier) <= config.purge_sessions:
            continue
        training_dates = earlier[:-config.purge_sessions]
        train = work[date_column].isin(training_dates) & np.isfinite(work[target_column])
        valid = work[date_column].isin(validation_dates) & np.isfinite(work[target_column])
        if train.sum() < 200 or valid.sum() == 0:
            continue
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", RobustScaler(quantile_range=(10.0, 90.0))),
            ("ridge", Ridge(alpha=config.ridge_alpha)),
        ])
        model.fit(work.loc[train, feature_columns], work.loc[train, target_column])
        predictions.loc[valid] = model.predict(work.loc[valid, feature_columns])
        fold_rows.append({
            "fold": fold,
            "train_start": training_dates.min(), "train_end": training_dates.max(),
            "validation_start": validation_dates.min(), "validation_end": validation_dates.max(),
            "train_rows": int(train.sum()), "validation_rows": int(valid.sum()),
            "purge_sessions": config.purge_sessions,
        })
    residuals = (work[target_column] - predictions).rename("cross_fitted_residual")
    return CrossFitResult(predictions, residuals, pd.DataFrame(fold_rows))
