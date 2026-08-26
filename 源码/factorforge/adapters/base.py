from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from factorforge.contracts import PanelData


@dataclass(frozen=True)
class LongFrameAdapter:
    """Convert a long table into a deterministic time-by-asset panel."""

    timestamp_column: str
    asset_column: str
    field_columns: Sequence[str]
    membership_column: str | None = None

    def materialize(self, frame: pd.DataFrame) -> PanelData:
        required = {self.timestamp_column, self.asset_column, *self.field_columns}
        if self.membership_column:
            required.add(self.membership_column)
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"input frame is missing fields: {missing}")
        local = frame.loc[:, sorted(required)].copy()
        local[self.timestamp_column] = pd.to_datetime(
            local[self.timestamp_column], utc=True
        )
        if local.duplicated([self.timestamp_column, self.asset_column]).any():
            raise ValueError("input contains duplicate timestamp/asset rows")
        timestamps = pd.DatetimeIndex(sorted(local[self.timestamp_column].unique()))
        assets = tuple(sorted(local[self.asset_column].astype(str).unique()))
        fields: dict[str, np.ndarray] = {}
        for field in self.field_columns:
            matrix = local.pivot(
                index=self.timestamp_column, columns=self.asset_column, values=field
            )
            matrix = matrix.reindex(index=timestamps, columns=assets)
            fields[field] = matrix.to_numpy(dtype=np.float64)
        if self.membership_column:
            membership = (
                local.pivot(
                    index=self.timestamp_column,
                    columns=self.asset_column,
                    values=self.membership_column,
                )
                .reindex(index=timestamps, columns=assets)
                .fillna(False)
                .to_numpy(dtype=bool)
            )
            fields = {
                name: np.where(membership, values, np.nan)
                for name, values in fields.items()
            }
        panel = PanelData(timestamps, assets, fields)
        panel.validate()
        return panel
