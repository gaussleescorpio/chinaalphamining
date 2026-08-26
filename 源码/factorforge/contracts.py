from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PanelData:
    timestamps: pd.DatetimeIndex
    assets: tuple[str, ...]
    fields: Mapping[str, np.ndarray]

    def validate(self) -> None:
        if (
            not self.timestamps.is_monotonic_increasing
            or self.timestamps.has_duplicates
        ):
            raise ValueError("timestamps must be strictly increasing")
        shape = (len(self.timestamps), len(self.assets))
        if not self.assets or len(set(self.assets)) != len(self.assets):
            raise ValueError("assets must be nonempty and unique")
        for name, values in self.fields.items():
            if np.asarray(values).shape != shape:
                raise ValueError(
                    f"field {name!r} has shape {np.asarray(values).shape}, expected {shape}"
                )


@dataclass(frozen=True)
class AtomSpec:
    atom_id: str
    field: str
    unit: str
    description: str
    available_lag: int = 0
    family: str = "source"
    lineage: tuple[str, ...] = ()
    parameters: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class CandidateRecord:
    candidate_id: str
    formula_json: str
    formula_sha256: str
    atoms: tuple[str, ...]
    family: str
    complexity: int
    maximum_lookback: int
    unit: str
    generation_batch: str
    seed: int


@dataclass(frozen=True)
class DecisionRecord:
    candidate_id: str
    stage: str
    accepted: bool
    reason_code: str
    reason_text: str
    metrics: Mapping[str, Any]
