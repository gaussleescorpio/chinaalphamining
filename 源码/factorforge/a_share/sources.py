from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import pandas as pd

from factorforge.a_share.contracts import DatasetContract


class DatasetSource(Protocol):
    """Vendor-neutral source boundary used by the cleaning pipeline."""

    source_id: str
    contract: DatasetContract

    def read(self, columns: Sequence[str] | None = None) -> pd.DataFrame: ...


@dataclass(frozen=True)
class FileDatasetSource:
    source_id: str
    path: Path
    contract: DatasetContract
    column_mapping: Mapping[str, str] | None = None

    def read(self, columns: Sequence[str] | None = None) -> pd.DataFrame:
        source = Path(self.path)
        if not source.exists():
            raise FileNotFoundError(source)
        reverse = {target: origin for origin, target in (self.column_mapping or {}).items()}
        raw_columns = [reverse.get(name, name) for name in columns] if columns else None
        if source.is_dir() or source.suffix.lower() in {".parquet", ".pq"}:
            frame = pd.read_parquet(source, columns=raw_columns)
        elif source.suffix.lower() in {".csv", ".txt"}:
            frame = pd.read_csv(source, usecols=raw_columns)
        else:
            raise ValueError(f"unsupported source format: {source}")
        if self.column_mapping:
            frame = frame.rename(columns=dict(self.column_mapping))
        return frame


class SourceRegistry:
    """Explicit registry: no implicit network calls or hidden credentials."""

    def __init__(self) -> None:
        self._sources: dict[str, DatasetSource] = {}

    def register(self, source: DatasetSource) -> None:
        source_id = str(source.source_id).strip()
        if not source_id or source_id in self._sources:
            raise ValueError(f"invalid or duplicate source id: {source_id!r}")
        self._sources[source_id] = source

    def read(self, source_id: str, columns: Sequence[str] | None = None) -> pd.DataFrame:
        try:
            source = self._sources[source_id]
        except KeyError as exc:
            raise KeyError(f"source is not registered: {source_id!r}") from exc
        return source.read(columns)

    def contracts(self) -> dict[str, DatasetContract]:
        return {key: value.contract for key, value in sorted(self._sources.items())}
