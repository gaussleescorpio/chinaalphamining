from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SourceSchema:
    timestamp_column: str
    asset_column: str
    fields: tuple[str, ...]
    membership_column: str | None = None

    def validate(self, frame: pd.DataFrame) -> None:
        required = {self.timestamp_column, self.asset_column, *self.fields}
        if self.membership_column:
            required.add(self.membership_column)
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"source is missing required fields: {missing}")


class TabularSourceAdapter:
    """Read one file or a partitioned directory without changing source values."""

    def __init__(self, schema: SourceSchema):
        self.schema = schema

    def read(
        self, path: str | Path, columns: Sequence[str] | None = None
    ) -> pd.DataFrame:
        source = Path(path)
        suffix = source.suffix.lower()
        requested = list(columns) if columns else None
        if source.is_dir() or suffix in {".parquet", ".pq"}:
            frame = pd.read_parquet(source, columns=requested)
        elif suffix == ".csv":
            frame = pd.read_csv(source, usecols=requested)
        else:
            raise ValueError(f"unsupported tabular source: {source}")
        self.schema.validate(frame)
        return frame


@dataclass(frozen=True)
class AShareEquityContract:
    """A股逐日研究面板的时点可用性与历史股票池契约。"""

    symbol: str = "symbol"
    date: str = "date"
    membership: str = "is_research_eligible"
    listing_date: str = "listing_date"
    delisting_date: str = "delisting_date"
    board: str = "board"
    available_at: str = "available_at"

    def validate(self, frame: pd.DataFrame) -> None:
        required = {
            self.symbol,
            self.date,
            self.membership,
            self.listing_date,
            self.delisting_date,
            self.board,
            self.available_at,
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"A股时点面板缺少字段: {missing}")
        if frame.duplicated([self.date, self.symbol]).any():
            raise ValueError("A股时点面板存在重复的日期和证券代码")
        research_date = pd.to_datetime(frame[self.date], errors="coerce")
        available_at = pd.to_datetime(frame[self.available_at], errors="coerce")
        listing_date = pd.to_datetime(frame[self.listing_date], errors="coerce")
        raw_delisting = frame[self.delisting_date]
        delisting_date = pd.to_datetime(raw_delisting, errors="coerce")
        if research_date.isna().any() or available_at.isna().any():
            raise ValueError("A股时点面板含无效研究日期或可用时间")
        if listing_date.isna().any():
            raise ValueError("A股时点面板含无效上市日期")
        if (raw_delisting.notna() & delisting_date.isna()).any():
            raise ValueError("A股时点面板含无效退市日期")
        if (available_at > research_date).any():
            raise ValueError("A股时点面板使用了研究时点以后才可获得的信息")
        membership = frame[self.membership]
        if membership.isna().any() or not membership.map(
            lambda value: isinstance(value, (bool, np.bool_))
        ).all():
            raise ValueError("历史研究资格必须是无缺失的布尔值")
        outside_life = (listing_date > research_date) | (
            delisting_date.notna() & (delisting_date < research_date)
        )
        if (membership.astype(bool) & outside_life).any():
            raise ValueError("尚未上市或已经退市的证券被错误纳入历史股票池")
        if frame[self.board].astype("string").str.strip().eq("").any():
            raise ValueError("A股板块字段不能为空")
