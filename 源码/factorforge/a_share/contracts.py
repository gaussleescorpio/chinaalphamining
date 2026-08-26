from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class SourceKind(str, Enum):
    """A股研究支持的数据类型；未使用的数据源可以不配置。"""

    OHLCV = "ohlcv"
    CORPORATE_ACTION = "corporate_action"
    FUNDAMENTAL = "fundamental"
    SECURITY_MASTER = "security_master"
    ETF_CLASSIFICATION = "etf_classification"
    ALTERNATIVE = "alternative"


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class FieldContract:
    canonical_name: str
    dtype: str
    required: bool = True
    nullable: bool = False
    aliases: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    description: str = ""


@dataclass(frozen=True)
class DatasetContract:
    """Machine-readable input contract independent of any vendor."""

    dataset_id: str
    kind: SourceKind
    fields: tuple[FieldContract, ...]
    key_columns: tuple[str, ...]
    event_time_column: str
    available_time_column: str | None = "available_at"
    timezone: str = "UTC"
    allow_extra_fields: bool = True
    metadata: Mapping[str, str] = field(default_factory=dict)

    @property
    def required_columns(self) -> tuple[str, ...]:
        return tuple(item.canonical_name for item in self.fields if item.required)


@dataclass(frozen=True)
class QualityIssue:
    code: str
    severity: Severity
    dataset_id: str
    message: str
    affected_rows: int = 0
    examples: tuple[str, ...] = ()


def ohlcv_contract(dataset_id: str = "a_share_ohlcv") -> DatasetContract:
    return DatasetContract(
        dataset_id=dataset_id,
        kind=SourceKind.OHLCV,
        timezone="Asia/Shanghai",
        key_columns=("symbol", "timestamp"),
        event_time_column="timestamp",
        fields=(
            FieldContract("symbol", "string", nullable=False),
            FieldContract("timestamp", "datetime", nullable=False),
            FieldContract("available_at", "datetime", nullable=False),
            FieldContract("open", "float", minimum=0.0),
            FieldContract("high", "float", minimum=0.0),
            FieldContract("low", "float", minimum=0.0),
            FieldContract("close", "float", minimum=0.0),
            FieldContract("volume", "float", minimum=0.0),
            FieldContract("vwap", "float", required=False, minimum=0.0),
            FieldContract("trade_count", "float", required=False, minimum=0.0),
            FieldContract("exchange", "string", required=False),
            FieldContract("currency", "string", required=False),
        ),
    )


def corporate_action_contract(dataset_id: str = "a_share_corporate_actions") -> DatasetContract:
    return DatasetContract(
        dataset_id=dataset_id,
        kind=SourceKind.CORPORATE_ACTION,
        timezone="Asia/Shanghai",
        key_columns=("symbol", "effective_at", "action_type"),
        event_time_column="effective_at",
        fields=(
            FieldContract("symbol", "string"),
            FieldContract("action_type", "string"),
            FieldContract("effective_at", "datetime"),
            FieldContract("available_at", "datetime"),
            FieldContract("split_ratio", "float", required=False, minimum=0.0),
            FieldContract("cash_amount", "float", required=False),
            FieldContract("new_symbol", "string", required=False),
            FieldContract("currency", "string", required=False),
        ),
    )


def fundamental_contract(dataset_id: str = "a_share_fundamentals") -> DatasetContract:
    return DatasetContract(
        dataset_id=dataset_id,
        kind=SourceKind.FUNDAMENTAL,
        timezone="Asia/Shanghai",
        key_columns=("symbol", "fiscal_period_end", "field", "available_at"),
        event_time_column="fiscal_period_end",
        fields=(
            FieldContract("symbol", "string"),
            FieldContract("fiscal_period_end", "datetime"),
            FieldContract("filed_at", "datetime"),
            FieldContract("available_at", "datetime"),
            FieldContract("field", "string"),
            FieldContract("value", "float"),
            FieldContract("revision_id", "string", required=False),
        ),
    )


def security_master_contract(dataset_id: str = "a_share_security_master") -> DatasetContract:
    return DatasetContract(
        dataset_id=dataset_id,
        kind=SourceKind.SECURITY_MASTER,
        timezone="Asia/Shanghai",
        key_columns=("permanent_id", "effective_from"),
        event_time_column="effective_from",
        fields=(
            FieldContract("permanent_id", "string"),
            FieldContract("symbol", "string"),
            FieldContract("effective_from", "datetime"),
            FieldContract("effective_to", "datetime", required=False),
            FieldContract("available_at", "datetime"),
            FieldContract("listing_at", "datetime"),
            FieldContract("delisting_at", "datetime", required=False),
            FieldContract("exchange", "string"),
            FieldContract("board", "string"),
            FieldContract("is_st", "bool", required=False),
            FieldContract("security_type", "string"),
            FieldContract("is_primary", "bool", required=False),
        ),
    )


def etf_classification_contract(
    dataset_id: str = "a_share_etf_classification",
) -> DatasetContract:
    return DatasetContract(
        dataset_id=dataset_id,
        kind=SourceKind.ETF_CLASSIFICATION,
        key_columns=("symbol", "effective_at", "classification_field"),
        event_time_column="effective_at",
        fields=(
            FieldContract("symbol", "string"),
            FieldContract("effective_at", "datetime"),
            FieldContract("available_at", "datetime"),
            FieldContract("classification_field", "string"),
            FieldContract("classification_value", "string"),
            FieldContract("leverage_multiple", "float", required=False),
            FieldContract("inverse_flag", "bool", required=False),
            FieldContract("aum", "float", required=False, minimum=0.0),
        ),
    )


def alternative_contract(
    value_fields: tuple[str, ...], dataset_id: str = "a_share_alternative"
) -> DatasetContract:
    """Create an explicit PIT contract for optional licensed/alternative data."""

    return DatasetContract(
        dataset_id=dataset_id,
        kind=SourceKind.ALTERNATIVE,
        key_columns=("symbol", "event_at", "available_at"),
        event_time_column="event_at",
        fields=(
            FieldContract("symbol", "string"),
            FieldContract("event_at", "datetime"),
            FieldContract("available_at", "datetime"),
            *(FieldContract(name, "float", required=False) for name in value_fields),
        ),
        metadata={"license_boundary": "caller_must_verify"},
    )
