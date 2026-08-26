from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from factorforge.a_share.contracts import (
    DatasetContract,
    QualityIssue,
    Severity,
    SourceKind,
)


@dataclass(frozen=True)
class CleaningPolicy:
    duplicate_policy: str = "error"  # error or latest_available
    market_timezone: str = "Asia/Shanghai"
    missing_warning_fraction: float = 0.20
    price_jump_warning: float = 1.00
    fail_on_error: bool = True


@dataclass(frozen=True)
class CleanResult:
    frame: pd.DataFrame
    issues: tuple[QualityIssue, ...]

    @property
    def passed(self) -> bool:
        return not any(issue.severity == Severity.ERROR for issue in self.issues)

    def issue_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "code": item.code,
                    "severity": item.severity.value,
                    "dataset_id": item.dataset_id,
                    "message": item.message,
                    "affected_rows": item.affected_rows,
                    "examples": " | ".join(item.examples),
                }
                for item in self.issues
            ]
        )


class ContractCleaner:
    """Deterministic cleaner that never invents missing market observations."""

    def __init__(self, contract: DatasetContract, policy: CleaningPolicy | None = None):
        self.contract = contract
        self.policy = policy or CleaningPolicy()

    def clean(self, frame: pd.DataFrame) -> CleanResult:
        local, issues = self._canonicalize_columns(frame)
        issues.extend(self._check_required(local))
        if any(item.severity == Severity.ERROR for item in issues):
            return self._finish(local, issues)
        local, type_issues = self._coerce_types(local)
        issues.extend(type_issues)
        local, duplicate_issues = self._resolve_duplicates(local)
        issues.extend(duplicate_issues)
        issues.extend(self._check_missing(local))
        issues.extend(self._check_ranges(local))
        if self.contract.kind == SourceKind.OHLCV:
            issues.extend(self._check_ohlcv(local))
        elif self.contract.kind == SourceKind.CORPORATE_ACTION:
            issues.extend(self._check_actions(local))
        elif self.contract.kind == SourceKind.FUNDAMENTAL:
            issues.extend(self._check_fundamentals(local))
        local = local.sort_values(list(self.contract.key_columns), kind="stable")
        return self._finish(local.reset_index(drop=True), issues)

    def _canonicalize_columns(
        self, frame: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[QualityIssue]]:
        local = frame.copy()
        issues: list[QualityIssue] = []
        rename: dict[str, str] = {}
        for field in self.contract.fields:
            present = [name for name in (field.canonical_name, *field.aliases) if name in local]
            if len(present) > 1:
                issues.append(
                    self._issue(
                        "ambiguous_alias",
                        Severity.ERROR,
                        f"字段 {field.canonical_name} 同时出现多个别名: {present}",
                    )
                )
            elif present and present[0] != field.canonical_name:
                rename[present[0]] = field.canonical_name
        return local.rename(columns=rename), issues

    def _check_required(self, frame: pd.DataFrame) -> list[QualityIssue]:
        missing = sorted(set(self.contract.required_columns) - set(frame.columns))
        return (
            [self._issue("missing_required", Severity.ERROR, f"缺少必需字段: {missing}")]
            if missing
            else []
        )

    def _coerce_types(
        self, frame: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[QualityIssue]]:
        local = frame.copy()
        issues: list[QualityIssue] = []
        for field in self.contract.fields:
            name = field.canonical_name
            if name not in local:
                continue
            before = int(local[name].notna().sum())
            if field.dtype == "datetime":
                parsed = pd.to_datetime(local[name], errors="coerce", format="mixed")
                if parsed.dt.tz is None:
                    parsed = parsed.dt.tz_localize(
                        self.contract.timezone, ambiguous="NaT", nonexistent="NaT"
                    )
                local[name] = parsed.dt.tz_convert("UTC")
            elif field.dtype == "float":
                local[name] = pd.to_numeric(local[name], errors="coerce").replace(
                    [np.inf, -np.inf], np.nan
                )
            elif field.dtype == "bool":
                mapping = {
                    True: True,
                    False: False,
                    1: True,
                    0: False,
                    "1": True,
                    "0": False,
                    "true": True,
                    "false": False,
                    "yes": True,
                    "no": False,
                }
                local[name] = (
                    local[name]
                    .map(lambda value: mapping.get(str(value).strip().lower(), mapping.get(value)))
                    .astype("boolean")
                )
            elif field.dtype == "string":
                local[name] = local[name].astype("string").str.strip()
            lost = before - int(local[name].notna().sum())
            if lost:
                issues.append(
                    self._issue(
                        "type_coercion_loss",
                        Severity.ERROR,
                        f"字段 {name} 有 {lost} 个值无法转换为 {field.dtype}",
                        lost,
                    )
                )
        return local, issues

    def _resolve_duplicates(
        self, frame: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[QualityIssue]]:
        duplicate = frame.duplicated(list(self.contract.key_columns), keep=False)
        count = int(duplicate.sum())
        if not count:
            return frame, []
        if (
            self.policy.duplicate_policy == "latest_available"
            and self.contract.available_time_column in frame
        ):
            ordered = frame.sort_values(self.contract.available_time_column, kind="stable")
            clean = ordered.drop_duplicates(list(self.contract.key_columns), keep="last")
            return clean, [
                self._issue(
                    "duplicate_resolved",
                    Severity.WARNING,
                    "重复键按 available_at 保留最后可用版本",
                    count,
                )
            ]
        return frame, [
            self._issue(
                "duplicate_key",
                Severity.ERROR,
                "发现重复业务键；未配置明确的版本解决规则",
                count,
            )
        ]

    def _check_missing(self, frame: pd.DataFrame) -> list[QualityIssue]:
        issues: list[QualityIssue] = []
        for field in self.contract.fields:
            name = field.canonical_name
            if name not in frame:
                continue
            count = int(frame[name].isna().sum())
            fraction = count / max(len(frame), 1)
            if count and not field.nullable and field.required:
                severity = Severity.ERROR
            elif fraction > self.policy.missing_warning_fraction:
                severity = Severity.WARNING
            else:
                continue
            issues.append(
                self._issue(
                    "missing_values",
                    severity,
                    f"字段 {name} 缺失 {fraction:.2%}",
                    count,
                )
            )
        return issues

    def _check_ranges(self, frame: pd.DataFrame) -> list[QualityIssue]:
        issues: list[QualityIssue] = []
        for field in self.contract.fields:
            name = field.canonical_name
            if name not in frame or field.dtype != "float":
                continue
            invalid = pd.Series(False, index=frame.index)
            if field.minimum is not None:
                invalid |= frame[name] < field.minimum
            if field.maximum is not None:
                invalid |= frame[name] > field.maximum
            count = int(invalid.fillna(False).sum())
            if count:
                issues.append(
                    self._issue(
                        "out_of_range",
                        Severity.ERROR,
                        f"字段 {name} 存在超出契约范围的值",
                        count,
                    )
                )
        return issues

    def _check_ohlcv(self, frame: pd.DataFrame) -> list[QualityIssue]:
        issues: list[QualityIssue] = []
        required = {"open", "high", "low", "close"}
        if required.issubset(frame):
            bad = (frame["high"] < frame[["open", "close", "low"]].max(axis=1)) | (
                frame["low"] > frame[["open", "close", "high"]].min(axis=1)
            )
            count = int(bad.fillna(False).sum())
            if count:
                issues.append(
                    self._issue("invalid_ohlc", Severity.ERROR, "OHLC价格包络关系不成立", count)
                )
        if {"symbol", "timestamp", "close"}.issubset(frame):
            returns = frame.sort_values(["symbol", "timestamp"]).groupby("symbol")[
                "close"
            ].pct_change(fill_method=None)
            jumps = returns.abs() > self.policy.price_jump_warning
            count = int(jumps.fillna(False).sum())
            if count:
                issues.append(
                    self._issue(
                        "unexplained_price_jump",
                        Severity.WARNING,
                        "发现大幅价格跳变；应与拆并股、分红或错误报价交叉核对",
                        count,
                    )
                )
        if {"timestamp", "available_at"}.issubset(frame):
            early = frame["available_at"] < frame["timestamp"]
            count = int(early.fillna(False).sum())
            if count:
                issues.append(
                    self._issue(
                        "bar_available_too_early",
                        Severity.ERROR,
                        "行情可用时间早于行情结束时间",
                        count,
                    )
                )
        return issues

    def _check_actions(self, frame: pd.DataFrame) -> list[QualityIssue]:
        issues: list[QualityIssue] = []
        if "action_type" not in frame:
            return issues
        known = {"split", "cash_dividend", "symbol_change", "delisting", "spinoff"}
        unknown = ~frame["action_type"].isin(known)
        count = int(unknown.fillna(False).sum())
        if count:
            issues.append(
                self._issue("unknown_action", Severity.WARNING, "存在未注册的公司行动类型", count)
            )
        if "split_ratio" in frame:
            bad_split = frame["action_type"].eq("split") & (
                frame["split_ratio"].isna() | frame["split_ratio"].le(0)
            )
            count = int(bad_split.sum())
            if count:
                issues.append(
                    self._issue("invalid_split", Severity.ERROR, "拆并股缺少有效比例", count)
                )
        return issues

    def _check_fundamentals(self, frame: pd.DataFrame) -> list[QualityIssue]:
        issues: list[QualityIssue] = []
        if {"filed_at", "available_at"}.issubset(frame):
            bad = frame["available_at"] < frame["filed_at"]
            count = int(bad.fillna(False).sum())
            if count:
                issues.append(
                    self._issue(
                        "fundamental_available_before_filing",
                        Severity.ERROR,
                        "基本面字段可用时间早于申报时间",
                        count,
                    )
                )
        return issues

    def _finish(self, frame: pd.DataFrame, issues: Iterable[QualityIssue]) -> CleanResult:
        result = CleanResult(frame, tuple(issues))
        if self.policy.fail_on_error and not result.passed:
            messages = "; ".join(
                item.message
                for item in result.issues
                if item.severity == Severity.ERROR
            )
            raise ValueError(f"dataset {self.contract.dataset_id} failed cleaning: {messages}")
        return result

    def _issue(
        self, code: str, severity: Severity, message: str, affected_rows: int = 0
    ) -> QualityIssue:
        return QualityIssue(code, severity, self.contract.dataset_id, message, affected_rows)


def audit_session_continuity(
    frame: pd.DataFrame,
    expected_sessions: Iterable[pd.Timestamp],
    dataset_id: str,
    symbol_column: str = "symbol",
    timestamp_column: str = "timestamp",
    membership_column: str | None = None,
) -> tuple[QualityIssue, ...]:
    """使用外部交易日历核对缺口，不自行猜测休市日。"""

    expected = pd.DatetimeIndex(pd.to_datetime(list(expected_sessions), utc=True)).normalize()
    local = frame.copy()
    local["__session"] = pd.to_datetime(local[timestamp_column], utc=True).dt.normalize()
    issues: list[QualityIssue] = []
    for symbol, group in local.groupby(symbol_column, sort=False):
        active = group
        if membership_column and membership_column in group:
            active = group[group[membership_column].fillna(False)]
        if active.empty:
            continue
        observed = pd.DatetimeIndex(active["__session"].unique())
        span = expected[(expected >= observed.min()) & (expected <= observed.max())]
        missing = span.difference(observed)
        if len(missing):
            issues.append(
                QualityIssue(
                    "missing_sessions",
                    Severity.WARNING,
                    dataset_id,
                    f"{symbol} 在活动区间缺少 {len(missing)} 个交易日",
                    len(missing),
                    tuple(str(value.date()) for value in missing[:5]),
                )
            )
    return tuple(issues)


def audit_survivorship(
    prices: pd.DataFrame,
    security_master: pd.DataFrame | None,
    dataset_id: str = "us_ohlcv",
) -> tuple[QualityIssue, ...]:
    if security_master is None or security_master.empty:
        return (
            QualityIssue(
                "missing_security_master",
                Severity.ERROR,
                dataset_id,
                "没有动态证券主表，无法排除幸存者偏差、退市遗漏和代码变更断裂",
            ),
        )
    issues: list[QualityIssue] = []
    price_symbols = set(prices["symbol"].astype(str))
    master_symbols = set(security_master["symbol"].astype(str))
    absent = sorted(price_symbols - master_symbols)
    if absent:
        issues.append(
            QualityIssue(
                "unmapped_symbols",
                Severity.ERROR,
                dataset_id,
                "行情中存在无法映射到动态证券主表的代码",
                len(absent),
                tuple(absent[:5]),
            )
        )
    has_delisted = (
        "delisting_at" in security_master
        and security_master["delisting_at"].notna().any()
    )
    if not has_delisted:
        issues.append(
            QualityIssue(
                "no_delisting_history",
                Severity.WARNING,
                dataset_id,
                "证券主表没有任何退市记录，需确认是否只保留了当前存续证券",
            )
        )
    return tuple(issues)
