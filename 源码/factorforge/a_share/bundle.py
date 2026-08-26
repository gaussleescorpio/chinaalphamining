from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import pandas as pd

from factorforge.a_share.cleaning import (
    CleanResult,
    CleaningPolicy,
    ContractCleaner,
    audit_session_continuity,
    audit_survivorship,
)
from factorforge.a_share.contracts import DatasetContract, QualityIssue, Severity
from factorforge.a_share.pit import resolve_symbol_history


@dataclass(frozen=True)
class BundleResult:
    tables: Mapping[str, pd.DataFrame]
    issues: tuple[QualityIssue, ...]

    @property
    def passed(self) -> bool:
        return not any(item.severity == Severity.ERROR for item in self.issues)


class AShareDataBundleProcessor:
    """Clean a multi-source bundle and enforce cross-source identity rules."""

    def __init__(
        self,
        contracts: Mapping[str, DatasetContract],
        policy: CleaningPolicy | None = None,
    ) -> None:
        self.contracts = dict(contracts)
        base = policy or CleaningPolicy()
        self.fail_on_error = base.fail_on_error
        self.policy = CleaningPolicy(
            duplicate_policy=base.duplicate_policy,
            market_timezone=base.market_timezone,
            missing_warning_fraction=base.missing_warning_fraction,
            price_jump_warning=base.price_jump_warning,
            fail_on_error=False,
        )

    def process(
        self,
        frames: Mapping[str, pd.DataFrame],
        expected_sessions: Iterable[pd.Timestamp] | None = None,
    ) -> BundleResult:
        unknown = sorted(set(frames) - set(self.contracts))
        if unknown:
            raise ValueError(f"bundle contains datasets without contracts: {unknown}")
        tables: dict[str, pd.DataFrame] = {}
        issues: list[QualityIssue] = []
        for dataset_id, contract in sorted(self.contracts.items()):
            if dataset_id not in frames:
                issues.append(
                    QualityIssue(
                        "dataset_not_supplied",
                        Severity.INFO,
                        dataset_id,
                        "该数据契约未启用；相关特征家族应保持关闭",
                    )
                )
                continue
            result: CleanResult = ContractCleaner(contract, self.policy).clean(
                frames[dataset_id]
            )
            tables[dataset_id] = result.frame
            issues.extend(result.issues)
        issues.extend(self._cross_source_checks(tables, expected_sessions))
        result = BundleResult(tables, tuple(issues))
        if self.fail_on_error and not result.passed:
            raise ValueError("A股数据包未通过跨数据源核对")
        return result

    def _cross_source_checks(
        self,
        tables: Mapping[str, pd.DataFrame],
        expected_sessions: Iterable[pd.Timestamp] | None,
    ) -> list[QualityIssue]:
        issues: list[QualityIssue] = []
        ohlcv_key = self._dataset_of_kind("ohlcv", tables)
        master_key = self._dataset_of_kind("security_master", tables)
        if ohlcv_key:
            prices = tables[ohlcv_key]
            master = tables.get(master_key) if master_key else None
            issues.extend(audit_survivorship(prices, master, ohlcv_key))
            if expected_sessions is not None:
                issues.extend(
                    audit_session_continuity(
                        prices,
                        expected_sessions,
                        ohlcv_key,
                        membership_column=(
                            "is_research_eligible"
                            if "is_research_eligible" in prices
                            else None
                        ),
                    )
                )
            if master is not None:
                try:
                    resolve_symbol_history(prices, master)
                except ValueError as exc:
                    issues.append(
                        QualityIssue(
                            "security_identity_failure",
                            Severity.ERROR,
                            ohlcv_key,
                            str(exc),
                        )
                    )
        if master_key:
            master_symbols = set(tables[master_key]["symbol"].astype(str))
            for dataset_id, frame in tables.items():
                if dataset_id == master_key or "symbol" not in frame:
                    continue
                unknown = sorted(set(frame["symbol"].astype(str)) - master_symbols)
                if unknown:
                    issues.append(
                        QualityIssue(
                            "cross_source_unmapped_symbol",
                            Severity.ERROR,
                            dataset_id,
                            "数据源包含动态证券主表无法识别的代码",
                            len(unknown),
                            tuple(unknown[:5]),
                        )
                    )
        return issues

    def _dataset_of_kind(
        self, kind: str, tables: Mapping[str, pd.DataFrame]
    ) -> str | None:
        matches = [
            dataset_id
            for dataset_id, contract in self.contracts.items()
            if contract.kind.value == kind and dataset_id in tables
        ]
        if len(matches) > 1:
            raise ValueError(f"bundle has multiple canonical {kind} datasets: {matches}")
        return matches[0] if matches else None
