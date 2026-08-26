"""A股数据治理、因子发现和执行约束。"""

from factorforge.a_share.bundle import AShareDataBundleProcessor, BundleResult
from factorforge.a_share.cleaning import CleanResult, CleaningPolicy, ContractCleaner
from factorforge.a_share.execution import (
    AShareExecutionEngine,
    BacktestResult,
    CostModel,
    ExecutionRules,
)

__all__ = [
    "AShareDataBundleProcessor",
    "AShareExecutionEngine",
    "BacktestResult",
    "BundleResult",
    "CleanResult",
    "CleaningPolicy",
    "CostModel",
    "ContractCleaner",
    "ExecutionRules",
]
