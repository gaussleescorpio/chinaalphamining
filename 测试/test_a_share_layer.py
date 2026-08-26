from __future__ import annotations

import numpy as np
import pandas as pd

from factorforge.a_share.execution import AShareExecutionEngine
from factorforge.a_share.period_gate import evaluate_required_years
from factorforge.adapters.sources import AShareEquityContract


def test_a_share_contract_rejects_future_availability() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2025-01-02"], "symbol": ["000001.SZ"],
            "is_research_eligible": [True], "listing_date": ["1991-04-03"],
            "delisting_date": [None], "board": ["主板"],
            "available_at": ["2025-01-03"],
        }
    )
    try:
        AShareEquityContract().validate(frame)
    except ValueError as error:
        assert "以后" in str(error)
    else:
        raise AssertionError("future availability should be rejected")


def test_year_gate_requires_both_years() -> None:
    evidence = pd.DataFrame(
        {
            "candidate_id": ["A"] * 130,
            "date": pd.date_range("2025-01-01", periods=130, freq="B"),
            "rank_ic": np.full(130, 0.02),
            "tail_spread": np.full(130, 0.001),
        }
    )
    result = evaluate_required_years(evidence)
    final = result[result["year"].eq("总体裁决")].iloc[0]
    assert not bool(final["year_passed"])
    assert final["qualification"] == "尚未完成全部年度验证"


def test_a_share_defaults_are_t_plus_one_cash_long_only() -> None:
    engine = AShareExecutionEngine()
    assert engine.rules.signal_lag_sessions == 1
    assert engine.rules.round_lot == 100
    assert engine.rules.allow_short is False
