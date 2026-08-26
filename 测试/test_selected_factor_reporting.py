from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from factorforge.contracts import AtomSpec, CandidateRecord
from factorforge.evaluation.crossfit import CandidateEvidence
from factorforge.reporting import create_factor_charts, create_selected_factor_reports


def _evidence() -> CandidateEvidence:
    returns = np.array(
        [np.nan, 0.01, -0.005, 0.004, 0.002, -0.003, 0.006, 0.001],
        dtype=float,
    )
    metrics = {
        "oof_observations": 7,
        "rank_ic_mean": 0.03,
        "annual_return": 0.12,
        "sharpe": 1.1,
        "max_drawdown": 0.04,
        "calmar": 3.0,
        "positive_fold_fraction": 0.75,
        "top_5_percent_profit_concentration": 0.40,
        "mean_after_removing_best_event": 0.001,
        "block_sign_flip_p_value": 0.08,
        "effective_sample_size": 5.5,
        "block_bootstrap_mean_low": -0.001,
        "block_bootstrap_mean_high": 0.004,
        "stationary_bootstrap_mean_low": -0.002,
        "stationary_bootstrap_mean_high": 0.005,
        "hac_mean_t": 1.4,
        "hac_mean_standard_error": 0.001,
        "hac_lag": 3,
    }
    values = np.zeros((8, 3), dtype=float)
    return CandidateEvidence(
        "candidate_001", 1, values, values, values, returns, metrics, 0.07
    )


def test_selected_factor_report_contains_decision_useful_evidence(tmp_path: Path) -> None:
    evidence = _evidence()
    formula = {"op": "atom", "children": [], "params": {"atom_id": "close"}}
    record = CandidateRecord(
        candidate_id=evidence.candidate_id,
        formula_json=json.dumps(formula),
        formula_sha256="a" * 64,
        atoms=("close",),
        family="price_path",
        complexity=1,
        maximum_lookback=1,
        unit="price",
        generation_batch="test",
        seed=1,
    )
    atoms = [AtomSpec("close", "close", "USD", "收盘价格", 0)]
    dates = pd.date_range("2024-12-27", periods=8, freq="D", tz="UTC")
    paths = create_selected_factor_reports(
        [evidence],
        {evidence.candidate_id: 0.09},
        {evidence.candidate_id: record},
        atoms,
        dates,
        {},
        tmp_path,
        reporting_context={
            "holding_period": 5,
            "periods_per_year": 252,
            "one_way_cost_bps": 5.0,
            "label_return_type": "simple_return",
            "label_cost_status": "gross_before_cost",
        },
    )
    folder = tmp_path / "精选因子说明" / evidence.candidate_id
    assert paths == [folder / "因子说明.txt"]
    assert (folder / "逐年证据.parquet").is_file()
    assert (folder / "滚动证据.parquet").is_file()
    assert (folder / "收益结构摘要.csv").is_file()
    assert (folder / "统计验证摘要.json").is_file()
    annual = pd.read_parquet(folder / "逐年证据.parquet")
    assert set(annual["year"]) == {2024, 2025}
    assert {"compounded_return", "sharpe", "max_drawdown"}.issubset(annual.columns)
    definition = json.loads((folder / "因子定义.json").read_text(encoding="utf-8"))
    assert definition["reporting_context"]["holding_period"] == 5
    text = (folder / "因子说明.txt").read_text(encoding="utf-8")
    assert "成本与持有口径" in text
    assert "不推导其他成本假设下的收益" in text
    assert "收益结构与集中性" in text


def test_selected_factor_charts_cover_performance_and_concentration(
    tmp_path: Path,
) -> None:
    evidence = _evidence()
    dates = pd.date_range("2024-01-01", periods=8, freq="D", tz="UTC")
    paths = create_factor_charts(
        [evidence], tmp_path, timestamps=dates, periods_per_year=252
    )
    assert {path.name for path in paths} == {
        "candidate_001.png",
        "candidate_001_return_structure.png",
    }
    assert all(path.stat().st_size > 1_000 for path in paths)
