from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from factorforge.adapters import LongFrameAdapter
from factorforge.config import ResearchConfig
from factorforge.pipeline import FactorDiscoveryPipeline


def test_end_to_end_pipeline_preserves_every_candidate(
    tmp_path: Path, monkeypatch
) -> None:
    total = 64 * 1024**3
    monkeypatch.setattr(
        "factorforge.runtime.psutil.virtual_memory",
        lambda: SimpleNamespace(total=total, available=60 * 1024**3),
    )
    dates = pd.date_range("2023-01-01", "2026-03-01", freq="W", tz="UTC")
    assets = tuple(f"A{index}" for index in range(6))
    rng = np.random.default_rng(11)
    signal = rng.normal(size=(len(dates), len(assets)))
    rows = []
    for time_index, date in enumerate(dates):
        for asset_index, asset in enumerate(assets):
            rows.append(
                {
                    "date": date,
                    "symbol": asset,
                    "field_a": signal[time_index, asset_index],
                    "field_b": rng.normal(),
                    "future_return": 0.003 * signal[time_index, asset_index]
                    + 0.0005 * rng.normal(),
                }
            )
    frame = pd.DataFrame(rows)
    payload = {
        "project_name": "test",
        "timestamp_column": "date",
        "asset_column": "symbol",
        "input_fields": ["field_a", "field_b"],
        "development_end": "2024-12-31",
        "oos_start": "2025-01-01",
        "oos_end": "2026-12-31",
        "holding_periods": [1, 3],
        "primary_holding_period": 3,
        "periods_per_year": 52,
        "one_way_cost_bps": 1.0,
        "rolling_windows": [3, 5],
        "candidate_limit": 24,
        "batch_size": 8,
        "workers": 1,
        "random_seed": 17,
        "max_memory_fraction": 0.75,
        "minimum_coverage": 0.50,
        "maximum_abs_rank_correlation": 0.90,
        "fdr_alpha": 0.10,
        "fdr_gate_enabled": False,
        "minimum_positive_fold_fraction": 0.50,
        "minimum_oof_rank_ic": -0.01,
        "quantile_count": 5,
        "target_pool_size": 3,
        "persist_values": "all_evaluated",
        "output_root": str(tmp_path / "run"),
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    config = ResearchConfig.from_json(config_path)
    panel = LongFrameAdapter("date", "symbol", ["field_a", "field_b"]).materialize(
        frame
    )
    labels = (
        LongFrameAdapter("date", "symbol", ["future_return"])
        .materialize(frame)
        .fields["future_return"]
    )
    summary = FactorDiscoveryPipeline(config).run(panel, labels)
    assert summary["catalog_candidates"] == 24
    catalog = pd.read_parquet(tmp_path / "run" / "完整候选目录.parquet")
    decisions = pd.read_parquet(tmp_path / "run" / "完整筛选淘汰记录.parquet")
    assert len(catalog) == 24
    assert set(catalog["candidate_id"]).issubset(set(decisions["candidate_id"]))
    assert (tmp_path / "run" / "FROZEN_POOL.json").is_file()
    assert (tmp_path / "run" / "SHA256_MANIFEST.csv").is_file()
    payload["resume"] = True
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    resumed = FactorDiscoveryPipeline(ResearchConfig.from_json(config_path)).run(
        panel, labels
    )
    assert resumed["resumed_from_ledger"] is True
    assert resumed["catalog_candidates"] == 24
    catalog_after = pd.read_parquet(tmp_path / "run" / "完整候选目录.parquet")
    assert len(catalog_after) == len(catalog) == 24
    assert set(catalog_after["candidate_id"]) == set(catalog["candidate_id"])
    decisions_after = pd.read_parquet(tmp_path / "run" / "完整筛选淘汰记录.parquet")
    assert len(decisions_after) == len(decisions)
    protected = {
        name: (tmp_path / "run" / name).read_bytes()
        for name in (
            "RUN_CONTRACT.json",
            "INPUT_FINGERPRINT.json",
            "SOFTWARE_ENVIRONMENT.json",
        )
    }
    changed_labels = labels.copy()
    changed_labels[-1, -1] += 1.0
    with pytest.raises(ValueError, match="resume contract mismatch"):
        FactorDiscoveryPipeline(ResearchConfig.from_json(config_path)).run(
            panel, changed_labels
        )
    for name, expected in protected.items():
        assert (tmp_path / "run" / name).read_bytes() == expected


def test_pipeline_rejects_an_input_field_identical_to_the_label(tmp_path: Path) -> None:
    dates = pd.date_range("2023-01-01", periods=24, freq="W", tz="UTC")
    assets = ("A", "B", "C")
    labels = np.arange(len(dates) * len(assets), dtype=float).reshape(
        len(dates), len(assets)
    )
    panel = LongFrameAdapter("date", "symbol", ["leaked_label"]).materialize(
        pd.DataFrame(
            {
                "date": np.repeat(dates, len(assets)),
                "symbol": np.tile(assets, len(dates)),
                "leaked_label": labels.reshape(-1),
            }
        )
    )
    payload = {
        "project_name": "leak_test",
        "timestamp_column": "date",
        "asset_column": "symbol",
        "input_fields": ["leaked_label"],
        "development_end": "2023-04-30",
        "oos_start": "2023-05-01",
        "oos_end": "2024-01-01",
        "holding_periods": [1],
        "primary_holding_period": 1,
        "periods_per_year": 52,
        "one_way_cost_bps": 1.0,
        "rolling_windows": [2],
        "candidate_limit": 4,
        "batch_size": 2,
        "workers": 1,
        "random_seed": 1,
        "max_memory_fraction": 0.5,
        "minimum_coverage": 0.5,
        "maximum_abs_rank_correlation": 0.9,
        "fdr_alpha": 0.1,
        "fdr_gate_enabled": False,
        "minimum_positive_fold_fraction": 0.5,
        "minimum_oof_rank_ic": 0.0,
        "quantile_count": 3,
        "target_pool_size": 1,
        "persist_values": "selected_only",
        "output_root": str(tmp_path / "leak_run"),
    }
    config_path = tmp_path / "leak_config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    pipeline = FactorDiscoveryPipeline(ResearchConfig.from_json(config_path))
    with pytest.raises(ValueError, match="identical to the evaluation label"):
        pipeline.validate_run_identity(panel, labels)
