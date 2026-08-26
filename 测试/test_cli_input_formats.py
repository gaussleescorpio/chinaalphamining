from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

from factorforge.cli import read_research_frame, run_panel


@pytest.mark.parametrize("suffix", [".parquet", ".feather", ".arrow"])
def test_research_frame_reader_supports_declared_formats(
    tmp_path: Path, suffix: str
) -> None:
    expected = pd.DataFrame({"date": [1, 2], "symbol": ["A", "B"], "value": [3.0, 4.0]})
    path = tmp_path / f"panel{suffix}"
    if suffix == ".parquet":
        expected.to_parquet(path, index=False)
    elif suffix == ".feather":
        expected.to_feather(path)
    else:
        table = pa.Table.from_pandas(expected, preserve_index=False)
        with pa.OSFile(str(path), "wb") as sink, ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    actual = read_research_frame(path)
    pd.testing.assert_frame_equal(actual, expected)


def test_cli_rejects_label_column_listed_as_an_input(tmp_path: Path) -> None:
    panel = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=6, tz="UTC"),
            "symbol": ["A"] * 6,
            "future_return": [0.0] * 6,
        }
    )
    panel_path = tmp_path / "panel.parquet"
    panel.to_parquet(panel_path, index=False)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        """{
          "project_name":"leak",
          "timestamp_column":"date",
          "asset_column":"symbol",
          "input_fields":["future_return"],
          "development_end":"2024-01-02",
          "oos_start":"2024-01-03",
          "oos_end":"2024-12-31",
          "holding_periods":[1],
          "primary_holding_period":1,
          "periods_per_year":252,
          "one_way_cost_bps":1,
          "rolling_windows":[2],
          "candidate_limit":1,
          "batch_size":1,
          "workers":1,
          "random_seed":1,
          "max_memory_fraction":0.5,
          "minimum_coverage":0.5,
          "maximum_abs_rank_correlation":0.9,
          "fdr_alpha":0.1,
          "fdr_gate_enabled":false,
          "minimum_positive_fold_fraction":0.5,
          "minimum_oof_rank_ic":0,
          "quantile_count":3,
          "target_pool_size":1,
          "persist_values":"selected_only",
          "output_root":"outputs/leak"
        }""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must not appear in input_fields"):
        run_panel(config_path, panel_path, "future_return")
