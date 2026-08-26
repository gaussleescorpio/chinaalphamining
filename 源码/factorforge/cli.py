from __future__ import annotations

import argparse
import json
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import sqlite3

from factorforge.adapters import LongFrameAdapter
from factorforge.config import ResearchConfig
from factorforge.data_quality import audit_long_frame, clean_long_frame
from factorforge.pipeline import FactorDiscoveryPipeline
from factorforge.runtime import machine_report, resolve_runtime_limits, simulate_run
from factorforge.adapters.sources import AShareEquityContract


def _example_payload() -> dict[str, object]:
    resource = files("factorforge.resources").joinpath("example.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def _demo_frame(seed: int = 20260822) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", "2026-12-31", freq="B", tz="UTC")
    assets = [f"ASSET_{index:02d}" for index in range(8)]
    rows = []
    latent = rng.normal(size=(len(dates), len(assets)))
    volume = np.exp(8.0 + 0.4 * rng.normal(size=latent.shape))
    trades = np.maximum(
        10.0, volume / np.exp(4.0 + 0.2 * rng.normal(size=latent.shape))
    )
    active = (
        0.5 * volume
        + 0.12 * volume * latent
        + 0.05 * volume * rng.normal(size=latent.shape)
    )
    close = 100.0 * np.exp(
        np.cumsum(0.0002 + 0.006 * rng.normal(size=latent.shape), axis=0)
    )
    # Deliberately visible synthetic signal: this verifies the research plumbing,
    # not attainable market performance.
    future = 0.0040 * latent + 0.0010 * rng.normal(size=latent.shape)
    for time_index, date in enumerate(dates):
        for asset_index, asset in enumerate(assets):
            rows.append(
                {
                    "date": date,
                    "symbol": asset,
                    "close": close[time_index, asset_index],
                    "volume": volume[time_index, asset_index],
                    "trade_count": trades[time_index, asset_index],
                    "active_buy_volume": active[time_index, asset_index],
                    "future_return": future[time_index, asset_index],
                }
            )
    return pd.DataFrame(rows)


def run_demo(output: Path) -> dict[str, object]:
    payload = _example_payload()
    payload.update(
        {
            "candidate_limit": 64,
            "target_pool_size": 8,
            "batch_size": 32,
            "holding_periods": [1],
            "primary_holding_period": 1,
            "label_horizon_periods": 1,
            "label_start_offset_periods": 1,
            "output_root": str(output),
            "evidence_provenance": "synthetic_example",
        }
    )
    config_path = output.parent / "demo_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    config = ResearchConfig.from_json(config_path)
    frame = _demo_frame(config.random_seed)
    quality = audit_long_frame(
        frame,
        config.timestamp_column,
        config.asset_column,
        [*config.input_fields, "future_return"],
        config.membership_column,
    )
    frame = clean_long_frame(
        frame,
        config.timestamp_column,
        config.asset_column,
        [*config.input_fields, "future_return"],
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "DATA_QUALITY_REPORT.json").write_text(
        json.dumps(quality.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    panel = LongFrameAdapter(
        config.timestamp_column,
        config.asset_column,
        config.input_fields,
        config.membership_column,
    ).materialize(frame)
    label_panel = LongFrameAdapter(
        config.timestamp_column, config.asset_column, ["future_return"]
    ).materialize(frame)
    return FactorDiscoveryPipeline(config).run(
        panel, label_panel.fields["future_return"]
    )


def read_research_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".feather", ".ft"}:
        return pd.read_feather(path)
    if suffix in {".arrow", ".ipc"}:
        with pa.memory_map(str(path), "r") as source:
            try:
                table = ipc.open_file(source).read_all()
            except pa.ArrowInvalid:
                source.seek(0)
                table = ipc.open_stream(source).read_all()
        return table.to_pandas()
    raise ValueError(
        f"unsupported panel format {suffix!r}; use Parquet, Feather or Arrow IPC"
    )


def run_panel(
    config_path: Path, panel_path: Path, label_column: str
) -> dict[str, object]:
    config = ResearchConfig.from_json(config_path)
    if label_column in config.input_fields:
        raise ValueError(
            f"label column {label_column!r} must not appear in input_fields"
        )
    frame = read_research_frame(panel_path)
    if config.market_contract == "a_share_pit":
        AShareEquityContract(
            symbol=config.asset_column,
            date=config.timestamp_column,
            membership=config.membership_column or "is_research_eligible",
        ).validate(frame)
    quality = audit_long_frame(
        frame,
        config.timestamp_column,
        config.asset_column,
        [*config.input_fields, label_column],
        config.membership_column,
    )
    if not quality.passed:
        raise ValueError(f"data quality gate failed: {quality.to_dict()}")
    frame = clean_long_frame(
        frame,
        config.timestamp_column,
        config.asset_column,
        [*config.input_fields, label_column],
    )
    output = Path(config.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    panel = LongFrameAdapter(
        config.timestamp_column,
        config.asset_column,
        config.input_fields,
        config.membership_column,
    ).materialize(frame)
    label_panel = LongFrameAdapter(
        config.timestamp_column, config.asset_column, [label_column]
    ).materialize(frame)
    if (
        panel.timestamps.tolist() != label_panel.timestamps.tolist()
        or panel.assets != label_panel.assets
    ):
        raise ValueError("input and label panel keys do not align")
    pipeline = FactorDiscoveryPipeline(config)
    resumed = pipeline.validate_run_identity(panel, label_panel.fields[label_column])
    quality_path = output / "DATA_QUALITY_REPORT.json"
    if not resumed or not quality_path.exists():
        quality_path.write_text(
            json.dumps(quality.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return pipeline.run(panel, label_panel.fields[label_column])


def main() -> int:
    parser = argparse.ArgumentParser(description="FactorForge Max")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser(
        "demo", help="run a deterministic end-to-end demonstration"
    )
    demo.add_argument("--output", type=Path, default=Path("outputs/demo"))
    run = subparsers.add_parser(
        "run", help="run a long-format Parquet, Feather or Arrow IPC research panel"
    )
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--panel", type=Path, required=True)
    run.add_argument("--label-column", default="future_return")
    doctor = subparsers.add_parser(
        "doctor", help="inspect hardware and resolve safe runtime limits"
    )
    doctor.add_argument("--config", type=Path)
    initialize = subparsers.add_parser(
        "init", help="create an executable configuration from the clean template"
    )
    initialize.add_argument("--output", type=Path, required=True)
    initialize.add_argument(
        "--profile", choices=("safe", "balanced", "performance"), default="safe"
    )
    initialize.add_argument("--gpu", choices=("cpu", "auto", "cuda"), default="auto")
    inspect = subparsers.add_parser(
        "inspect-candidate", help="show one candidate and its complete decision history"
    )
    inspect.add_argument("--run", type=Path, required=True)
    inspect.add_argument("--id", required=True)
    simulate = subparsers.add_parser(
        "simulate", help="estimate safe resources before a long research run"
    )
    simulate.add_argument("--config", type=Path, required=True)
    simulate.add_argument("--rows", type=int, required=True)
    simulate.add_argument("--assets", type=int, required=True)
    args = parser.parse_args()
    if args.command == "demo":
        print(json.dumps(run_demo(args.output), ensure_ascii=False, indent=2))
        return 0
    if args.command == "run":
        print(
            json.dumps(
                run_panel(args.config, args.panel, args.label_column),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "doctor":
        report = machine_report()
        if args.config:
            report["resolved_limits"] = resolve_runtime_limits(
                ResearchConfig.from_json(args.config)
            ).__dict__
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "init":
        payload = _example_payload()
        payload["runtime_profile"] = args.profile
        payload["gpu_backend"] = args.gpu
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "configuration": str(args.output),
                    "profile": args.profile,
                    "gpu": args.gpu,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "inspect-candidate":
        ledger = args.run / "research_ledger.sqlite"
        if not ledger.is_file():
            raise FileNotFoundError(ledger)
        with sqlite3.connect(ledger) as connection:
            candidate = connection.execute(
                "SELECT * FROM candidates WHERE candidate_id=?", (args.id,)
            ).fetchone()
            decisions = connection.execute(
                "SELECT stage,accepted,reason_code,reason_text,metrics_json FROM decisions WHERE candidate_id=? ORDER BY decision_id",
                (args.id,),
            ).fetchall()
        if candidate is None:
            raise KeyError(args.id)
        candidate_fields = (
            "candidate_id",
            "formula_json",
            "formula_sha256",
            "atoms_json",
            "family",
            "complexity",
            "maximum_lookback",
            "unit",
            "generation_batch",
            "seed",
        )
        print(
            json.dumps(
                {
                    "candidate": dict(zip(candidate_fields, candidate, strict=True)),
                    "decisions": [
                        {
                            "stage": row[0],
                            "accepted": bool(row[1]),
                            "reason_code": row[2],
                            "reason_text": row[3],
                            "metrics": json.loads(row[4]),
                        }
                        for row in decisions
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "simulate":
        config = ResearchConfig.from_json(args.config)
        report = simulate_run(
            args.rows,
            args.assets,
            config.candidate_limit,
            resolve_runtime_limits(config),
            config.evaluation_shortlist_size,
            config.target_pool_size,
            config.formula_cache_entries,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
