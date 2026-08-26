from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "源码"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from factorforge.a_share.period_gate import evaluate_required_years  # noqa: E402
from factorforge.cli import main as core_main  # noqa: E402


def _run_core(arguments: list[str]) -> int:
    """把中文操作入口转换为稳定的内部命令。"""

    previous = sys.argv[:]
    try:
        sys.argv = ["因子挖掘", *arguments]
        return int(core_main())
    finally:
        sys.argv = previous


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("年度证据只支持Parquet或CSV")


def main() -> int:
    parser = argparse.ArgumentParser(description="A股因子挖掘系统人工操作入口")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("环境检查", help="检查CPU、内存、显卡和安全运行容量")

    demo = commands.add_parser("试运行", help="运行不含真实市场结论的合成样例")
    demo.add_argument("--输出", type=Path, default=Path("输出/试运行"))

    simulate = commands.add_parser("资源预估", help="长任务开始前估算资源")
    simulate.add_argument("--配置", type=Path, required=True)
    simulate.add_argument("--行数", type=int, required=True)
    simulate.add_argument("--证券数", type=int, required=True)

    run = commands.add_parser("正式挖掘", help="运行标准A股长表研究面板")
    run.add_argument("--配置", type=Path, required=True)
    run.add_argument("--面板", type=Path, required=True)
    run.add_argument("--标签列", default="future_return")

    yearly = commands.add_parser("年度复核", help="要求2025与2026分别通过")
    yearly.add_argument("--证据", type=Path, required=True)
    yearly.add_argument("--输出", type=Path, required=True)

    inspect = commands.add_parser("查看因子", help="查看公式和完整裁决链")
    inspect.add_argument("--运行目录", type=Path, required=True)
    inspect.add_argument("--编号", required=True)

    args = parser.parse_args()
    if args.command == "环境检查":
        return _run_core(["doctor"])
    if args.command == "试运行":
        return _run_core(["demo", "--output", str(args.输出)])
    if args.command == "资源预估":
        return _run_core(
            [
                "simulate",
                "--config",
                str(args.配置),
                "--rows",
                str(args.行数),
                "--assets",
                str(args.证券数),
            ]
        )
    if args.command == "正式挖掘":
        return _run_core(
            [
                "run",
                "--config",
                str(args.配置),
                "--panel",
                str(args.面板),
                "--label-column",
                args.标签列,
            ]
        )
    if args.command == "年度复核":
        result = evaluate_required_years(_read_table(args.证据))
        args.输出.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.输出, index=False, encoding="utf-8-sig")
        print(json.dumps({"输出": str(args.输出), "记录数": len(result)}, ensure_ascii=False))
        return 0
    if args.command == "查看因子":
        return _run_core(
            [
                "inspect-candidate",
                "--run",
                str(args.运行目录),
                "--id",
                args.编号,
            ]
        )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
