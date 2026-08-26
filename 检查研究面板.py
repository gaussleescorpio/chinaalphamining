from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "源码"))

from factorforge.adapters.sources import AShareEquityContract  # noqa: E402


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("研究面板只支持Parquet或CSV")


def main() -> int:
    parser = argparse.ArgumentParser(description="检查A股研究面板的关键时间与股票池字段")
    parser.add_argument("面板", type=Path)
    parser.add_argument("--输出", type=Path, default=Path("输出/数据检查.json"))
    args = parser.parse_args()
    frame = read_table(args.面板)
    AShareEquityContract().validate(frame)
    dates = pd.to_datetime(frame["date"])
    report = {
        "通过": True,
        "记录数": int(len(frame)),
        "证券数": int(frame["symbol"].nunique()),
        "开始日期": str(dates.min().date()),
        "结束日期": str(dates.max().date()),
        "包含2025": bool(dates.dt.year.eq(2025).any()),
        "包含2026": bool(dates.dt.year.eq(2026).any()),
    }
    args.输出.parent.mkdir(parents=True, exist_ok=True)
    args.输出.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
