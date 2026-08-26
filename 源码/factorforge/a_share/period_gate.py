from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StabilityGate:
    """冻结后的年度稳定性闸门，不参与候选方向和参数选择。"""

    required_years: tuple[int, ...] = (2025, 2026)
    minimum_observations_per_year: int = 120
    require_positive_rank_ic: bool = True
    require_positive_tail_spread: bool = True


def evaluate_required_years(
    daily_evidence: pd.DataFrame, *, gate: StabilityGate | None = None
) -> pd.DataFrame:
    """逐因子检查指定年份；缺失年份绝不按通过处理。"""

    gate = gate or StabilityGate()
    required = {"candidate_id", "date", "rank_ic", "tail_spread"}
    missing = sorted(required - set(daily_evidence.columns))
    if missing:
        raise ValueError(f"年度稳定性证据缺少字段: {missing}")
    frame = daily_evidence.loc[:, list(required)].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any():
        raise ValueError("年度稳定性证据含无效日期")
    frame["year"] = frame["date"].dt.year
    rows: list[dict[str, object]] = []
    for candidate_id, candidate in frame.groupby("candidate_id", sort=True):
        all_passed = True
        states: list[str] = []
        for year in gate.required_years:
            sample = candidate[candidate["year"].eq(year)]
            count = int(len(sample))
            rank_ic = float(np.nanmean(sample["rank_ic"])) if count else np.nan
            spread = float(np.nanmean(sample["tail_spread"])) if count else np.nan
            passed = count >= gate.minimum_observations_per_year
            if gate.require_positive_rank_ic:
                passed &= np.isfinite(rank_ic) and rank_ic > 0.0
            if gate.require_positive_tail_spread:
                passed &= np.isfinite(spread) and spread > 0.0
            all_passed &= bool(passed)
            state = "通过" if passed else ("缺少数据" if count == 0 else "未通过")
            states.append(f"{year}:{state}")
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "year": year,
                    "observations": count,
                    "rank_ic_mean": rank_ic,
                    "tail_spread_mean": spread,
                    "year_passed": bool(passed),
                }
            )
        rows.append(
            {
                "candidate_id": candidate_id,
                "year": "总体裁决",
                "observations": int(len(candidate)),
                "rank_ic_mean": np.nan,
                "tail_spread_mean": np.nan,
                "year_passed": bool(all_passed),
                "qualification": "具备初步可研究性" if all_passed else "尚未完成全部年度验证",
                "detail": "；".join(states),
            }
        )
    return pd.DataFrame(rows)
