from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from factorforge.evaluation.crossfit import CandidateEvidence
from factorforge.contracts import AtomSpec, CandidateRecord

OPERATOR_NAMES = {
    "atom": "原始信息",
    "signed_log1p": "对称对数压缩",
    "cs_rank": "横截面排序",
    "cs_zscore": "横截面标准化",
    "delta": "历史变化",
    "rolling_mean": "滚动均值",
    "rolling_std": "滚动离散度",
    "rolling_zscore": "滚动偏离",
    "add": "信息合成",
    "subtract": "相对差",
    "multiply": "状态交互",
    "safe_divide": "单位基准比率",
    "normalized_difference": "归一化差异",
    "cross_projection_residual": "共同成分剥离",
}

COMMON_ATOM_NAMES = {
    "open": "开盘价格",
    "high": "最高价格",
    "low": "最低价格",
    "close": "收盘价格",
    "adjusted_close": "复权收盘价格",
    "volume": "成交量",
    "dollar_volume": "成交金额",
    "trade_count": "成交笔数",
    "active_buy_volume": "主动买入量",
    "market_cap": "总市值",
    "float_market_cap": "流通市值",
    "shares_outstanding": "总股本",
    "turnover": "换手率",
}


def _max_drawdown(returns: np.ndarray) -> float:
    clean = np.asarray(returns, dtype=float)
    clean = clean[np.isfinite(clean)]
    if not len(clean):
        return 0.0
    wealth = np.cumprod(1.0 + np.clip(clean, -0.999999, None))
    peak = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
    return float(np.max(1.0 - wealth / peak))


def _annualized_metrics(returns: np.ndarray, periods_per_year: int) -> dict[str, float]:
    clean = np.asarray(returns, dtype=float)
    clean = clean[np.isfinite(clean)]
    if not len(clean):
        return {"annual_return": np.nan, "sharpe": np.nan, "max_drawdown": np.nan}
    annual = float(
        np.expm1(
            np.mean(np.log1p(np.clip(clean, -0.999999, None))) * periods_per_year
        )
    )
    standard_deviation = float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0
    sharpe = (
        float(np.mean(clean) / standard_deviation * np.sqrt(periods_per_year))
        if standard_deviation > 0.0
        else 0.0
    )
    return {
        "annual_return": annual,
        "sharpe": sharpe,
        "max_drawdown": _max_drawdown(clean),
    }


def _profit_concentration(returns: np.ndarray, fraction: float) -> float:
    positive = np.sort(np.asarray(returns, dtype=float))
    positive = positive[np.isfinite(positive) & (positive > 0.0)][::-1]
    if not len(positive) or float(np.sum(positive)) <= 0.0:
        return 1.0
    count = max(1, int(np.ceil(len(positive) * fraction)))
    return float(np.sum(positive[:count]) / np.sum(positive))


def _evidence_tables(
    timestamps: pd.DatetimeIndex,
    returns: np.ndarray,
    periods_per_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path = pd.DataFrame({"date": timestamps, "net_return": returns})
    path = path[np.isfinite(path["net_return"])].copy()
    path["year"] = path["date"].dt.year
    annual_rows: list[dict[str, float | int]] = []
    for year, group in path.groupby("year", sort=True):
        values = group["net_return"].to_numpy(dtype=float)
        metrics = _annualized_metrics(values, periods_per_year)
        annual_rows.append(
            {
                "year": int(year),
                "observations": int(len(values)),
                "positive_period_fraction": float(np.mean(values > 0.0)),
                "mean_net_return": float(np.mean(values)),
                "total_net_return": float(np.sum(values)),
                "compounded_return": float(
                    np.prod(1.0 + np.clip(values, -0.999999, None)) - 1.0
                ),
                **metrics,
            }
        )
    annual = pd.DataFrame(annual_rows)

    window = min(max(20, periods_per_year // 4), max(2, len(path)))
    rolling = path[["date", "net_return"]].copy()
    rolling["rolling_window"] = window
    rolling["rolling_mean"] = rolling["net_return"].rolling(window).mean()
    rolling_std = rolling["net_return"].rolling(window).std(ddof=1)
    rolling["rolling_sharpe"] = (
        rolling["rolling_mean"] / rolling_std * np.sqrt(periods_per_year)
    )
    rolling_log = np.log1p(rolling["net_return"].clip(lower=-0.999999))
    rolling["rolling_annual_return"] = np.expm1(
        rolling_log.rolling(window).mean() * periods_per_year
    )
    rolling["rolling_positive_fraction"] = (
        (rolling["net_return"] > 0.0).astype(float).rolling(window).mean()
    )

    values = path["net_return"].to_numpy(dtype=float)
    quantiles = np.quantile(values, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    structure = pd.DataFrame(
        [
            {
                "observations": int(len(values)),
                "positive_period_fraction": float(np.mean(values > 0.0)),
                "mean_net_return": float(np.mean(values)),
                "median_net_return": float(np.median(values)),
                "standard_deviation": float(np.std(values, ddof=1))
                if len(values) > 1
                else 0.0,
                "minimum": float(np.min(values)),
                "p01": float(quantiles[0]),
                "p05": float(quantiles[1]),
                "p25": float(quantiles[2]),
                "p50": float(quantiles[3]),
                "p75": float(quantiles[4]),
                "p95": float(quantiles[5]),
                "p99": float(quantiles[6]),
                "maximum": float(np.max(values)),
                "top_1_percent_profit_concentration": _profit_concentration(values, 0.01),
                "top_5_percent_profit_concentration": _profit_concentration(values, 0.05),
                "top_10_percent_profit_concentration": _profit_concentration(values, 0.10),
                "mean_after_removing_best_event": float(
                    np.mean(np.delete(values, int(np.argmax(values))))
                )
                if len(values) > 1
                else 0.0,
            }
        ]
    )
    return annual, rolling, structure


def _formula_text(node: Mapping[str, object]) -> str:
    op = str(node["op"])
    params = dict(node.get("params", {}))
    children = [_formula_text(child) for child in node.get("children", [])]
    if op == "atom":
        return str(params["atom_id"])
    if "window" in params:
        return f"{op}({children[0]}, window={params['window']})"
    return f"{op}({', '.join(children)})"


def _factor_name(record: CandidateRecord, atom_specs: Mapping[str, AtomSpec]) -> str:
    formula = json.loads(record.formula_json)
    operator = OPERATOR_NAMES.get(str(formula["op"]), str(formula["op"]))
    primary = atom_specs.get(record.atoms[0]) if record.atoms else None
    fallback = record.atoms[0] if record.atoms else "市场信息"
    subject = COMMON_ATOM_NAMES.get(
        fallback, primary.description if primary else fallback
    )
    subject = subject.strip().rstrip("。")[:28]
    return f"{subject}的{operator}"


def create_selected_factor_reports(
    selected: Sequence[CandidateEvidence],
    q_values: Mapping[str, float],
    records: Mapping[str, CandidateRecord],
    atom_specs: Sequence[AtomSpec],
    timestamps: pd.DatetimeIndex,
    oos_metrics: Mapping[str, Mapping[str, object]],
    output: str | Path,
    evidence_provenance: str = "program_recomputation",
    reporting_context: Mapping[str, object] | None = None,
) -> list[Path]:
    """Create one self-contained Chinese report folder for every frozen factor."""

    root = Path(output) / "精选因子说明"
    root.mkdir(parents=True, exist_ok=True)
    spec_map = {item.atom_id: item for item in atom_specs}
    context = dict(reporting_context or {})
    periods_per_year = int(context.get("periods_per_year", 252))
    holding_period = int(context.get("holding_period", 1))
    one_way_cost_bps = float(context.get("one_way_cost_bps", np.nan))
    label_return_type = str(context.get("label_return_type", "未声明"))
    label_cost_status = str(context.get("label_cost_status", "未声明"))
    paths = []
    for evidence in selected:
        record = records[evidence.candidate_id]
        folder = root / evidence.candidate_id
        folder.mkdir(parents=True, exist_ok=True)
        name = _factor_name(record, spec_map)
        formula = json.loads(record.formula_json)
        atom_lines = []
        for atom_id in record.atoms:
            spec = spec_map[atom_id]
            atom_lines.append(
                f"{atom_id}：{spec.description}；单位为 {spec.unit}；最晚可用滞后为 {spec.available_lag} 个基础周期。"
            )
        annual, rolling, structure = _evidence_tables(
            timestamps,
            evidence.oof_return_path,
            periods_per_year,
        )
        annual.to_parquet(folder / "逐年证据.parquet", index=False)
        rolling.to_parquet(folder / "滚动证据.parquet", index=False)
        structure.to_csv(folder / "收益结构摘要.csv", index=False, encoding="utf-8-sig")
        definition = {
            "candidate_id": record.candidate_id,
            "name": name,
            "formula": formula,
            "formula_text": _formula_text(formula),
            "formula_sha256": record.formula_sha256,
            "atoms": list(record.atoms),
            "family": record.family,
            "unit": record.unit,
            "maximum_lookback": record.maximum_lookback,
            "direction_from_development": evidence.final_direction,
            "development_metrics": evidence.metrics,
            "bh_q_value": q_values[evidence.candidate_id],
            "outer_interval_metrics": dict(oos_metrics.get(evidence.candidate_id, {})),
            "evidence_provenance": evidence_provenance,
            "reporting_context": context,
        }
        (folder / "因子定义.json").write_text(
            json.dumps(definition, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        direction = (
            "数值越大，后续收益倾向越高"
            if evidence.final_direction > 0
            else "数值越小，后续收益倾向越高"
        )
        metrics = evidence.metrics
        statistical_summary = {
            "candidate_id": record.candidate_id,
            "hac": {
                "one_sided_p_value": evidence.p_value,
                "t_statistic": metrics.get("hac_mean_t"),
                "standard_error": metrics.get("hac_mean_standard_error"),
                "lag": metrics.get("hac_lag"),
                "meaning": "修正收益序列自相关后，检验平均收益是否大于零。",
            },
            "multiple_testing": {
                "bh_q_value": q_values[evidence.candidate_id],
                "meaning": "在完整候选目录口径下控制多重检验的预期错误发现比例。",
            },
            "moving_block_bootstrap": {
                "mean_low": metrics.get("block_bootstrap_mean_low"),
                "mean_high": metrics.get("block_bootstrap_mean_high"),
                "meaning": "按连续时间块重抽样，保留块内依赖并检查均值区间。",
            },
            "stationary_bootstrap": {
                "mean_low": metrics.get("stationary_bootstrap_mean_low"),
                "mean_high": metrics.get("stationary_bootstrap_mean_high"),
                "meaning": "使用随机长度时间块，检查结论是否依赖固定分块边界。",
            },
            "block_sign_flip": {
                "p_value": metrics.get("block_sign_flip_p_value"),
                "meaning": "整块翻转收益方向，检查正均值是否容易由偶然方向产生。",
            },
            "effective_sample_size": {
                "value": metrics.get("effective_sample_size"),
                "meaning": "按自相关折算的近似独立观察数量，不等同于原始交易日数。",
            },
            "interpretation_boundary": (
                "各检验回答不同问题；任一单项通过都不能单独证明未来收益。"
            ),
        }
        (folder / "统计验证摘要.json").write_text(
            json.dumps(statistical_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if evidence_provenance == "synthetic_example":
            provenance_text = (
                "来源类型：合成示例。输入与未来收益均由演示程序人工生成，"
                "并包含用于检验流程的合成共同信号。允许用途：验证安装、接口、"
                "账本和报告生成。禁止用途：真实市场因子效果、统计显著性或未来收益证明。"
            )
        else:
            provenance_text = (
                "来源类型：程序直接复算。数字来自本次运行的标准输入、固定配置和研究账本。"
                "适用范围只限本次运行；数据授权、PIT完整性和真实执行要求需另行验收。"
            )
        content = f"""{name}

因子编号：{record.candidate_id}

证据来源

{provenance_text}

用途说明

该因子向每个交易日、每个当日合资格证券输出一个连续分数。{direction}。它适合作为多因子池中的增量输入，不把单一阈值写成固定交易策略。

准确公式

{_formula_text(formula)}

输入字段与可用时间

{chr(10).join(atom_lines)}

计算约束

最大历史回看为 {record.maximum_lookback} 个基础周期。方向只在开发区间确定，外层区间不参与改名、改方向、改公式或调参。缺失值沿公式原样传播，横截面有效证券不足时不产生分数。

开发期交叉拟合证据

平均 RankIC：{metrics['rank_ic_mean']:.6f}
年化收益：{metrics['annual_return']:.2%}
夏普比率：{metrics['sharpe']:.3f}
最大回撤：{metrics['max_drawdown']:.2%}
卡玛比率：{metrics['calmar']:.3f}
HAC 单侧 p 值：{evidence.p_value:.6g}
时间块符号翻转 p 值：{metrics.get('block_sign_flip_p_value', float('nan')):.6g}
自相关折算后的有效样本数：{metrics.get('effective_sample_size', float('nan')):.1f}
固定区块 Bootstrap 均值区间：[{metrics.get('block_bootstrap_mean_low', float('nan')):.8f}, {metrics.get('block_bootstrap_mean_high', float('nan')):.8f}]
随机区块 Bootstrap 均值区间：[{metrics.get('stationary_bootstrap_mean_low', float('nan')):.8f}, {metrics.get('stationary_bootstrap_mean_high', float('nan')):.8f}]
全目录口径 BH q 值：{q_values[evidence.candidate_id]:.6g}
正向时间折比例：{metrics['positive_fold_fraction']:.2%}
最佳 5% 盈利集中度：{metrics['top_5_percent_profit_concentration']:.2%}
删除最大盈利事件后的平均收益：{metrics['mean_after_removing_best_event']:.8f}

年度与滚动稳定性

逐年证据.parquet 同时保存每年复合收益、年化收益、夏普、最大回撤、正收益期比例和观察数量。滚动证据.parquet 保存 {min(max(20, periods_per_year // 4), max(2, int(metrics.get('oof_observations', 2))))} 个基础周期窗口的滚动年化收益、夏普和正收益比例。年度或滚动结果用于识别衰减、单一年份依赖和收益路径不平滑，不用于运行后重新选择公式。

收益结构与集中性

收益结构摘要.csv 保存收益分位数、正收益期比例以及最佳 1%、5%、10% 正收益对总正收益的贡献。集中度过高表示结果依赖少数时期；它是脆弱性诊断，不是独立的统计显著性检验。

成本与持有口径

主要持有期：{holding_period} 个基础周期
每年基础周期数：{periods_per_year}
标签收益类型：{label_return_type}
标签成本状态：{label_cost_status}
研究评价使用的单边成本：{one_way_cost_bps:.4g} bp

本报告中的年化收益、夏普、最大回撤和卡玛来自交叉拟合的实际资金路径。成本只在评价层扣除一次。报告没有完整成交名义本金或毛收益路径时，不推导其他成本假设下的收益；成本敏感性必须由执行引擎重新回放，不能按比例外推。

适用角色与使用边界

该因子适合作为候选池中的连续增量输入，可用于横截面排序、组合权重输入或已有因子池的补充信息。它不是单独的买卖指令，也没有在本报告中确定容量、借券可得性、订单成交率或实盘阈值。正式上线前仍需在统一执行引擎中验证交易约束、成本、组合重叠和资金容量。

复现文件

因子定义.json 保存机器可读公式、方向和全部指标。逐年证据.parquet 与滚动证据.parquet 保存时间稳定性。收益结构摘要.csv 保存分布与集中性。统计验证摘要.json 保存检验数值、用途和解释边界。逐事件分数位于包根目录的 candidate_values 文件夹。图表位于包根目录 charts 文件夹。research_ledger.sqlite 保存该因子的全部入选与淘汰记录。

解释边界

本页报告的是冻结研究证据。统计显著性、经济收益和信息独立性分别核验；任何一项都不能单独替代其余两项。外层区间只用于冻结后的延续性观察。合成示例中的指标不得作为市场结论。
"""
        path = folder / "因子说明.txt"
        path.write_text(content, encoding="utf-8")
        paths.append(path)
    return paths


def create_family_coverage_report(frame: pd.DataFrame, output: str | Path) -> Path:
    target = Path(output) / "信息家族覆盖报告.txt"
    lines = ["信息家族覆盖报告", "", "下列数量来自同一研究账本，不混用旧版本结果。", ""]
    for row in frame.itertuples(index=False):
        lines.append(
            f"{row.family}：目录 {int(row.catalog_count)} 条；预筛 {int(row.screening_evaluated)} 条；"
            f"预筛通过 {int(row.screening_pass)} 条；精算 {int(row.exact_evaluated)} 条；入池 {int(row.selected)} 条。"
        )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def build_factor_analysis(
    selected: Sequence[CandidateEvidence],
    q_values: Mapping[str, float],
    output: str | Path,
) -> Path:
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for evidence in selected:
        path = evidence.oof_return_path[np.isfinite(evidence.oof_return_path)]
        positive = np.sort(path[path > 0.0])[::-1]
        profit = float(np.sum(positive))
        top_count = max(1, int(len(positive) * 0.05)) if len(positive) else 0
        concentration = (
            float(np.sum(positive[:top_count]) / profit) if profit > 0.0 else 1.0
        )
        rows.append(
            {
                "candidate_id": evidence.candidate_id,
                **{
                    key: value
                    for key, value in evidence.metrics.items()
                    if key != "payoff_shape"
                },
                "p_value": evidence.p_value,
                "bh_q_value": q_values[evidence.candidate_id],
                "top_5_percent_profit_concentration": concentration,
            }
        )
    target = root / "精选因子分析.parquet"
    pd.DataFrame(rows).to_parquet(target, index=False)
    return target


def create_factor_charts(
    selected: Sequence[CandidateEvidence],
    output: str | Path,
    timestamps: pd.DatetimeIndex | None = None,
    periods_per_year: int = 252,
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    root = Path(output) / "charts"
    root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for evidence in selected:
        finite = np.isfinite(evidence.oof_return_path)
        returns = np.asarray(evidence.oof_return_path, dtype=float)[finite]
        if not len(returns):
            continue
        dates = (
            pd.DatetimeIndex(timestamps)[finite]
            if timestamps is not None
            else pd.RangeIndex(len(returns))
        )
        wealth = np.cumprod(1.0 + np.clip(returns, -0.999999, None))
        peak = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
        drawdown = wealth / peak - 1.0
        path = pd.DataFrame({"date": dates, "return": returns})
        if timestamps is not None:
            path["year"] = pd.DatetimeIndex(dates).year
        else:
            path["year"] = 0
        grouped_year = path.groupby("year")["return"]
        annual = grouped_year.apply(
            lambda values: np.prod(1.0 + np.clip(values.to_numpy(), -0.999999, None))
            - 1.0
        )
        annual_counts = grouped_year.size()
        window = min(max(20, periods_per_year // 4), len(returns))
        rolling_mean = pd.Series(returns).rolling(window).mean()
        rolling_std = pd.Series(returns).rolling(window).std(ddof=1)
        rolling_sharpe = rolling_mean / rolling_std * np.sqrt(periods_per_year)

        figure, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].plot(dates, wealth, color="#17365d", linewidth=1.3)
        axes[0, 0].set_title("Cross-fitted wealth")
        axes[0, 0].set_ylabel("Wealth")
        axes[0, 0].grid(alpha=0.2)
        axes[0, 1].fill_between(
            dates, drawdown, 0.0, color="#a94442", alpha=0.75
        )
        axes[0, 1].set_title("Drawdown")
        axes[0, 1].set_ylabel("Drawdown")
        axes[0, 1].grid(alpha=0.2)
        partial = annual_counts.to_numpy() < max(2, int(periods_per_year * 0.5))
        colors = np.where(
            partial,
            "#8a8a8a",
            np.where(annual.to_numpy() >= 0.0, "#2f6b4f", "#a94442"),
        )
        year_labels = [
            f"{year}*" if is_partial else str(year)
            for year, is_partial in zip(annual.index, partial, strict=True)
        ]
        axes[1, 0].bar(year_labels, annual.to_numpy(), color=colors)
        axes[1, 0].axhline(0.0, color="#333333", linewidth=0.8)
        axes[1, 0].set_title("Compounded return by year")
        axes[1, 0].set_ylabel("Return")
        axes[1, 0].grid(axis="y", alpha=0.2)
        if np.any(partial):
            axes[1, 0].text(
                0.01,
                0.98,
                "* partial year",
                transform=axes[1, 0].transAxes,
                va="top",
                color="#666666",
                fontsize=9,
            )
        axes[1, 1].plot(dates, rolling_sharpe, color="#8c5a2b", linewidth=1.1)
        axes[1, 1].axhline(0.0, color="#333333", linewidth=0.8)
        axes[1, 1].set_title(f"Rolling Sharpe ({window} periods)")
        axes[1, 1].set_ylabel("Sharpe")
        axes[1, 1].grid(alpha=0.2)
        figure.suptitle(evidence.candidate_id)
        figure.tight_layout()
        target = root / f"{evidence.candidate_id}.png"
        figure.savefig(target, dpi=150, bbox_inches="tight")
        plt.close(figure)
        paths.append(target)

        structure_figure, structure_axes = plt.subplots(1, 2, figsize=(12, 4.5))
        lower, upper = np.quantile(returns, [0.01, 0.99])
        clipped = returns[(returns >= lower) & (returns <= upper)]
        structure_axes[0].hist(clipped, bins=40, color="#416a8c", alpha=0.85)
        structure_axes[0].axvline(0.0, color="#333333", linewidth=0.8)
        structure_axes[0].set_title("Return distribution (1%-99%)")
        structure_axes[0].set_xlabel("Net return")
        structure_axes[0].set_ylabel("Observations")
        structure_axes[0].grid(axis="y", alpha=0.2)
        positive = np.sort(returns[returns > 0.0])[::-1]
        if len(positive) and float(np.sum(positive)) > 0.0:
            cumulative = np.cumsum(positive) / np.sum(positive)
            share = np.arange(1, len(positive) + 1) / len(positive)
            structure_axes[1].plot(share, cumulative, color="#2f6b4f", linewidth=1.4)
            structure_axes[1].plot([0, 1], [0, 1], color="#777777", linestyle="--")
        structure_axes[1].set_title("Concentration of positive returns")
        structure_axes[1].set_xlabel("Share of positive periods")
        structure_axes[1].set_ylabel("Cumulative share of positive return")
        structure_axes[1].grid(alpha=0.2)
        structure_figure.suptitle(evidence.candidate_id)
        structure_figure.tight_layout()
        structure_target = root / f"{evidence.candidate_id}_return_structure.png"
        structure_figure.savefig(structure_target, dpi=150, bbox_inches="tight")
        plt.close(structure_figure)
        paths.append(structure_target)
    return paths
