from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata

from factorforge.contracts import AtomSpec, PanelData

EPSILON = 1e-12


@dataclass(frozen=True)
class CompiledAtomLibrary:
    specs: tuple[AtomSpec, ...]
    values: Mapping[str, np.ndarray]
    family_counts: Mapping[str, int]


def _shift(values: np.ndarray, periods: int) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    if periods == 0:
        return source.copy()
    output = np.full_like(source, np.nan)
    if periods < len(source):
        output[periods:] = source[:-periods]
    return output


def _safe_divide(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.divide(
        left,
        right,
        out=np.full_like(left, np.nan, dtype=np.float64),
        where=np.isfinite(right) & (np.abs(right) > EPSILON),
    )


def _rolling_moments(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(values)
    clean = np.where(valid, values, 0.0)
    prefix = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(clean, axis=0)])
    counts = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(valid, axis=0)])
    sums = prefix[window:] - prefix[:-window]
    count = counts[window:] - counts[:-window]
    mean = np.full_like(values, np.nan, dtype=np.float64)
    mean[window - 1 :] = _safe_divide(sums, count)
    squared = np.vstack(
        [np.zeros((1, values.shape[1])), np.cumsum(clean * clean, axis=0)]
    )
    sums2 = squared[window:] - squared[:-window]
    variance = np.divide(
        sums2 - np.divide(sums * sums, count, out=np.zeros_like(sums), where=count > 0),
        count - 1,
        out=np.full_like(sums, np.nan),
        where=count > 1,
    )
    std = np.full_like(values, np.nan, dtype=np.float64)
    std[window - 1 :] = np.sqrt(np.maximum(variance, 0.0))
    return mean, std


def _rolling_extreme(values: np.ndarray, window: int, mode: str) -> np.ndarray:
    output = np.full_like(values, np.nan, dtype=np.float64)
    reducer = np.nanmax if mode == "max" else np.nanmin
    for end in range(window - 1, len(values)):
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            output[end] = reducer(values[end - window + 1 : end + 1], axis=0)
    return output


def _cross_rank(values: np.ndarray) -> np.ndarray:
    output = np.full_like(values, np.nan, dtype=np.float64)
    for index, row in enumerate(values):
        valid = np.isfinite(row)
        count = int(valid.sum())
        if count < 2:
            continue
        output[index, valid] = (rankdata(row[valid], method="average") - 1) / (
            count - 1
        ) - 0.5
    return output


def _cross_robust_z(values: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        median = np.nanmedian(values, axis=1, keepdims=True)
        mad = np.nanmedian(np.abs(values - median), axis=1, keepdims=True)
    return np.divide(
        values - median,
        1.4826 * mad,
        out=np.full_like(values, np.nan),
        where=mad > EPSILON,
    )


def _cross_residual(values: np.ndarray, control: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values) & np.isfinite(control)
    x = np.where(valid, control, 0.0)
    y = np.where(valid, values, 0.0)
    count = valid.sum(axis=1, keepdims=True)
    x_mean = _safe_divide(x.sum(axis=1, keepdims=True), count)
    y_mean = _safe_divide(y.sum(axis=1, keepdims=True), count)
    xc = np.where(valid, x - x_mean, 0.0)
    yc = np.where(valid, y - y_mean, 0.0)
    beta = _safe_divide(
        (xc * yc).sum(axis=1, keepdims=True), (xc * xc).sum(axis=1, keepdims=True)
    )
    residual = values - (y_mean + beta * (control - x_mean))
    return np.where(valid, residual, np.nan)


def _cross_residual_many(
    values: np.ndarray, controls: tuple[np.ndarray, ...]
) -> np.ndarray:
    """Daily cross-sectional OLS residual using only same-day visible controls."""

    if not controls:
        return values.copy()
    result = np.full_like(values, np.nan, dtype=float)
    for row in range(values.shape[0]):
        y = values[row]
        x = np.column_stack([control[row] for control in controls])
        valid = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
        if valid.sum() <= x.shape[1] + 2:
            continue
        design = np.column_stack([np.ones(valid.sum()), x[valid]])
        coefficients, *_ = np.linalg.lstsq(design, y[valid], rcond=None)
        result[row, valid] = y[valid] - design @ coefficients
    return result


def _rolling_correlation(
    left: np.ndarray, right: np.ndarray, window: int
) -> np.ndarray:
    left_mean, left_std = _rolling_moments(left, window)
    right_mean, right_std = _rolling_moments(right, window)
    product_mean, _ = _rolling_moments(left * right, window)
    return _safe_divide(product_mean - left_mean * right_mean, left_std * right_std)


def _rolling_sign_balance(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing persistence of changes, bounded between minus and plus one."""

    changes = values - _shift(values, 1)
    mean, _ = _rolling_moments(np.sign(changes), window)
    return mean


def _rolling_flip_rate(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing fraction of non-zero changes whose direction switched."""

    changes = np.sign(values - _shift(values, 1))
    previous = _shift(changes, 1)
    active = (changes != 0.0) & (previous != 0.0)
    switched = np.where(active, (changes != previous).astype(float), np.nan)
    mean, _ = _rolling_moments(switched, window)
    return mean


class AtomCompiler:
    """Compile a wide, causal atom library before formula generation.

    Source definitions describe availability and semantic role.  The compiler never
    accepts returns labels and every temporal transform is trailing-only.
    """

    FAMILIES = (
        "price_path",
        "volume_flow",
        "amount_liquidity",
        "volatility_tail",
        "cross_section_centering",
        "residual_relative_structure",
        "fundamental_event",
        "mathematical_shape",
    )

    def __init__(
        self,
        definitions: Mapping[str, Mapping[str, object]],
        windows: Sequence[int],
        enabled_families: Sequence[str] | None = None,
        maximum_atoms: int = 8_000,
        residual_model_version: str = "rank_controls_v2",
        required_atom_ids: Sequence[str] | None = None,
    ):
        self.definitions = definitions
        self.windows = tuple(
            sorted({int(value) for value in windows if int(value) > 1})
        )
        self.enabled = frozenset(enabled_families or self.FAMILIES)
        unknown = sorted(self.enabled - set(self.FAMILIES))
        if unknown:
            raise ValueError(f"unknown atom families: {unknown}")
        self.maximum_atoms = int(maximum_atoms)
        self.residual_model_version = residual_model_version
        self.required_atom_ids = (
            frozenset(str(value) for value in required_atom_ids)
            if required_atom_ids is not None
            else None
        )
        if self.maximum_atoms < 1:
            raise ValueError("maximum_atoms must be positive")
        if residual_model_version not in {"legacy_raw_v1", "rank_controls_v2"}:
            raise ValueError("unknown residual_model_version")

    def compile(self, panel: PanelData) -> CompiledAtomLibrary:
        """Compile trailing-only atoms from a validated decision-time panel.

        The method accepts no outcome labels. Field availability lags are applied
        before any cross-sectional, rolling or residual transform.
        """

        panel.validate()
        values: dict[str, np.ndarray] = {}
        specs: list[AtomSpec] = []
        roles: dict[str, str] = {}
        sources: dict[str, np.ndarray] = {}
        family_weights = {
            "mathematical_shape": 0.30,
            "price_path": 0.25,
            "residual_relative_structure": 0.15,
            "volatility_tail": 0.10,
            "cross_section_centering": 0.08,
            "volume_flow": 0.06,
            "amount_liquidity": 0.06,
            "fundamental_event": 0.10,
        }
        active_weight = sum(
            weight
            for family, weight in family_weights.items()
            if family in self.enabled and family != "fundamental_event"
        )
        family_limits = {
            family: max(1, int(self.maximum_atoms * weight / active_weight))
            for family, weight in family_weights.items()
            if family in self.enabled
        }
        family_used = {family: 0 for family in self.FAMILIES}

        def add(
            atom_id: str,
            array: np.ndarray,
            *,
            family: str,
            unit: str,
            description: str,
            lineage: Sequence[str],
            lookback: int = 0,
            parameters: Mapping[str, object] | None = None,
        ) -> None:
            if self.required_atom_ids is not None and atom_id not in self.required_atom_ids:
                return
            priority_atom = atom_id.startswith("source__") or any(
                token in atom_id
                for token in (
                    "path_efficiency",
                    "path_reversal_pressure",
                    "jump_concentration",
                    "return_sign_balance",
                    "semivariance_balance",
                    "realized_volatility",
                    "price_volume_corr",
                    "amihud_impact",
                    "gap_return",
                    "intraday_return",
                    "range_to_close",
                    "close_location",
                    "market_dispersion",
                )
            )
            if (
                family not in self.enabled
                or atom_id in values
                or len(values) >= self.maximum_atoms
                or (
                    self.required_atom_ids is None
                    and not priority_atom
                    and family_used[family]
                    >= family_limits.get(family, self.maximum_atoms)
                )
            ):
                return
            values[atom_id] = np.asarray(array, dtype=np.float64)
            family_used[family] += 1
            specs.append(
                AtomSpec(
                    atom_id=atom_id,
                    field=atom_id,
                    unit=unit,
                    description=description,
                    available_lag=lookback,
                    family=family,
                    lineage=tuple(lineage),
                    parameters=tuple(sorted((parameters or {}).items())),
                )
            )

        for field in sorted(panel.fields):
            definition = self.definitions.get(field, {})
            lag = int(definition.get("available_lag", 0))
            if lag < 0:
                raise ValueError(f"field {field} has a negative available_lag")
            source = _shift(np.asarray(panel.fields[field], dtype=np.float64), lag)
            sources[field] = source
            roles[field] = str(definition.get("role", field)).lower()
            unit = str(definition.get("unit", "input_native"))
            role = roles[field]
            if any(token in role for token in ("volume", "trade_count", "flow")):
                temporal_family = "volume_flow"
            elif any(token in role for token in ("amount", "turnover", "liquidity")):
                temporal_family = "amount_liquidity"
            elif any(
                token in role
                for token in (
                    "fundamental",
                    "income",
                    "cash_flow",
                    "revenue",
                    "estimate",
                    "valuation",
                    "short_interest",
                )
            ):
                temporal_family = "fundamental_event"
            else:
                temporal_family = "price_path"
            add(
                f"source__{field}",
                source,
                family="mathematical_shape",
                unit=unit,
                description=f"可用时点校正后的原始字段 {field}",
                lineage=(field,),
                lookback=lag,
            )
            add(
                f"signed_log__{field}",
                np.sign(source) * np.log1p(np.abs(source)),
                family="mathematical_shape",
                unit=f"signed_log({unit})",
                description=f"{field} 的有符号对数压缩",
                lineage=(field,),
                lookback=lag,
            )
            add(
                f"signed_sqrt__{field}",
                np.sign(source) * np.sqrt(np.abs(source)),
                family="mathematical_shape",
                unit=f"sqrt({unit})",
                description=f"{field} 的有符号平方根压缩",
                lineage=(field,),
                lookback=lag,
            )
            add(
                f"cs_rank__{field}",
                _cross_rank(source),
                family="cross_section_centering",
                unit="unitless",
                description=f"{field} 的当日横截面排名",
                lineage=(field,),
                lookback=lag,
            )
            add(
                f"cs_robust_z__{field}",
                _cross_robust_z(source),
                family="cross_section_centering",
                unit="unitless",
                description=f"{field} 的当日横截面稳健标准分",
                lineage=(field,),
                lookback=lag,
            )
            for window in self.windows:
                previous = _shift(source, window)
                delta = source - previous
                mean, std = _rolling_moments(source, window)
                add(
                    f"delta_{window}__{field}",
                    delta,
                    family=temporal_family,
                    unit=unit,
                    description=f"{field} 的 {window} 期变化",
                    lineage=(field,),
                    lookback=lag + window,
                    parameters={"window": window},
                )
                add(
                    f"relative_change_{window}__{field}",
                    _safe_divide(delta, np.abs(previous)),
                    family=temporal_family,
                    unit="unitless",
                    description=f"{field} 的 {window} 期相对变化",
                    lineage=(field,),
                    lookback=lag + window,
                    parameters={"window": window},
                )
                add(
                    f"ts_z_{window}__{field}",
                    _safe_divide(source - mean, std),
                    family="mathematical_shape",
                    unit="unitless",
                    description=f"{field} 相对过去 {window} 期的标准化偏离",
                    lineage=(field,),
                    lookback=lag + window,
                    parameters={"window": window},
                )
                add(
                    f"mean_gap_{window}__{field}",
                    _safe_divide(source - mean, np.abs(mean)),
                    family="mathematical_shape",
                    unit="unitless",
                    description=f"{field} 相对过去 {window} 期均值的比例偏离",
                    lineage=(field,),
                    lookback=lag + window,
                    parameters={"window": window},
                )
                half = max(1, window // 2)
                acceleration = (source - _shift(source, half)) - (
                    _shift(source, half) - _shift(source, window)
                )
                add(
                    f"acceleration_{window}__{field}",
                    acceleration,
                    family=temporal_family,
                    unit=unit,
                    description=f"{field} 在 {window} 期内的离散加速度",
                    lineage=(field,),
                    lookback=lag + window,
                    parameters={"window": window},
                )
                add(
                    f"slope_{window}__{field}",
                    delta / float(window),
                    family=temporal_family,
                    unit=unit,
                    description=f"{field} 在 {window} 期内的单位时间变化斜率",
                    lineage=(field,),
                    lookback=lag + window,
                    parameters={"window": window},
                )
                add(
                    f"sign_balance_{window}__{field}",
                    _rolling_sign_balance(source, window),
                    family="mathematical_shape",
                    unit="unitless",
                    description=f"{field} 在 {window} 期内的变化方向持续度",
                    lineage=(field,),
                    lookback=lag + window + 1,
                    parameters={"window": window},
                )
                add(
                    f"flip_rate_{window}__{field}",
                    _rolling_flip_rate(source, window),
                    family="mathematical_shape",
                    unit="unitless",
                    description=f"{field} 在 {window} 期内的方向切换比例",
                    lineage=(field,),
                    lookback=lag + window + 1,
                    parameters={"window": window},
                )

            if any(
                token in role
                for token in ("return", "volatility", "residual", "beta", "correlation")
            ):
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message="All-NaN slice encountered"
                    )
                    center = np.nanmedian(source, axis=1, keepdims=True)
                    dispersion = np.nanmedian(
                        np.abs(source - center), axis=1, keepdims=True
                    )
                add(
                    f"market_dispersion__{field}",
                    np.broadcast_to(dispersion, source.shape),
                    family="cross_section_centering",
                    unit=unit,
                    description=f"{field} 当日横截面绝对离差中位数形成的市场状态",
                    lineage=(field,),
                    lookback=lag,
                )

        role_field = {role: field for field, role in roles.items()}
        close_name = role_field.get("close")
        if close_name and close_name in sources:
            close = sources[close_name]
            open_ = sources.get(role_field.get("open", ""))
            high = sources.get(role_field.get("high", ""))
            low = sources.get(role_field.get("low", ""))
            volume = sources.get(role_field.get("volume", ""))
            amount = sources.get(role_field.get("amount", ""))
            trade_count = sources.get(role_field.get("trade_count", ""))
            previous_close = _shift(close, 1)
            returns = _safe_divide(close - previous_close, previous_close)
            add(
                "return_1",
                returns,
                family="price_path",
                unit="unitless",
                description="收盘到收盘的一期收益",
                lineage=(close_name,),
                lookback=1,
            )
            if open_ is not None:
                add(
                    "gap_return",
                    _safe_divide(open_ - previous_close, previous_close),
                    family="price_path",
                    unit="unitless",
                    description="隔夜跳空收益",
                    lineage=(close_name, role_field["open"]),
                    lookback=1,
                )
                add(
                    "intraday_return",
                    _safe_divide(close - open_, open_),
                    family="price_path",
                    unit="unitless",
                    description="开盘到收盘的日内收益",
                    lineage=(close_name, role_field["open"]),
                )
            if high is not None and low is not None:
                add(
                    "range_to_close",
                    _safe_divide(high - low, close),
                    family="volatility_tail",
                    unit="unitless",
                    description="高低区间相对收盘价",
                    lineage=(close_name, role_field["high"], role_field["low"]),
                )
                add(
                    "close_location",
                    _safe_divide(2 * close - high - low, high - low),
                    family="price_path",
                    unit="unitless",
                    description="收盘价在当日高低区间中的有符号位置",
                    lineage=(close_name, role_field["high"], role_field["low"]),
                )
            for window in self.windows:
                rolling_high = _rolling_extreme(close, window, "max")
                add(
                    f"drawdown_{window}",
                    _safe_divide(close - rolling_high, rolling_high),
                    family="price_path",
                    unit="unitless",
                    description=f"过去 {window} 期滚动回撤",
                    lineage=(close_name,),
                    lookback=window,
                    parameters={"window": window},
                )
                path = np.abs(close - _shift(close, window))
                step = np.abs(close - _shift(close, 1))
                step_mean, _ = _rolling_moments(step, window)
                add(
                    f"path_efficiency_{window}",
                    _safe_divide(path, step_mean * window),
                    family="price_path",
                    unit="unitless",
                    description=f"过去 {window} 期价格路径效率",
                    lineage=(close_name,),
                    lookback=window,
                    parameters={"window": window},
                )
                total_path = step_mean * window
                add(
                    f"path_reversal_pressure_{window}",
                    _safe_divide(total_path - path, total_path),
                    family="price_path",
                    unit="unitless",
                    description=f"过去 {window} 期总路径中未转化为净位移的比例",
                    lineage=(close_name,),
                    lookback=window,
                    parameters={"window": window},
                )
                largest_step = _rolling_extreme(np.abs(returns), window, "max")
                absolute_mean, _ = _rolling_moments(np.abs(returns), window)
                add(
                    f"jump_concentration_{window}",
                    _safe_divide(largest_step, absolute_mean * window),
                    family="volatility_tail",
                    unit="unitless",
                    description=f"过去 {window} 期最大单日变动占总绝对变动的比例",
                    lineage=(close_name,),
                    lookback=window,
                    parameters={"window": window},
                )
                add(
                    f"return_sign_balance_{window}",
                    _rolling_sign_balance(close, window),
                    family="price_path",
                    unit="unitless",
                    description=f"过去 {window} 期收益方向持续度",
                    lineage=(close_name,),
                    lookback=window + 1,
                    parameters={"window": window},
                )
                _, vol = _rolling_moments(returns, window)
                downside_mean, _ = _rolling_moments(
                    np.minimum(returns, 0.0) ** 2, window
                )
                upside_mean, _ = _rolling_moments(np.maximum(returns, 0.0) ** 2, window)
                add(
                    f"realized_volatility_{window}",
                    vol,
                    family="volatility_tail",
                    unit="unitless",
                    description=f"过去 {window} 期实现波动率",
                    lineage=(close_name,),
                    lookback=window,
                    parameters={"window": window},
                )
                add(
                    f"semivariance_balance_{window}",
                    _safe_divide(
                        upside_mean - downside_mean, upside_mean + downside_mean
                    ),
                    family="volatility_tail",
                    unit="unitless",
                    description=f"过去 {window} 期上下行半方差平衡",
                    lineage=(close_name,),
                    lookback=window,
                    parameters={"window": window},
                )
            if volume is not None:
                for window in self.windows:
                    add(
                        f"price_volume_corr_{window}",
                        _rolling_correlation(
                            returns, np.sign(returns) * np.log1p(np.abs(volume)), window
                        ),
                        family="residual_relative_structure",
                        unit="unitless",
                        description=f"过去 {window} 期价格变化与有符号成交量的相关性",
                        lineage=(close_name, role_field["volume"]),
                        lookback=window,
                        parameters={"window": window},
                    )
            if amount is not None:
                add(
                    "amihud_impact",
                    _safe_divide(np.abs(returns), amount),
                    family="amount_liquidity",
                    unit="1/currency",
                    description="单位成交额对应的绝对价格冲击",
                    lineage=(close_name, role_field["amount"]),
                    lookback=1,
                )
            if amount is not None and volume is not None:
                add(
                    "vwap_proxy",
                    _safe_divide(amount, volume),
                    family="amount_liquidity",
                    unit="price",
                    description="成交额除以成交量形成的均价代理",
                    lineage=(role_field["amount"], role_field["volume"]),
                )
            if amount is not None and trade_count is not None:
                add(
                    "amount_per_trade",
                    _safe_divide(amount, trade_count),
                    family="amount_liquidity",
                    unit="currency/count",
                    description="平均每笔成交额",
                    lineage=(role_field["amount"], role_field["trade_count"]),
                )

        control_names = [
            field
            for field, role in roles.items()
            if role
            in {
                "market_cap",
                "size",
                "liquidity_control",
                "volatility_control",
                "beta_control",
            }
        ]
        if control_names:
            controls = tuple(_cross_rank(sources[name]) for name in control_names)
            # Residual atoms are deliberately capped.  They are expensive full-panel
            # matrices and should represent normalized information, not raw price
            # levels whose nominal scale can masquerade as a cross-sectional signal.
            if self.residual_model_version == "legacy_raw_v1":
                eligible = list(values.items())[: min(96, len(values))]
            else:
                unit_by_atom = {spec.atom_id: spec.unit for spec in specs}
                lineage_by_atom = {spec.atom_id: set(spec.lineage) for spec in specs}
                eligible = [
                    (atom_id, array)
                    for atom_id, array in values.items()
                    if unit_by_atom.get(atom_id) == "unitless"
                    and atom_id not in control_names
                    and not lineage_by_atom.get(atom_id, set()).issubset(control_names)
                ][:96]
            for atom_id, array in eligible:
                residual_input = (
                    array
                    if self.residual_model_version == "legacy_raw_v1"
                    else _cross_rank(array)
                )
                add(
                    f"residual_controls__{atom_id}",
                    _cross_residual_many(residual_input, controls),
                    family="residual_relative_structure",
                    unit="unitless",
                    description=(
                        f"{atom_id} 对 {', '.join(control_names)} 的当日横截面投影残差"
                    ),
                    lineage=(atom_id, *control_names),
                )

        counts = {family: 0 for family in self.FAMILIES}
        for spec in specs:
            counts[spec.family] += 1
        return CompiledAtomLibrary(tuple(specs), values, counts)
