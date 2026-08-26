from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Mapping

import numpy as np
import pandas as pd


PIT_LAG_ONE_FIELDS = {
    "turnover_rate", "turnover_rate_f", "volume_ratio", "total_mv", "circ_mv",
    "total_share", "float_share", "free_share", "free_ratio", "mv_ratio",
    "pe_ttm", "pb", "ps_ttm", "dv_ttm", "ep", "bp", "sp",
}


@dataclass(frozen=True)
class FormulaSpec:
    candidate_id: str
    family: str
    formula: str
    atoms: tuple[str, ...]
    lookback_sessions: int
    availability_lag_sessions: int


def _safe_ratio(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    denominator = right.where(right.abs() > 1.0e-12)
    return left / denominator


def _time_zscore(value: pd.DataFrame, window: int) -> pd.DataFrame:
    mean = value.rolling(window, min_periods=max(5, window // 2)).mean()
    std = value.rolling(window, min_periods=max(5, window // 2)).std()
    return _safe_ratio(value - mean, std)


def _rank(value: pd.DataFrame) -> pd.DataFrame:
    return value.rank(axis=1, pct=True, method="average") - 0.5


def visible_panel(name: str, panels: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    value = panels[name]
    return value.shift(1) if name in PIT_LAG_ONE_FIELDS else value


def generate_formula_panels(panels: Mapping[str, pd.DataFrame]) -> Iterator[tuple[FormulaSpec, pd.DataFrame]]:
    """Yield a bounded, auditable mathematical space without reading labels."""

    unary = (
        "gap", "oc_ret", "vwap_close", "amihud", "volume_ratio", "turn_z",
        "mom_accel", "stoch_pos", "rsi14", "on_id_diff", "body_ratio",
        "up_shadow", "dn_shadow", "free_ratio", "mv_ratio", "ep", "bp", "sp",
        "ret5", "ret20", "ret60", "rvol20", "rvol60", "atr14", "vol_adv",
        "turnover_rate", "turnover_rate_f",
    )
    for name in unary:
        value = visible_panel(name, panels)
        lag = 1 if name in PIT_LAG_ONE_FIELDS else 0
        yield FormulaSpec(f"{name}__level", "level", f"rank({name})", (name,), 1, lag), _rank(value)
        for window in (5, 20, 60):
            yield FormulaSpec(
                f"{name}__z{window}", "state_normalized", f"rank(ts_zscore({name},{window}))",
                (name,), window, lag,
            ), _rank(_time_zscore(value, window))
        for window in (1, 5, 20):
            yield FormulaSpec(
                f"{name}__delta{window}", "change", f"rank({name}-delay({name},{window}))",
                (name,), window + 1, lag,
            ), _rank(value - value.shift(window))

    structured = (
        ("momentum_curve", "path", "rank(ret5-ret20/4)", ("ret5", "ret20"), 20,
         lambda p: visible_panel("ret5", p) - visible_panel("ret20", p) / 4.0),
        ("momentum_slope", "path", "rank(ret20-ret60/3)", ("ret20", "ret60"), 60,
         lambda p: visible_panel("ret20", p) - visible_panel("ret60", p) / 3.0),
        ("volatility_term", "risk_state", "rank(rvol20/rvol60)", ("rvol20", "rvol60"), 60,
         lambda p: _safe_ratio(visible_panel("rvol20", p), visible_panel("rvol60", p))),
        ("shadow_asymmetry", "path", "rank(up_shadow-dn_shadow)", ("up_shadow", "dn_shadow"), 1,
         lambda p: visible_panel("up_shadow", p) - visible_panel("dn_shadow", p)),
        ("close_vwap_pressure", "price_volume", "rank((vwap_close-1)*volume_ratio_lag1)", ("vwap_close", "volume_ratio"), 2,
         lambda p: (visible_panel("vwap_close", p) - 1.0) * visible_panel("volume_ratio", p)),
        ("gap_volume_confirmation", "price_volume", "rank(gap*volume_ratio_lag1)", ("gap", "volume_ratio"), 2,
         lambda p: visible_panel("gap", p) * visible_panel("volume_ratio", p)),
        ("intraday_volume_confirmation", "price_volume", "rank(oc_ret*volume_ratio_lag1)", ("oc_ret", "volume_ratio"), 2,
         lambda p: visible_panel("oc_ret", p) * visible_panel("volume_ratio", p)),
        ("illiquidity_shock", "liquidity", "rank(amihud*vol_adv)", ("amihud", "vol_adv"), 20,
         lambda p: visible_panel("amihud", p) * visible_panel("vol_adv", p)),
        ("float_supply_activity", "supply", "rank(free_ratio_lag1*turn_z)", ("free_ratio", "turn_z"), 20,
         lambda p: visible_panel("free_ratio", p) * visible_panel("turn_z", p)),
        ("value_momentum", "interaction", "rank(bp_lag1*ret20)", ("bp", "ret20"), 21,
         lambda p: visible_panel("bp", p) * visible_panel("ret20", p)),
        ("earnings_momentum", "interaction", "rank(ep_lag1*ret20)", ("ep", "ret20"), 21,
         lambda p: visible_panel("ep", p) * visible_panel("ret20", p)),
        ("range_efficiency", "path", "rank(abs(oc_ret)/atr14)", ("oc_ret", "atr14"), 14,
         lambda p: _safe_ratio(visible_panel("oc_ret", p).abs(), visible_panel("atr14", p))),
    )
    for candidate_id, family, formula, atoms, lookback, builder in structured:
        lag = max(1 if atom in PIT_LAG_ONE_FIELDS else 0 for atom in atoms)
        yield FormulaSpec(candidate_id, family, formula, atoms, lookback, lag), _rank(builder(panels))
