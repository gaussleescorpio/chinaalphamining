from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostModel:
    commission_bps: float = 1.0
    slippage_bps: float = 3.0
    minimum_commission: float = 0.0
    sell_stamp_duty_bps: float = 5.0
    transfer_fee_bps: float = 0.1
    annual_borrow_bps: float = 0.0
    annual_cash_yield_bps: float = 0.0
    annual_debit_bps: float = 550.0


@dataclass(frozen=True)
class ExecutionRules:
    signal_lag_sessions: int = 1
    fill_price: str = "open"  # open or close
    fractional_shares: bool = False
    round_lot: int = 100
    allow_short: bool = False
    gross_leverage_limit: float = 1.0
    net_exposure_limit: float = 1.0
    position_weight_limit: float = 0.10
    unavailable_order_policy: str = "reject"  # reject; no hidden forward fill
    initial_cash: float = 1_000_000.0


class TargetWeightTransform(Protocol):
    """Extension point for a risk model or leverage controller."""

    transform_id: str

    def transform(self, weights: pd.Series, context: pd.Series) -> pd.Series: ...


@dataclass(frozen=True)
class BacktestResult:
    nav: pd.DataFrame
    trades: pd.DataFrame
    rejected_orders: pd.DataFrame
    diagnostics: dict[str, float | int | str]


def normalize_target_weights(weights: pd.Series, rules: ExecutionRules) -> pd.Series:
    local = pd.to_numeric(weights, errors="coerce").fillna(0.0).astype(float)
    local = local.clip(-rules.position_weight_limit, rules.position_weight_limit)
    if not rules.allow_short:
        local = local.clip(lower=0.0)
    net = float(local.sum())
    if abs(net) > rules.net_exposure_limit and abs(net) > 0:
        local *= rules.net_exposure_limit / abs(net)
    gross = float(local.abs().sum())
    if gross > rules.gross_leverage_limit and gross > 0:
        local *= rules.gross_leverage_limit / gross
    return local


class AShareExecutionEngine:
    """A股日线执行回放：T日收盘信号，最早T+1成交。"""

    def __init__(
        self,
        rules: ExecutionRules | None = None,
        costs: CostModel | None = None,
        weight_transform: TargetWeightTransform | None = None,
    ) -> None:
        self.rules = rules or ExecutionRules()
        self.costs = costs or CostModel()
        self.weight_transform = weight_transform
        if self.rules.signal_lag_sessions < 1:
            raise ValueError("A股收盘信号最早只能在下一交易日执行")
        if self.rules.fill_price not in {"open", "close"}:
            raise ValueError("fill_price must be open or close")
        if self.rules.unavailable_order_policy != "reject":
            raise ValueError("only explicit order rejection is supported")
        if self.rules.round_lot < 1:
            raise ValueError("round_lot must be positive")
        if min(
            self.rules.gross_leverage_limit,
            self.rules.net_exposure_limit,
            self.rules.position_weight_limit,
        ) < 0:
            raise ValueError("exposure limits cannot be negative")

    def run(
        self,
        signals: pd.DataFrame,
        bars: pd.DataFrame,
        corporate_actions: pd.DataFrame | None = None,
    ) -> BacktestResult:
        self._validate_inputs(signals, bars)
        market = bars.copy()
        market["session"] = pd.to_datetime(market["session"], utc=True).dt.normalize()
        market = market.sort_values(["session", "symbol"], kind="stable")
        sessions = pd.DatetimeIndex(market["session"].drop_duplicates())
        targets = self._schedule_signals(signals, sessions)
        actions = self._prepare_actions(corporate_actions)
        holdings: dict[str, float] = {}
        cash = float(self.rules.initial_cash)
        nav_rows: list[dict[str, float | pd.Timestamp]] = []
        trades: list[dict[str, object]] = []
        rejects: list[dict[str, object]] = []
        previous_close: dict[str, float] = {}

        for session in sessions:
            day = market[market["session"].eq(session)].set_index("symbol", drop=False)
            cash = self._apply_actions(session, actions, holdings, cash, trades)
            self._require_prices_for_holdings(day, holdings, previous_close, session)
            open_prices = day["open"].to_dict()
            close_prices = day["close"].to_dict()
            mark_prices = open_prices if self.rules.fill_price == "open" else close_prices
            nav_before = cash + sum(
                shares * float(mark_prices.get(symbol, previous_close.get(symbol)))
                for symbol, shares in holdings.items()
            )
            turnover = 0.0
            trading_cost = 0.0
            if session in targets:
                desired = targets[session].set_index("symbol")["target_weight"]
                if self.weight_transform:
                    desired = self.weight_transform.transform(
                        desired.copy(), pd.Series({"nav": nav_before, "session": session})
                    )
                desired = normalize_target_weights(desired, self.rules)
                symbols = set(holdings) | set(desired.index)
                for symbol in sorted(symbols):
                    price = mark_prices.get(symbol)
                    row = day.loc[symbol] if symbol in day.index else None
                    reason = self._unavailable_reason(row, desired.get(symbol, 0.0))
                    if price is None or not np.isfinite(price) or price <= 0 or reason:
                        rejects.append(
                            {
                                "session": session,
                                "symbol": symbol,
                                "target_weight": float(desired.get(symbol, 0.0)),
                                "reason": reason or "missing_execution_price",
                            }
                        )
                        continue
                    target_shares = nav_before * float(desired.get(symbol, 0.0)) / float(price)
                    target_shares = self._round_shares(target_shares)
                    old_shares = holdings.get(symbol, 0.0)
                    delta = target_shares - old_shares
                    if abs(delta) < 1e-12:
                        continue
                    notional = abs(delta * float(price))
                    sell_tax_bps = self.costs.sell_stamp_duty_bps if delta < 0 else 0.0
                    cost = max(
                        self.costs.minimum_commission,
                        notional
                        * (
                            self.costs.commission_bps
                            + self.costs.slippage_bps
                            + self.costs.transfer_fee_bps
                            + sell_tax_bps
                        )
                        / 10_000.0,
                    )
                    cash -= delta * float(price) + cost
                    trading_cost += cost
                    turnover += notional
                    if abs(target_shares) < 1e-12:
                        holdings.pop(symbol, None)
                    else:
                        holdings[symbol] = target_shares
                    trades.append(
                        {
                            "session": session,
                            "symbol": symbol,
                            "shares": delta,
                            "price": float(price),
                            "notional": notional,
                            "cost": cost,
                            "event": "trade",
                        }
                    )
            short_value = sum(
                abs(shares) * float(close_prices[symbol])
                for symbol, shares in holdings.items()
                if shares < 0 and symbol in close_prices
            )
            borrow_cost = short_value * self.costs.annual_borrow_bps / 10_000.0 / 252.0
            cash -= borrow_cost
            financing_cost = 0.0
            if cash > 0:
                cash += cash * self.costs.annual_cash_yield_bps / 10_000.0 / 252.0
            elif cash < 0:
                financing_cost = (
                    abs(cash) * self.costs.annual_debit_bps / 10_000.0 / 252.0
                )
                cash -= financing_cost
            nav_close = cash + sum(
                shares * float(close_prices.get(symbol, previous_close.get(symbol)))
                for symbol, shares in holdings.items()
            )
            gross = sum(
                abs(shares * float(close_prices.get(symbol, previous_close.get(symbol))))
                for symbol, shares in holdings.items()
            )
            net = sum(
                shares * float(close_prices.get(symbol, previous_close.get(symbol)))
                for symbol, shares in holdings.items()
            )
            nav_rows.append(
                {
                    "session": session,
                    "nav": nav_close,
                    "cash": cash,
                    "gross_exposure": gross / nav_close if nav_close else np.nan,
                    "net_exposure": net / nav_close if nav_close else np.nan,
                    "turnover": turnover / nav_before if nav_before else np.nan,
                    "trading_cost": trading_cost,
                    "borrow_cost": borrow_cost,
                    "financing_cost": financing_cost,
                }
            )
            previous_close.update({key: float(value) for key, value in close_prices.items()})

        nav = pd.DataFrame(nav_rows)
        trades_frame = pd.DataFrame(trades)
        rejects_frame = pd.DataFrame(rejects)
        return BacktestResult(
            nav=nav,
            trades=trades_frame,
            rejected_orders=rejects_frame,
            diagnostics={
                "signal_clock": f"close_T_to_{self.rules.fill_price}_next_session",
                "sessions": len(sessions),
                "trades": len(trades_frame),
                "rejected_orders": len(rejects_frame),
                "final_nav": float(nav["nav"].iloc[-1]) if len(nav) else self.rules.initial_cash,
            },
        )

    def _schedule_signals(
        self, signals: pd.DataFrame, sessions: pd.DatetimeIndex
    ) -> dict[pd.Timestamp, pd.DataFrame]:
        local = signals.copy()
        local["signal_session"] = pd.to_datetime(
            local["signal_session"], utc=True
        ).dt.normalize()
        scheduled: list[pd.DataFrame] = []
        for signal_session, group in local.groupby("signal_session", sort=True):
            if signal_session not in sessions:
                raise ValueError(
                    "signal session is not in the supplied exchange calendar: "
                    f"{signal_session}"
                )
            position = int(sessions.searchsorted(signal_session, side="right"))
            execution_index = position + self.rules.signal_lag_sessions - 1
            if execution_index >= len(sessions):
                continue
            clean = group.copy()
            clean["execution_session"] = sessions[execution_index]
            scheduled.append(clean)
        if not scheduled:
            return {}
        combined = pd.concat(scheduled, ignore_index=True)
        duplicate = combined.duplicated(["execution_session", "symbol"], keep=False)
        if duplicate.any():
            raise ValueError("multiple target weights for the same execution session and symbol")
        return {key: group for key, group in combined.groupby("execution_session", sort=True)}

    @staticmethod
    def _prepare_actions(actions: pd.DataFrame | None) -> pd.DataFrame:
        if actions is None or actions.empty:
            return pd.DataFrame(columns=["session", "symbol", "action_type"])
        local = actions.copy()
        local["session"] = pd.to_datetime(local["effective_at"], utc=True).dt.normalize()
        return local.sort_values(["session", "symbol"], kind="stable")

    @staticmethod
    def _apply_actions(
        session: pd.Timestamp,
        actions: pd.DataFrame,
        holdings: dict[str, float],
        cash: float,
        ledger: list[dict[str, object]],
    ) -> float:
        for row in actions[actions["session"].eq(session)].itertuples(index=False):
            symbol = str(row.symbol)
            shares = holdings.get(symbol, 0.0)
            if not shares:
                continue
            action = str(row.action_type)
            if action == "split":
                ratio = float(row.split_ratio)
                holdings[symbol] = shares * ratio
                ledger.append(
                    {
                        "session": session,
                        "symbol": symbol,
                        "event": "split",
                        "shares": 0.0,
                        "price": np.nan,
                        "notional": 0.0,
                        "cost": 0.0,
                    }
                )
            elif action == "cash_dividend":
                amount = shares * float(row.cash_amount)
                cash += amount
                ledger.append(
                    {
                        "session": session,
                        "symbol": symbol,
                        "event": "cash_dividend",
                        "shares": 0.0,
                        "price": np.nan,
                        "notional": amount,
                        "cost": 0.0,
                    }
                )
            elif action == "delisting":
                recovery = shares * float(getattr(row, "cash_amount", 0.0) or 0.0)
                cash += recovery
                holdings.pop(symbol, None)
                ledger.append(
                    {
                        "session": session,
                        "symbol": symbol,
                        "event": "delisting",
                        "shares": -shares,
                        "price": np.nan,
                        "notional": recovery,
                        "cost": 0.0,
                    }
                )
        return cash

    @staticmethod
    def _require_prices_for_holdings(
        day: pd.DataFrame,
        holdings: dict[str, float],
        previous_close: dict[str, float],
        session: pd.Timestamp,
    ) -> None:
        missing = [symbol for symbol in holdings if symbol not in day.index]
        if missing:
            raise ValueError(
                f"held positions are absent from the session table on {session}; "
                f"provide a halted/delisted status row: {missing[:5]}"
            )
        invalid = [
            symbol
            for symbol in holdings
            if not np.isfinite(day.loc[symbol, "close"]) or day.loc[symbol, "close"] <= 0
        ]
        if invalid:
            raise ValueError(f"held positions have invalid valuation prices: {invalid[:5]}")

    def _unavailable_reason(self, row: pd.Series | None, target: float) -> str | None:
        if row is None:
            return "symbol_not_in_session"
        if "is_tradable" in row and not bool(row["is_tradable"]):
            return "not_tradable"
        if "is_halted" in row and bool(row["is_halted"]):
            return "halted"
        if target > 0 and "buy_locked" in row and bool(row["buy_locked"]):
            return "limit_up_locked"
        if target <= 0 and "sell_locked" in row and bool(row["sell_locked"]):
            return "limit_down_locked"
        if target < 0 and "is_shortable" in row and not bool(row["is_shortable"]):
            return "not_shortable"
        if target < 0 and "is_shortable" not in row:
            return "borrow_status_missing"
        return None

    def _round_shares(self, shares: float) -> float:
        if self.rules.fractional_shares:
            return float(shares)
        lot = max(int(self.rules.round_lot), 1)
        return float(np.trunc(shares / lot) * lot)

    @staticmethod
    def _validate_inputs(signals: pd.DataFrame, bars: pd.DataFrame) -> None:
        missing_signals = sorted({"signal_session", "symbol", "target_weight"} - set(signals))
        missing_bars = sorted(
            {"session", "symbol", "open", "close", "is_tradable"} - set(bars)
        )
        if missing_signals or missing_bars:
            raise ValueError(
                f"execution inputs missing fields: signals={missing_signals}, bars={missing_bars}"
            )
        if bars.duplicated(["session", "symbol"]).any():
            raise ValueError("bars contain duplicate session/symbol rows")
