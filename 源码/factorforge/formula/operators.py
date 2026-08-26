from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import contextmanager
from typing import Iterator

import numpy as np
from scipy.stats import rankdata

from factorforge.formula.ast import FormulaNode


def _rolling(values: np.ndarray, window: int, reducer: str) -> np.ndarray:
    if window < 2:
        raise ValueError("rolling window must be at least two")
    valid = np.isfinite(values)
    clean = np.where(valid, values, 0.0)
    cumulative = np.vstack((np.zeros((1, values.shape[1])), np.cumsum(clean, axis=0)))
    counts = np.vstack((np.zeros((1, values.shape[1])), np.cumsum(valid, axis=0)))
    sums = cumulative[window:] - cumulative[:-window]
    count = counts[window:] - counts[:-window]
    output = np.full_like(values, np.nan, dtype=np.float64)
    if reducer == "mean":
        output[window - 1 :] = np.divide(
            sums, count, out=np.full_like(sums, np.nan), where=count > 0
        )
    elif reducer == "std":
        squared = np.vstack(
            (np.zeros((1, values.shape[1])), np.cumsum(clean * clean, axis=0))
        )
        sum_squares = squared[window:] - squared[:-window]
        variance = np.divide(
            sum_squares
            - np.divide(sums * sums, count, out=np.zeros_like(sums), where=count > 0),
            count - 1,
            out=np.full_like(sums, np.nan),
            where=count > 1,
        )
        output[window - 1 :] = np.sqrt(np.maximum(variance, 0.0))
    else:
        raise ValueError(reducer)
    return output


def _cross_sectional_rank(values: np.ndarray) -> np.ndarray:
    output = np.full_like(values, np.nan, dtype=np.float64)
    for index, row in enumerate(values):
        active = np.isfinite(row)
        if np.sum(active) < 2:
            continue
        ranks = rankdata(row[active], method="average")
        output[index, active] = (ranks - 1.0) / (len(ranks) - 1.0) - 0.5
    return output


def _cross_sectional_zscore(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    count = finite.sum(axis=1, keepdims=True)
    clean = np.where(finite, values, 0.0)
    mean = np.divide(
        clean.sum(axis=1, keepdims=True),
        count,
        out=np.zeros((values.shape[0], 1), dtype=float),
        where=count > 0,
    )
    squared = np.where(finite, (values - mean) ** 2, 0.0).sum(axis=1, keepdims=True)
    variance = np.divide(
        squared,
        count,
        out=np.zeros((values.shape[0], 1), dtype=float),
        where=count > 0,
    )
    scale = np.sqrt(variance)
    return np.divide(
        values - mean, scale, out=np.full_like(values, np.nan), where=scale > 1e-12
    )


class FormulaEvaluator:
    """Evaluate formula trees using only current and historical observations."""

    def __init__(self, atoms: Mapping[str, np.ndarray], cache_entries: int = 32):
        self._atoms = {
            name: np.asarray(values, dtype=np.float64) for name, values in atoms.items()
        }
        shapes = {values.shape for values in self._atoms.values()}
        if len(shapes) != 1:
            raise ValueError("all atom matrices must share one shape")
        self._cache_entries = max(0, int(cache_entries))
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._batch_cache: dict[str, np.ndarray] = {}
        self._batch_keys: set[str] = set()
        self._batch_plans: dict[str, Mapping[str, object]] = {}
        self._batch_hits = 0
        self._batch_sessions = 0
        self._batch_pinned_peak = 0

    def evaluate_json(self, formula_json: str) -> np.ndarray:
        node = self._batch_plans.get(formula_json)
        if node is None:
            node = json.loads(formula_json)
        return self._evaluate_mapping(node).copy()

    def evaluate(self, node: FormulaNode) -> np.ndarray:
        return self._evaluate_mapping(node.canonical()).copy()

    def evaluate_many(self, formulas: list[str]):
        """Evaluate a bounded formula batch while reusing shared DAG nodes."""

        with self.shared_batch(formulas):
            for formula in formulas:
                yield self.evaluate_json(formula)

    @contextmanager
    def shared_batch(self, formulas: list[str]) -> Iterator[None]:
        """Pin frequently reused DAG nodes for one memory-bounded batch."""

        if self._batch_plans:
            raise RuntimeError("nested formula batch sessions are not supported")
        plans = {formula: json.loads(formula) for formula in dict.fromkeys(formulas)}
        counts: dict[str, int] = {}
        complexity: dict[str, int] = {}
        operations: dict[str, str] = {}

        def visit(node: Mapping[str, object]) -> int:
            size = 1 + sum(visit(child) for child in node.get("children", []))
            key = self._node_key(node)
            counts[key] = counts.get(key, 0) + 1
            complexity[key] = size
            operations[key] = str(node["op"])
            return size

        for plan in plans.values():
            visit(plan)
        reusable_atoms = {
            key for key, count in counts.items() if count > 1 and operations[key] == "atom"
        }
        reusable = [
            key
            for key, count in counts.items()
            if count > 1 and operations[key] != "atom"
        ]
        reusable.sort(
            key=lambda key: (counts[key] * complexity[key], counts[key], complexity[key]),
            reverse=True,
        )
        self._cache.clear()
        # Atom entries are references to already-resident inputs and consume no
        # additional matrix allocation, so they do not use intermediate slots.
        self._batch_keys = reusable_atoms | set(reusable[: self._cache_entries])
        self._batch_plans = plans
        self._batch_sessions += 1
        self._batch_pinned_peak = max(
            self._batch_pinned_peak, min(len(reusable), self._cache_entries)
        )
        try:
            yield
        finally:
            retained = list(self._batch_cache.items())[-self._cache_entries :]
            self._cache = OrderedDict(retained)
            self._batch_cache = {}
            self._batch_keys = set()
            self._batch_plans = {}

    def cache_report(self) -> dict[str, int | float]:
        reported_hits = self._cache_hits + self._batch_hits
        total = reported_hits + self._cache_misses
        return {
            "entries": len(self._cache),
            "capacity": self._cache_entries,
            "hits": reported_hits,
            "misses": self._cache_misses,
            "hit_rate": reported_hits / total if total else 0.0,
            "batch_sessions": self._batch_sessions,
            "batch_hits": self._batch_hits,
            "batch_pinned_peak": self._batch_pinned_peak,
        }

    @staticmethod
    def _node_key(node: Mapping[str, object]) -> str:
        return json.dumps(node, sort_keys=True, separators=(",", ":"))

    def _evaluate_mapping(self, node: Mapping[str, object]) -> np.ndarray:
        cache_key = self._node_key(node)
        if cache_key in self._batch_cache:
            self._batch_hits += 1
            return self._batch_cache[cache_key]
        if cache_key in self._cache:
            self._cache_hits += 1
            value = self._cache.pop(cache_key)
            self._cache[cache_key] = value
            return value
        self._cache_misses += 1
        result = self._evaluate_uncached(node)
        if cache_key in self._batch_keys:
            self._batch_cache[cache_key] = result
        elif self._cache_entries and not self._batch_plans:
            self._cache[cache_key] = result
            while len(self._cache) > self._cache_entries:
                self._cache.popitem(last=False)
        return result

    def _evaluate_uncached(self, node: Mapping[str, object]) -> np.ndarray:
        op = str(node["op"])
        params = dict(node.get("params", {}))
        children = [self._evaluate_mapping(value) for value in node.get("children", [])]
        if op == "atom":
            atom_id = str(params["atom_id"])
            if atom_id not in self._atoms:
                raise KeyError(f"unknown atom: {atom_id}")
            return self._atoms[atom_id]
        if op == "signed_log1p":
            return np.sign(children[0]) * np.log1p(np.abs(children[0]))
        if op == "cs_rank":
            return _cross_sectional_rank(children[0])
        if op == "cs_zscore":
            return _cross_sectional_zscore(children[0])
        if op == "delta":
            window = int(params["window"])
            output = np.full_like(children[0], np.nan)
            output[window:] = children[0][window:] - children[0][:-window]
            return output
        if op in {"rolling_mean", "rolling_std"}:
            return _rolling(
                children[0], int(params["window"]), op.removeprefix("rolling_")
            )
        if op == "rolling_zscore":
            mean = _rolling(children[0], int(params["window"]), "mean")
            std = _rolling(children[0], int(params["window"]), "std")
            return np.divide(
                children[0] - mean,
                std,
                out=np.full_like(mean, np.nan),
                where=std > 1e-12,
            )
        left, right = children
        if op == "add":
            return left + right
        if op == "subtract":
            return left - right
        if op == "multiply":
            return left * right
        if op == "safe_divide":
            return np.divide(
                left, right, out=np.full_like(left, np.nan), where=np.abs(right) > 1e-12
            )
        if op == "normalized_difference":
            scale = np.abs(left) + np.abs(right)
            return np.divide(
                left - right, scale, out=np.zeros_like(left), where=scale > 1e-12
            )
        if op == "cross_projection_residual":
            numerator = np.nansum(left * right, axis=1, keepdims=True)
            denominator = np.nansum(right * right, axis=1, keepdims=True)
            beta = np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator),
                where=denominator > 1e-12,
            )
            return left - beta * right
        raise ValueError(f"unsupported formula operator: {op}")
