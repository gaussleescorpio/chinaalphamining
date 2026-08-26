from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Mapping

import numpy as np


class CudaFormulaEvaluator:
    """Optional CuPy evaluator. Ranking uses the CPU reference for exact ties."""

    def __init__(
        self,
        atoms: Mapping[str, np.ndarray],
        memory_fraction: float,
        cache_entries: int = 4,
    ):
        try:
            import cupy as cp
        except ImportError as error:
            raise RuntimeError(
                "CUDA backend requested but CuPy is not installed"
            ) from error
        self.cp = cp
        _free, total = cp.cuda.Device().mem_info
        cp.get_default_memory_pool().set_limit(size=int(total * memory_fraction))
        self._atoms = {
            name: cp.asarray(values, dtype=cp.float64) for name, values in atoms.items()
        }
        self._cache_entries = max(0, int(cache_entries))
        self._cache = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0

    def evaluate_json(self, formula_json: str) -> np.ndarray:
        return self.cp.asnumpy(self._evaluate(json.loads(formula_json)))

    def evaluate_many(self, formulas: list[str]):
        """Keep atoms and shared intermediate nodes resident on the GPU."""

        for formula in formulas:
            yield self.evaluate_json(formula)

    def cache_report(self) -> dict[str, int | float]:
        total = self._cache_hits + self._cache_misses
        return {
            "entries": len(self._cache),
            "capacity": self._cache_entries,
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": self._cache_hits / total if total else 0.0,
        }

    def _rolling(self, values, window: int, reducer: str):
        cp = self.cp
        valid = cp.isfinite(values)
        clean = cp.where(valid, values, 0.0)
        cumulative = cp.vstack(
            (cp.zeros((1, values.shape[1])), cp.cumsum(clean, axis=0))
        )
        counts = cp.vstack((cp.zeros((1, values.shape[1])), cp.cumsum(valid, axis=0)))
        sums = cumulative[window:] - cumulative[:-window]
        count = counts[window:] - counts[:-window]
        output = cp.full_like(values, cp.nan)
        if reducer == "mean":
            result = cp.divide(
                sums, count, out=cp.full_like(sums, cp.nan), where=count > 0
            )
        else:
            squared = cp.vstack(
                (cp.zeros((1, values.shape[1])), cp.cumsum(clean * clean, axis=0))
            )
            sum_squares = squared[window:] - squared[:-window]
            variance = cp.divide(
                sum_squares
                - cp.divide(
                    sums * sums, count, out=cp.zeros_like(sums), where=count > 0
                ),
                count - 1,
                out=cp.full_like(sums, cp.nan),
                where=count > 1,
            )
            result = cp.sqrt(cp.maximum(variance, 0.0))
        output[window - 1 :] = result
        return output

    def _evaluate(self, node):
        cp = self.cp
        key = json.dumps(node, sort_keys=True, separators=(",", ":"))
        if key in self._cache:
            self._cache_hits += 1
            value = self._cache.pop(key)
            self._cache[key] = value
            return value
        self._cache_misses += 1
        result = self._evaluate_uncached(node)
        if self._cache_entries:
            self._cache[key] = result
            while len(self._cache) > self._cache_entries:
                self._cache.popitem(last=False)
        return result

    def _evaluate_uncached(self, node):
        cp = self.cp
        op = str(node["op"])
        params = dict(node.get("params", {}))
        children = [self._evaluate(value) for value in node.get("children", [])]
        if op == "atom":
            return self._atoms[str(params["atom_id"])].copy()
        if op == "signed_log1p":
            return cp.sign(children[0]) * cp.log1p(cp.abs(children[0]))
        if op == "cs_rank":
            from factorforge.formula.operators import _cross_sectional_rank

            return cp.asarray(_cross_sectional_rank(cp.asnumpy(children[0])))
        if op == "cs_zscore":
            mean = cp.nanmean(children[0], axis=1, keepdims=True)
            scale = cp.nanstd(children[0], axis=1, keepdims=True)
            return cp.divide(
                children[0] - mean,
                scale,
                out=cp.full_like(children[0], cp.nan),
                where=scale > 1e-12,
            )
        if op == "delta":
            window = int(params["window"])
            output = cp.full_like(children[0], cp.nan)
            output[window:] = children[0][window:] - children[0][:-window]
            return output
        if op in {"rolling_mean", "rolling_std"}:
            return self._rolling(
                children[0], int(params["window"]), op.removeprefix("rolling_")
            )
        if op == "rolling_zscore":
            mean = self._rolling(children[0], int(params["window"]), "mean")
            std = self._rolling(children[0], int(params["window"]), "std")
            return cp.divide(
                children[0] - mean,
                std,
                out=cp.full_like(mean, cp.nan),
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
            return cp.divide(
                left, right, out=cp.full_like(left, cp.nan), where=cp.abs(right) > 1e-12
            )
        if op == "normalized_difference":
            scale = cp.abs(left) + cp.abs(right)
            return cp.divide(
                left - right, scale, out=cp.zeros_like(left), where=scale > 1e-12
            )
        if op == "cross_projection_residual":
            numerator = cp.nansum(left * right, axis=1, keepdims=True)
            denominator = cp.nansum(right * right, axis=1, keepdims=True)
            beta = cp.divide(
                numerator,
                denominator,
                out=cp.zeros_like(numerator),
                where=denominator > 1e-12,
            )
            return left - beta * right
        raise ValueError(f"unsupported CUDA formula operator: {op}")


def verify_cuda_consistency(
    atoms: Mapping[str, np.ndarray], formulas: list[str], memory_fraction: float
) -> dict[str, object]:
    from factorforge.formula.operators import FormulaEvaluator

    cpu = FormulaEvaluator(atoms)
    try:
        gpu = CudaFormulaEvaluator(atoms, memory_fraction)
    except RuntimeError as error:
        return {
            "available": False,
            "passed": False,
            "reason": str(error),
            "maximum_absolute_error": None,
        }
    errors = []
    for formula in formulas:
        left, right = cpu.evaluate_json(formula), gpu.evaluate_json(formula)
        active = np.isfinite(left) & np.isfinite(right)
        errors.append(
            float(np.max(np.abs(left[active] - right[active])))
            if np.any(active)
            else 0.0
        )
    maximum = max(errors, default=0.0)
    return {
        "available": True,
        "passed": maximum <= 1e-10,
        "reason": "OK" if maximum <= 1e-10 else "NUMERIC_MISMATCH",
        "maximum_absolute_error": maximum,
    }
