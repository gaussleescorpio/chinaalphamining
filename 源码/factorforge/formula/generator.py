from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from dataclasses import replace

import numpy as np

from factorforge.contracts import AtomSpec, CandidateRecord
from factorforge.formula.ast import FormulaNode, candidate_record


_SAME_UNIT_BINARY_OPERATORS = (
    "add",
    "subtract",
    "normalized_difference",
    "cross_projection_residual",
)


def _allowed_binary_operators(left_unit: str, right_unit: str) -> tuple[str, ...]:
    """Return the typed binary grammar without consulting labels or data values."""

    base = ("multiply", "safe_divide")
    return base + _SAME_UNIT_BINARY_OPERATORS if left_unit == right_unit else base


def _binary_result_unit(operator: str, left_unit: str, right_unit: str) -> str:
    """Resolve a binary formula unit; reject operators outside the public grammar."""

    if operator in {"add", "subtract", "cross_projection_residual"}:
        return left_unit
    if operator == "multiply":
        return f"({left_unit}*{right_unit})"
    if operator == "safe_divide":
        return f"({left_unit}/{right_unit})"
    if operator == "normalized_difference":
        return "unitless"
    raise ValueError(f"unsupported binary operator: {operator}")


class CandidateGenerator:
    """Deterministic, label-blind candidate stream with canonical deduplication."""

    def __init__(self, atoms: Sequence[AtomSpec], windows: Sequence[int], seed: int):
        if not atoms or not windows:
            raise ValueError("atoms and windows must be nonempty")
        self.atoms = tuple(sorted(atoms, key=lambda atom: atom.atom_id))
        self.windows = tuple(sorted(set(int(value) for value in windows)))
        self.seed = int(seed)

    def stream(
        self, limit: int, generation_batch: str = "initial"
    ) -> Iterable[CandidateRecord]:
        """Yield at most ``limit`` unique formulas in a deterministic order.

        Formula construction is label-blind: only atom metadata, windows and the
        configured random seed are available to this method.
        """

        if limit < 0:
            raise ValueError("limit must be nonnegative")
        seen: set[str] = set()
        emitted = 0
        atom_lags = {atom.atom_id: atom.available_lag for atom in self.atoms}

        def emit(node: FormulaNode, family: str, unit: str) -> CandidateRecord | None:
            nonlocal emitted
            record = candidate_record(
                node,
                family=family,
                unit=unit,
                generation_batch=generation_batch,
                seed=self.seed,
            )
            record = replace(
                record,
                maximum_lookback=record.maximum_lookback
                + max((atom_lags[name] for name in record.atoms), default=0),
            )
            if record.formula_sha256 in seen or emitted >= limit:
                return None
            seen.add(record.formula_sha256)
            emitted += 1
            return record

        atom_nodes = {
            atom.atom_id: FormulaNode.atom(atom.atom_id) for atom in self.atoms
        }
        for atom in self.atoms:
            record = emit(atom_nodes[atom.atom_id], "raw_atom", atom.unit)
            if record:
                yield record
            for op in ("signed_log1p", "cs_rank", "cs_zscore"):
                output_unit = (
                    f"signed_log({atom.unit})" if op == "signed_log1p" else "unitless"
                )
                record = emit(
                    FormulaNode(op, (atom_nodes[atom.atom_id],)),
                    f"unary_{op}",
                    output_unit,
                )
                if record:
                    yield record
            for window in self.windows:
                for op in ("delta", "rolling_mean", "rolling_std", "rolling_zscore"):
                    node = FormulaNode(
                        op, (atom_nodes[atom.atom_id],), (("window", window),)
                    )
                    record = emit(
                        node,
                        f"temporal_{op}",
                        atom.unit if op != "rolling_zscore" else "unitless",
                    )
                    if record:
                        yield record
                    if emitted >= limit:
                        return

        for left, right in itertools.combinations(self.atoms, 2):
            left_node, right_node = atom_nodes[left.atom_id], atom_nodes[right.atom_id]
            for op in _allowed_binary_operators(left.unit, right.unit):
                node = FormulaNode(op, (left_node, right_node))
                unit = _binary_result_unit(op, left.unit, right.unit)
                record = emit(node, f"relational_{op}", unit)
                if record:
                    yield record
                if emitted >= limit:
                    return
            for window in (self.windows if left.unit == right.unit else ()):
                left_delta = FormulaNode("delta", (left_node,), (("window", window),))
                right_delta = FormulaNode("delta", (right_node,), (("window", window),))
                node = FormulaNode("normalized_difference", (left_delta, right_delta))
                record = emit(node, "multiscale_relative_change", "unitless")
                if record:
                    yield record
                if emitted >= limit:
                    return

        # Continue through the typed grammar until the requested catalog size is
        # reached. The seed fixes the sequence; labels are neither accepted nor
        # available to this generator.
        rng = np.random.default_rng(self.seed)
        attempts = 0
        atom_values = tuple(
            (atom_nodes[atom.atom_id], atom.unit) for atom in self.atoms
        )
        unary_ops = ("signed_log1p", "cs_rank", "cs_zscore")
        temporal_ops = ("delta", "rolling_mean", "rolling_std", "rolling_zscore")
        def transformed(base: FormulaNode, unit: str) -> tuple[FormulaNode, str]:
            mode = int(rng.integers(0, 3))
            if mode == 0:
                op = unary_ops[int(rng.integers(len(unary_ops)))]
                return FormulaNode(op, (base,)), (
                    f"signed_log({unit})" if op == "signed_log1p" else "unitless"
                )
            if mode == 1:
                op = temporal_ops[int(rng.integers(len(temporal_ops)))]
                return FormulaNode(
                    op,
                    (base,),
                    (("window", self.windows[int(rng.integers(len(self.windows)))]),),
                ), ("unitless" if op == "rolling_zscore" else unit)
            return base, unit

        while emitted < limit:
            attempts += 1
            if attempts > max(limit * 30, 10_000):
                raise RuntimeError(
                    "typed grammar could not produce enough unique candidates"
                )
            left, left_unit = transformed(
                *atom_values[int(rng.integers(len(atom_values)))]
            )
            right, right_unit = transformed(
                *atom_values[int(rng.integers(len(atom_values)))]
            )
            allowed = _allowed_binary_operators(left_unit, right_unit)
            op = allowed[int(rng.integers(len(allowed)))]
            node = FormulaNode(op, (left, right))
            result_unit = _binary_result_unit(op, left_unit, right_unit)
            if int(rng.integers(0, 3)) == 0:
                third, third_unit = transformed(
                    *atom_values[int(rng.integers(len(atom_values)))]
                )
                allowed = _allowed_binary_operators(result_unit, third_unit)
                op = allowed[int(rng.integers(len(allowed)))]
                node = FormulaNode(op, (node, third))
                result_unit = _binary_result_unit(op, result_unit, third_unit)
            record = emit(node, "typed_mathematical_expansion", result_unit)
            if record:
                yield record
