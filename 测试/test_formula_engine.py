from __future__ import annotations

import json

import numpy as np
import pytest

from factorforge.contracts import AtomSpec
from factorforge.formula import CandidateGenerator, FormulaEvaluator, FormulaNode


def test_commutative_formula_has_one_identity() -> None:
    left = FormulaNode.atom("a")
    right = FormulaNode.atom("b")
    assert (
        FormulaNode("add", (left, right)).sha256()
        == FormulaNode("add", (right, left)).sha256()
    )
    assert (
        FormulaNode("subtract", (left, right)).sha256()
        != FormulaNode("subtract", (right, left)).sha256()
    )


def test_generator_is_label_blind_deterministic_and_unique() -> None:
    atoms = [AtomSpec("a", "a", "x", "a"), AtomSpec("b", "b", "x", "b")]
    first = list(CandidateGenerator(atoms, [3, 5], 7).stream(200))
    second = list(CandidateGenerator(atoms, [3, 5], 7).stream(200))
    assert [item.formula_sha256 for item in first] == [
        item.formula_sha256 for item in second
    ]
    assert len(first) == len({item.formula_sha256 for item in first}) == 200
    assert all("future" not in item.formula_json.lower() for item in first)


def test_generator_rejects_negative_limit() -> None:
    atoms = [AtomSpec("a", "a", "x", "a")]
    with pytest.raises(ValueError, match="limit must be nonnegative"):
        list(CandidateGenerator(atoms, [3], 7).stream(-1))


def test_formula_evaluator_is_causal() -> None:
    values = np.arange(30, dtype=float).reshape(10, 3)
    node = FormulaNode("delta", (FormulaNode.atom("a"),), (("window", 2),))
    evaluated = FormulaEvaluator({"a": values}).evaluate(node)
    changed = values.copy()
    changed[-1] = 1_000_000.0
    reevaluated = FormulaEvaluator({"a": changed}).evaluate(node)
    np.testing.assert_allclose(evaluated[:-1], reevaluated[:-1], equal_nan=True)


def test_batch_evaluation_reuses_shared_nodes() -> None:
    values = np.arange(40, dtype=float).reshape(10, 4)
    evaluator = FormulaEvaluator({"x": values}, cache_entries=8)
    atom = FormulaNode.atom("x")
    formulas = [
        FormulaNode("rolling_mean", (atom,), (("window", 3),)).canonical_json(),
        FormulaNode("delta", (atom,), (("window", 2),)).canonical_json(),
    ]
    results = list(evaluator.evaluate_many(formulas))
    assert len(results) == 2
    assert evaluator.cache_report()["hits"] >= 1


def test_batch_pins_common_transform_and_returns_independent_outputs() -> None:
    values = np.arange(40, dtype=float).reshape(10, 4)
    evaluator = FormulaEvaluator({"x": values}, cache_entries=4)
    atom = FormulaNode.atom("x")
    shared = FormulaNode("rolling_mean", (atom,), (("window", 3),))
    formulas = [
        FormulaNode("delta", (shared,), (("window", 2),)).canonical_json(),
        FormulaNode("signed_log1p", (shared,)).canonical_json(),
    ]
    outputs = list(evaluator.evaluate_many(formulas))
    references = [FormulaEvaluator({"x": values}).evaluate_json(item) for item in formulas]
    for actual, expected in zip(outputs, references, strict=True):
        np.testing.assert_allclose(actual, expected, equal_nan=True)
    outputs[0][3, 0] = 999.0
    assert outputs[1][3, 0] != 999.0
    report = evaluator.cache_report()
    assert report["batch_sessions"] == 1
    assert report["batch_hits"] >= 1
    assert report["batch_pinned_peak"] >= 1


def test_idempotent_cross_sectional_transform_is_canonicalized() -> None:
    atom = FormulaNode.atom("x")
    single = FormulaNode("cs_rank", (atom,))
    repeated = FormulaNode("cs_rank", (single,))
    assert single.canonical_json() == repeated.canonical_json()
