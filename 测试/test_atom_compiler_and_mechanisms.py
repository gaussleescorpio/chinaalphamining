from __future__ import annotations

import json
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from factorforge.atoms import AtomCompiler
from factorforge.contracts import PanelData
from factorforge.mechanisms import generate_card_candidates, load_mechanism_cards


def _panel() -> PanelData:
    rng = np.random.default_rng(23)
    shape = (90, 8)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, shape), axis=0))
    panel = PanelData(
        timestamps=pd.date_range("2020-01-01", periods=shape[0], tz="UTC"),
        assets=tuple(f"A{x}" for x in range(shape[1])),
        fields={
            "open": close * (1 + rng.normal(0, 0.002, shape)),
            "high": close * (1 + np.abs(rng.normal(0, 0.005, shape))),
            "low": close * (1 - np.abs(rng.normal(0, 0.005, shape))),
            "close": close,
            "volume": rng.lognormal(12, 1, shape),
            "amount": rng.lognormal(16, 1, shape),
            "trade_count": rng.lognormal(7, 0.6, shape),
            "market_cap": rng.lognormal(22, 1, shape),
        },
    )
    panel.validate()
    return panel


def test_wide_atom_compiler_is_causal_and_multifamily() -> None:
    panel = _panel()
    definitions = {
        name: {"role": name, "unit": "native", "available_lag": 0}
        for name in panel.fields
    }
    compiler = AtomCompiler(definitions, [3, 5, 10], maximum_atoms=2_000)
    first = compiler.compile(panel)
    changed_fields = dict(panel.fields)
    changed_fields["close"] = changed_fields["close"].copy()
    changed_fields["close"][-1] *= 100
    changed = compiler.compile(
        PanelData(panel.timestamps, panel.assets, changed_fields)
    )
    assert len(first.specs) > len(panel.fields) * 10
    assert sum(value > 0 for value in first.family_counts.values()) >= 6
    for atom_id in first.values:
        np.testing.assert_allclose(
            first.values[atom_id][:-1], changed.values[atom_id][:-1], equal_nan=True
        )


def test_mechanism_cards_are_complete_and_generate_constrained_seeds() -> None:
    card_path = Path(__file__).parents[1] / "配置" / "研究卡示例.json"
    cards = load_mechanism_cards(card_path)
    panel = _panel()
    definitions = {
        name: {"role": name, "unit": "native", "available_lag": 0}
        for name in panel.fields
    }
    library = AtomCompiler(definitions, [3, 5, 10], maximum_atoms=2_000).compile(panel)
    records = generate_card_candidates(cards, library.specs, 17)
    assert len(cards) >= 4
    assert len({card.mainline for card in cards}) >= 4
    assert records
    assert all(record.family.startswith("mechanism:") for record in records)
    assert all("future" not in record.formula_json.lower() for record in records)


def test_atom_compiler_handles_all_missing_cross_sections_without_warning() -> None:
    panel = _panel()
    fields = dict(panel.fields)
    fields["close"] = np.full_like(fields["close"], np.nan)
    missing_panel = PanelData(panel.timestamps, panel.assets, fields)
    definitions = {
        name: {"role": name, "unit": "native", "available_lag": 0}
        for name in fields
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        AtomCompiler(definitions, [3, 5], maximum_atoms=500).compile(missing_panel)


def test_overlapping_mechanism_cards_keep_distinct_candidate_coverage() -> None:
    card_path = Path(__file__).parents[1] / "配置" / "研究卡示例.json"
    first = replace(
        load_mechanism_cards(card_path)[0],
        atom_keywords=("close", "volume"),
        required_keyword_groups=(),
    )
    duplicate = replace(first, card_id="OVERLAP_TEST")
    panel = _panel()
    definitions = {
        name: {"role": name, "unit": "native", "available_lag": 0}
        for name in panel.fields
    }
    library = AtomCompiler(definitions, [3, 5, 10], maximum_atoms=2_000).compile(panel)
    records = generate_card_candidates((first, duplicate), library.specs, 17, 64)
    families = {record.family.split(":", 2)[1] for record in records}
    assert families == {first.card_id, duplicate.card_id}
