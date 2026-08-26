from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping, Sequence

from factorforge.contracts import AtomSpec, CandidateRecord
from factorforge.formula.ast import FormulaNode, candidate_record

REQUIRED_FIELDS = frozenset(
    {
        "card_id",
        "mainline",
        "name",
        "market_mechanism",
        "observable_proxies",
        "prediction_target",
        "time_scales",
        "expected_direction",
        "contrary_explanations",
        "failure_conditions",
        "cost_conditions",
        "abstention_conditions",
        "adjudication_data",
        "atom_keywords",
        "combiners",
    }
)


@dataclass(frozen=True)
class MechanismCard:
    card_id: str
    mainline: str
    name: str
    market_mechanism: str
    observable_proxies: tuple[str, ...]
    prediction_target: str
    time_scales: tuple[int, ...]
    expected_direction: str
    contrary_explanations: tuple[str, ...]
    failure_conditions: tuple[str, ...]
    cost_conditions: tuple[str, ...]
    abstention_conditions: tuple[str, ...]
    adjudication_data: tuple[str, ...]
    atom_keywords: tuple[str, ...]
    combiners: tuple[str, ...]
    required_keyword_groups: tuple[tuple[str, ...], ...]


def load_mechanism_cards(path: str | Path | None) -> tuple[MechanismCard, ...]:
    if not path:
        return ()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("mechanism card file must contain a JSON list")
    cards: list[MechanismCard] = []
    seen: set[str] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            raise ValueError(f"mechanism card {index} is not an object")
        missing = sorted(REQUIRED_FIELDS - set(item))
        if missing:
            raise ValueError(f"mechanism card {index} is missing fields: {missing}")
        card_id = str(item["card_id"])
        if card_id in seen:
            raise ValueError(f"duplicate mechanism card id: {card_id}")
        seen.add(card_id)
        cards.append(
            MechanismCard(
                card_id=card_id,
                mainline=str(item["mainline"]),
                name=str(item["name"]),
                market_mechanism=str(item["market_mechanism"]),
                observable_proxies=tuple(map(str, item["observable_proxies"])),
                prediction_target=str(item["prediction_target"]),
                time_scales=tuple(map(int, item["time_scales"])),
                expected_direction=str(item["expected_direction"]),
                contrary_explanations=tuple(map(str, item["contrary_explanations"])),
                failure_conditions=tuple(map(str, item["failure_conditions"])),
                cost_conditions=tuple(map(str, item["cost_conditions"])),
                abstention_conditions=tuple(map(str, item["abstention_conditions"])),
                adjudication_data=tuple(map(str, item["adjudication_data"])),
                atom_keywords=tuple(map(str, item["atom_keywords"])),
                combiners=tuple(map(str, item["combiners"])),
                required_keyword_groups=tuple(
                    tuple(map(str, group))
                    for group in item.get("required_keyword_groups", ())
                ),
            )
        )
    return tuple(cards)


def generate_card_candidates(
    cards: Sequence[MechanismCard],
    atoms: Sequence[AtomSpec],
    seed: int,
    candidates_per_card: int = 192,
) -> tuple[CandidateRecord, ...]:
    """Create deterministic mechanism-constrained seeds without reading labels."""

    records: dict[str, CandidateRecord] = {}
    for card in cards:
        card_offset = int(
            hashlib.sha256(card.card_id.encode("utf-8")).hexdigest()[:8], 16
        )
        searchable = {
            atom.atom_id: (
                atom.atom_id + " " + atom.family + " " + " ".join(atom.lineage)
            ).lower()
            for atom in atoms
        }
        grouped_matches = [
            [
                atom
                for atom in atoms
                if any(keyword.lower() in searchable[atom.atom_id] for keyword in group)
            ]
            for group in card.required_keyword_groups
        ]
        if grouped_matches and any(not group for group in grouped_matches):
            continue
        matched = [
            atom
            for atom in atoms
            if any(
                keyword.lower() in searchable[atom.atom_id]
                for keyword in card.atom_keywords
            )
        ]
        if grouped_matches:
            required_ids = {
                atom.atom_id for group in grouped_matches for atom in group[:12]
            }
            matched = [atom for atom in matched if atom.atom_id in required_ids]
        matched = matched[:32]
        if len(matched) < 2:
            continue
        attempts = 0
        index = 0
        card_count = 0
        while card_count < candidates_per_card and attempts < candidates_per_card * 20:
            attempts += 1
            shifted_index = index + card_offset
            if len(grouped_matches) >= 2:
                left_group = grouped_matches[shifted_index % len(grouped_matches)]
                right_group = grouped_matches[
                    (shifted_index + 1) % len(grouped_matches)
                ]
                left = left_group[shifted_index % len(left_group)]
                right = right_group[(shifted_index * 7 + 1) % len(right_group)]
            else:
                left = matched[shifted_index % len(matched)]
                right = matched[(shifted_index * 7 + 1) % len(matched)]
            if left.atom_id == right.atom_id:
                index += 1
                continue
            op = card.combiners[shifted_index % len(card.combiners)]
            if (
                op
                in {
                    "add",
                    "subtract",
                    "normalized_difference",
                    "cross_projection_residual",
                }
                and left.unit != right.unit
            ):
                op = "safe_divide"
            left_node = FormulaNode.atom(left.atom_id)
            right_node = FormulaNode.atom(right.atom_id)
            scale = card.time_scales[
                (shifted_index // max(1, len(matched))) % len(card.time_scales)
            ]
            transform = (shifted_index // max(1, len(card.combiners))) % 4
            if scale < 2:
                transform = 0
            if transform == 1:
                left_node = FormulaNode("delta", (left_node,), (("window", scale),))
                right_node = FormulaNode("delta", (right_node,), (("window", scale),))
            elif transform == 2:
                left_node = FormulaNode(
                    "rolling_zscore", (left_node,), (("window", scale),)
                )
                right_node = FormulaNode(
                    "rolling_zscore", (right_node,), (("window", scale),)
                )
            elif transform == 3:
                left_node = FormulaNode(
                    "rolling_mean", (left_node,), (("window", scale),)
                )
                right_node = FormulaNode(
                    "rolling_mean", (right_node,), (("window", scale),)
                )
            node = FormulaNode(op, (left_node, right_node))
            unit = "unitless"
            if op in {"add", "subtract", "cross_projection_residual"}:
                unit = left.unit
            elif op == "multiply":
                unit = f"({left.unit}*{right.unit})"
            elif op == "safe_divide":
                unit = f"({left.unit}/{right.unit})"
            record = candidate_record(
                node,
                family=f"mechanism:{card.card_id}:{card.mainline}",
                unit=unit,
                generation_batch="mechanism_constrained_seed",
                seed=seed,
            )
            if record.formula_sha256 not in records:
                records[record.formula_sha256] = record
                card_count += 1
            index += 1

        # Some cards intentionally share two-atom proxies.  If all of their
        # simple expressions already belong to an earlier card, create a
        # bounded three-atom standardized relation instead of silently dropping
        # the card or manufacturing a card-specific identifier.
        fallback_target = min(candidates_per_card, 256)
        fallback_attempts = 0
        while card_count < fallback_target and fallback_attempts < 10_000:
            shifted_index = card_offset + fallback_attempts
            left = matched[shifted_index % len(matched)]
            right = matched[(shifted_index * 5 + 1) % len(matched)]
            third = matched[(shifted_index * 11 + 2) % len(matched)]
            if len({left.atom_id, right.atom_id, third.atom_id}) < 2:
                fallback_attempts += 1
                continue
            first_window = max(2, card.time_scales[shifted_index % len(card.time_scales)])
            second_window = max(
                2, card.time_scales[(shifted_index + 1) % len(card.time_scales)]
            )
            third_window = max(
                2, card.time_scales[(shifted_index + 2) % len(card.time_scales)]
            )
            left_node = FormulaNode(
                "rolling_zscore",
                (FormulaNode.atom(left.atom_id),),
                (("window", first_window),),
            )
            right_node = FormulaNode(
                "rolling_zscore",
                (FormulaNode.atom(right.atom_id),),
                (("window", second_window),),
            )
            third_node = FormulaNode(
                "rolling_zscore",
                (FormulaNode.atom(third.atom_id),),
                (("window", third_window),),
            )
            inner = FormulaNode("normalized_difference", (left_node, right_node))
            outer_op = ("add", "subtract", "multiply", "safe_divide")[
                shifted_index % 4
            ]
            node = FormulaNode(outer_op, (inner, third_node))
            record = candidate_record(
                node,
                family=f"mechanism:{card.card_id}:{card.mainline}",
                unit="unitless",
                generation_batch="mechanism_constrained_three_atom_seed",
                seed=seed,
            )
            if record.formula_sha256 not in records:
                records[record.formula_sha256] = record
                card_count += 1
            fallback_attempts += 1
    return tuple(records.values())
