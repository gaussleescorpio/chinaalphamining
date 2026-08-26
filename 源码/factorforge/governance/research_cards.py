from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from factorforge.contracts import AtomSpec
from factorforge.mechanisms import MechanismCard


@dataclass(frozen=True)
class CardAudit:
    card_id: str
    structurally_complete: bool
    observable: bool
    falsifiable: bool
    label_blind: bool
    distinct_from_card_ids: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def research_ready(self) -> bool:
        return (
            self.structurally_complete
            and self.observable
            and self.falsifiable
            and self.label_blind
        )


_FUTURE_TOKENS = ("future", "forward", "label", "未来收益", "未来路径")


def _atom_text(atom: AtomSpec) -> str:
    return " ".join((atom.atom_id, atom.field, atom.family, *atom.lineage)).lower()


def _matched_atoms(card: MechanismCard, atoms: Sequence[AtomSpec]) -> set[str]:
    keywords = tuple(keyword.lower() for keyword in card.atom_keywords)
    return {
        atom.atom_id
        for atom in atoms
        if any(keyword in _atom_text(atom) for keyword in keywords)
    }


def audit_mechanism_cards(
    cards: Sequence[MechanismCard], atoms: Sequence[AtomSpec]
) -> tuple[CardAudit, ...]:
    """Audit whether a card is computable and falsifiable before labels are read.

    A research card is a hypothesis contract, not proof that an effect exists.
    Empirical support is attached later by the factor evaluation ledger.
    """

    atom_matches = {card.card_id: _matched_atoms(card, atoms) for card in cards}
    audits: list[CardAudit] = []
    for card in cards:
        warnings: list[str] = []
        matched = atom_matches[card.card_id]
        observable = len(matched) >= 2
        if not observable:
            warnings.append("FEWER_THAN_TWO_OBSERVABLE_ATOMS")
        falsifiable = bool(card.failure_conditions and card.contrary_explanations)
        if not falsifiable:
            warnings.append("NO_EXPLICIT_FALSIFICATION")
        card_input_text = " ".join(
            (
                *card.observable_proxies,
                *card.atom_keywords,
            )
        ).lower()
        label_blind = not any(token in card_input_text for token in _FUTURE_TOKENS)
        if not label_blind:
            warnings.append("POSSIBLE_FUTURE_INPUT_REFERENCE")
        structurally_complete = bool(
            card.market_mechanism.strip()
            and card.prediction_target.strip()
            and card.expected_direction.strip()
            and card.time_scales
            and card.cost_conditions
            and card.abstention_conditions
        )
        if not structurally_complete:
            warnings.append("INCOMPLETE_HYPOTHESIS_CONTRACT")
        overlaps = tuple(
            other.card_id
            for other in cards
            if other.card_id != card.card_id
            and matched
            and len(matched & atom_matches[other.card_id]) / len(matched) >= 0.8
        )
        if overlaps:
            warnings.append("HIGH_PROXY_OVERLAP_REQUIRES_EMPIRICAL_DEDUP")
        audits.append(
            CardAudit(
                card_id=card.card_id,
                structurally_complete=structurally_complete,
                observable=observable,
                falsifiable=falsifiable,
                label_blind=label_blind,
                distinct_from_card_ids=overlaps,
                warnings=tuple(warnings),
            )
        )
    return tuple(audits)
