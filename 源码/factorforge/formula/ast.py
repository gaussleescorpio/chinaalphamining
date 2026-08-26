from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from factorforge.contracts import CandidateRecord

COMMUTATIVE = frozenset({"add", "multiply"})


@dataclass(frozen=True)
class FormulaNode:
    op: str
    children: tuple["FormulaNode", ...] = ()
    params: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def atom(cls, atom_id: str) -> "FormulaNode":
        return cls("atom", params=(("atom_id", atom_id),))

    def canonical(self) -> Mapping[str, Any]:
        children = [child.canonical() for child in self.children]
        if (
            self.op in {"cs_rank", "cs_zscore"}
            and children
            and children[0].get("op") == self.op
        ):
            return children[0]
        if self.op in COMMUTATIVE:
            children.sort(
                key=lambda value: json.dumps(
                    value, sort_keys=True, separators=(",", ":")
                )
            )
        return {
            "op": self.op,
            "children": children,
            "params": dict(sorted(self.params)),
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.canonical(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def atoms(self) -> tuple[str, ...]:
        if self.op == "atom":
            return (str(dict(self.params)["atom_id"]),)
        return tuple(
            sorted({atom for child in self.children for atom in child.atoms()})
        )

    def complexity(self) -> int:
        return 1 + sum(child.complexity() for child in self.children)

    def maximum_lookback(self) -> int:
        own = int(dict(self.params).get("window", 0))
        child = max((item.maximum_lookback() for item in self.children), default=0)
        return own + child


def candidate_record(
    node: FormulaNode,
    *,
    family: str,
    unit: str,
    generation_batch: str,
    seed: int,
) -> CandidateRecord:
    digest = node.sha256()
    return CandidateRecord(
        candidate_id=f"ff_{digest[:24]}",
        formula_json=node.canonical_json(),
        formula_sha256=digest,
        atoms=node.atoms(),
        family=family,
        complexity=node.complexity(),
        maximum_lookback=node.maximum_lookback(),
        unit=unit,
        generation_batch=generation_batch,
        seed=seed,
    )
