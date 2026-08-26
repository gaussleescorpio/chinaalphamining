from __future__ import annotations

import importlib
from dataclasses import dataclass, replace
from typing import Protocol, Sequence

from factorforge.contracts import AtomSpec, CandidateRecord
from factorforge.formula.ast import FormulaNode, candidate_record


@dataclass(frozen=True)
class PluginContext:
    atoms: tuple[AtomSpec, ...]
    windows: tuple[int, ...]
    seed: int


@dataclass(frozen=True)
class PluginFormula:
    node: FormulaNode
    family: str
    unit: str


class FactorPlugin(Protocol):
    plugin_id: str

    def generate(self, context: PluginContext) -> Sequence[PluginFormula]: ...


class FactorPluginRegistry:
    """Explicit label-blind extension point for custom mathematical families."""

    def __init__(self) -> None:
        self._plugins: dict[str, FactorPlugin] = {}

    def register(self, plugin: FactorPlugin) -> None:
        plugin_id = str(plugin.plugin_id).strip()
        if not plugin_id or plugin_id in self._plugins:
            raise ValueError(f"invalid or duplicate factor plugin id: {plugin_id!r}")
        self._plugins[plugin_id] = plugin

    def records(self, context: PluginContext) -> list[CandidateRecord]:
        output: list[CandidateRecord] = []
        for plugin_id in sorted(self._plugins):
            for formula in self._plugins[plugin_id].generate(context):
                output.append(
                    candidate_record(
                        formula.node,
                        family=f"plugin:{plugin_id}:{formula.family}",
                        unit=formula.unit,
                        generation_batch=f"plugin:{plugin_id}",
                        seed=context.seed,
                    )
                )
        return output


def load_plugin_records(
    module_names: Sequence[str],
    atoms: Sequence[AtomSpec],
    windows: Sequence[int],
    seed: int,
) -> list[CandidateRecord]:
    registry = FactorPluginRegistry()
    for module_name in module_names:
        module = importlib.import_module(module_name)
        registration = getattr(module, "register_factor_plugins", None)
        if not callable(registration):
            raise ValueError(
                f"plugin module {module_name!r} must define register_factor_plugins(registry)"
            )
        registration(registry)
    atom_lags = {atom.atom_id: atom.available_lag for atom in atoms}
    return [
        replace(
            record,
            maximum_lookback=record.maximum_lookback
            + max((atom_lags[name] for name in record.atoms), default=0),
        )
        for record in registry.records(
            PluginContext(tuple(atoms), tuple(windows), int(seed))
        )
    ]
