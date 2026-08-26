from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import pandas as pd


@dataclass(frozen=True)
class ExtensionContext:
    decision_time_column: str
    symbol_column: str
    metadata: Mapping[str, str]


class ExternalFeatureProvider(Protocol):
    """Optional PIT-safe external data provider. Implementations stay outside core."""

    provider_id: str

    def enrich(self, frame: pd.DataFrame, context: ExtensionContext) -> pd.DataFrame: ...


class ForecastModelPlugin(Protocol):
    """Optional model boundary; the factor engine does not prescribe a library."""

    model_id: str

    def fit(self, features: pd.DataFrame, target: pd.Series) -> object: ...

    def predict(self, model: object, features: pd.DataFrame) -> pd.Series: ...


class ExtensionRegistry:
    def __init__(self) -> None:
        self._data: dict[str, ExternalFeatureProvider] = {}
        self._models: dict[str, ForecastModelPlugin] = {}

    def register_data(self, provider: ExternalFeatureProvider) -> None:
        self._register(self._data, provider.provider_id, provider)

    def register_model(self, plugin: ForecastModelPlugin) -> None:
        self._register(self._models, plugin.model_id, plugin)

    @staticmethod
    def _register(store: dict[str, object], key: str, value: object) -> None:
        clean = str(key).strip()
        if not clean or clean in store:
            raise ValueError(f"invalid or duplicate extension id: {clean!r}")
        store[clean] = value

    @property
    def data_providers(self) -> tuple[str, ...]:
        return tuple(sorted(self._data))

    @property
    def model_plugins(self) -> tuple[str, ...]:
        return tuple(sorted(self._models))
