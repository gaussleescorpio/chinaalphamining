from __future__ import annotations

import json
from dataclasses import dataclass, fields
from datetime import date
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ResearchConfig:
    project_name: str
    timestamp_column: str
    asset_column: str
    input_fields: tuple[str, ...]
    development_end: str
    oos_start: str
    oos_end: str
    holding_periods: tuple[int, ...]
    primary_holding_period: int
    periods_per_year: int
    one_way_cost_bps: float
    rolling_windows: tuple[int, ...]
    candidate_limit: int
    batch_size: int
    workers: int
    random_seed: int
    max_memory_fraction: float
    minimum_coverage: float
    maximum_abs_rank_correlation: float
    fdr_alpha: float
    fdr_gate_enabled: bool
    minimum_positive_fold_fraction: float
    minimum_oof_rank_ic: float
    quantile_count: int
    target_pool_size: int
    persist_values: str
    output_root: str
    runtime_profile: str
    gpu_backend: str
    gpu_memory_fraction: float
    gpu_temperature_limit_c: int
    pool_confirmation_fraction: float
    maximum_value_rank_correlation: float
    maximum_reconstruction_r2: float
    membership_column: str | None
    market_contract: str
    resume: bool
    checkpoint_every_batches: int
    atom_definitions_path: str | None
    evaluation_shortlist_size: int
    formula_cache_entries: int
    screening_stride: int
    screening_minimum_positive_block_fraction: float
    plugin_modules: tuple[str, ...]
    parent_pool_path: str | None
    numeric_screening_budget: int
    screening_max_rows: int
    screening_max_assets: int
    target_runtime_minutes: int
    atom_compiler_enabled: bool
    atom_families: tuple[str, ...]
    maximum_compiled_atoms: int
    mechanism_cards_path: str | None
    mechanism_candidates_per_card: int
    mechanism_numeric_quota_per_card: int
    residual_model_version: str
    evidence_provenance: str
    label_horizon_periods: int
    label_return_type: str
    label_start_offset_periods: int
    label_cost_status: str
    label_validation_mode: str
    label_price_field: str | None
    negative_control_gate_enabled: bool
    negative_control_max_abs_rank_ic: float
    negative_control_min_hac_p_value: float
    automatic_oos_evaluation: bool
    development_tail_selection_enabled: bool

    @classmethod
    def from_json(cls, path: str | Path) -> "ResearchConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        defaults = {
            "runtime_profile": "safe",
            "gpu_backend": "cpu",
            "gpu_memory_fraction": 0.45,
            "gpu_temperature_limit_c": 78,
            "pool_confirmation_fraction": 0.25,
            "maximum_value_rank_correlation": 0.90,
            "maximum_reconstruction_r2": 0.80,
            "membership_column": None,
            "market_contract": "generic",
            "resume": False,
            "checkpoint_every_batches": 20,
            "atom_definitions_path": None,
            "evaluation_shortlist_size": 64,
            "formula_cache_entries": 8,
            "screening_stride": 5,
            "screening_minimum_positive_block_fraction": 0.50,
            "plugin_modules": [],
            "parent_pool_path": None,
            "numeric_screening_budget": 20_000,
            "screening_max_rows": 1_500,
            "screening_max_assets": 256,
            "target_runtime_minutes": 20,
            "atom_compiler_enabled": False,
            "atom_families": [
                "price_path",
                "volume_flow",
                "amount_liquidity",
                "volatility_tail",
                "cross_section_centering",
                "residual_relative_structure",
                "fundamental_event",
                "mathematical_shape",
            ],
            "maximum_compiled_atoms": 8_000,
            "mechanism_cards_path": None,
            "mechanism_candidates_per_card": 192,
            "mechanism_numeric_quota_per_card": 24,
            "residual_model_version": "rank_controls_v2",
            "evidence_provenance": "program_recomputation",
        }
        payload = {**defaults, **payload}
        payload.setdefault(
            "label_horizon_periods", int(payload["primary_holding_period"])
        )
        payload.setdefault("label_return_type", "simple_return")
        payload.setdefault("label_start_offset_periods", 1)
        payload.setdefault("label_cost_status", "gross_before_cost")
        payload.setdefault("label_validation_mode", "declared_metadata")
        payload.setdefault("label_price_field", None)
        payload.setdefault("negative_control_gate_enabled", False)
        payload.setdefault("negative_control_max_abs_rank_ic", 0.02)
        payload.setdefault("negative_control_min_hac_p_value", 0.05)
        payload.setdefault("automatic_oos_evaluation", True)
        payload.setdefault("development_tail_selection_enabled", False)
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"unknown configuration fields: {unknown}")
        for key in (
            "input_fields",
            "holding_periods",
            "rolling_windows",
            "plugin_modules",
            "atom_families",
        ):
            payload[key] = tuple(payload[key])
        config = cls(**payload)
        config.validate()
        return config

    def validate(self) -> None:
        if not self.project_name.strip():
            raise ValueError("project_name cannot be empty")
        if not self.output_root.strip():
            raise ValueError("output_root cannot be empty")
        development_end = date.fromisoformat(self.development_end)
        oos_start = date.fromisoformat(self.oos_start)
        oos_end = date.fromisoformat(self.oos_end)
        if not development_end < oos_start <= oos_end:
            raise ValueError(
                "time boundary must satisfy development_end < oos_start <= oos_end"
            )
        if (
            len(set(self.input_fields)) != len(self.input_fields)
            or not self.input_fields
        ):
            raise ValueError("input_fields must be nonempty and unique")
        if any(value < 1 for value in self.holding_periods + self.rolling_windows):
            raise ValueError("holding periods and rolling windows must be positive")
        if self.primary_holding_period not in self.holding_periods:
            raise ValueError("primary_holding_period must be listed in holding_periods")
        if self.label_horizon_periods != self.primary_holding_period:
            raise ValueError(
                "label_horizon_periods must equal primary_holding_period"
            )
        if self.label_return_type not in {"simple_return", "log_return"}:
            raise ValueError("label_return_type must be simple_return or log_return")
        if self.label_start_offset_periods < 0:
            raise ValueError("label_start_offset_periods cannot be negative")
        if self.label_cost_status != "gross_before_cost":
            raise ValueError(
                "labels must be gross_before_cost; execution cost is applied once by the evaluator"
            )
        if self.label_validation_mode not in {
            "declared_metadata",
            "recompute_and_verify",
        }:
            raise ValueError(
                "label_validation_mode must be declared_metadata or recompute_and_verify"
            )
        if self.label_validation_mode == "recompute_and_verify":
            if not self.label_price_field:
                raise ValueError("recompute_and_verify requires label_price_field")
            if self.label_price_field not in self.input_fields:
                raise ValueError("label_price_field must appear in input_fields")
        if not 0.0 <= self.negative_control_max_abs_rank_ic < 1.0:
            raise ValueError("negative_control_max_abs_rank_ic must be in [0, 1)")
        if not 0.0 < self.negative_control_min_hac_p_value < 1.0:
            raise ValueError(
                "negative_control_min_hac_p_value must be between zero and one"
            )
        if self.periods_per_year < 2 or self.one_way_cost_bps < 0.0:
            raise ValueError(
                "periods_per_year must exceed one and cost cannot be negative"
            )
        if self.candidate_limit < 1 or self.batch_size < 1 or self.workers < 1:
            raise ValueError("candidate_limit, batch_size and workers must be positive")
        if self.workers != 1:
            raise ValueError(
                "version 1 uses one shared-panel worker to prevent silent panel duplication"
            )
        for name, value in (
            ("max_memory_fraction", self.max_memory_fraction),
            ("minimum_coverage", self.minimum_coverage),
            ("maximum_abs_rank_correlation", self.maximum_abs_rank_correlation),
            ("fdr_alpha", self.fdr_alpha),
            ("minimum_positive_fold_fraction", self.minimum_positive_fold_fraction),
        ):
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be between zero and one")
        if self.quantile_count < 3:
            raise ValueError("quantile_count must be at least three")
        if not -1.0 < self.minimum_oof_rank_ic < 1.0:
            raise ValueError("minimum_oof_rank_ic must be between -1 and 1")
        if self.target_pool_size < 1:
            raise ValueError("target_pool_size must be positive")
        if self.persist_values not in {"selected_only", "all_evaluated"}:
            raise ValueError("persist_values must be selected_only or all_evaluated")
        if self.market_contract not in {"generic", "a_share_pit"}:
            raise ValueError("market_contract must be generic or a_share_pit")
        if self.market_contract == "a_share_pit" and not self.membership_column:
            raise ValueError("point-in-time market contracts require membership_column")
        if self.runtime_profile not in {"safe", "balanced", "performance", "custom"}:
            raise ValueError(
                "runtime_profile must be safe, balanced, performance or custom"
            )
        if self.gpu_backend not in {"cpu", "auto", "cuda"}:
            raise ValueError("gpu_backend must be cpu, auto or cuda")
        for name, value in (
            ("gpu_memory_fraction", self.gpu_memory_fraction),
            ("pool_confirmation_fraction", self.pool_confirmation_fraction),
            ("maximum_value_rank_correlation", self.maximum_value_rank_correlation),
            ("maximum_reconstruction_r2", self.maximum_reconstruction_r2),
        ):
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be between zero and one")
        if not 50 <= self.gpu_temperature_limit_c <= 90:
            raise ValueError("gpu_temperature_limit_c must be between 50 and 90")
        if self.checkpoint_every_batches < 1:
            raise ValueError("checkpoint_every_batches must be positive")
        if self.evaluation_shortlist_size < self.target_pool_size:
            raise ValueError(
                "evaluation_shortlist_size cannot be smaller than target_pool_size"
            )
        if not 0 <= self.formula_cache_entries <= 128:
            raise ValueError("formula_cache_entries must be between zero and 128")
        if self.screening_stride < 1:
            raise ValueError("screening_stride must be positive")
        if not 0.0 <= self.screening_minimum_positive_block_fraction <= 1.0:
            raise ValueError(
                "screening_minimum_positive_block_fraction must be between zero and one"
            )
        if self.numeric_screening_budget < self.evaluation_shortlist_size:
            raise ValueError(
                "numeric_screening_budget cannot be smaller than evaluation_shortlist_size"
            )
        if self.screening_max_rows < 100 or self.screening_max_assets < 3:
            raise ValueError(
                "screening_max_rows must be at least 100 and screening_max_assets at least 3"
            )
        if self.target_runtime_minutes < 1:
            raise ValueError("target_runtime_minutes must be positive")
        if self.maximum_compiled_atoms < len(self.input_fields):
            raise ValueError(
                "maximum_compiled_atoms cannot be smaller than the source field count"
            )
        if self.mechanism_candidates_per_card < 1:
            raise ValueError("mechanism_candidates_per_card must be positive")
        if self.mechanism_numeric_quota_per_card < 1:
            raise ValueError("mechanism_numeric_quota_per_card must be positive")
        if self.residual_model_version not in {"legacy_raw_v1", "rank_controls_v2"}:
            raise ValueError(
                "residual_model_version must be legacy_raw_v1 or rank_controls_v2"
            )
        if self.evidence_provenance not in {
            "program_recomputation",
            "synthetic_example",
        }:
            raise ValueError(
                "evidence_provenance must be program_recomputation or synthetic_example"
            )
        valid_atom_families = {
            "price_path",
            "volume_flow",
            "amount_liquidity",
            "volatility_tail",
            "cross_section_centering",
            "residual_relative_structure",
            "fundamental_event",
            "mathematical_shape",
        }
        unknown_atom_families = sorted(set(self.atom_families) - valid_atom_families)
        if unknown_atom_families:
            raise ValueError(f"unknown atom_families: {unknown_atom_families}")
        if len(set(self.plugin_modules)) != len(self.plugin_modules):
            raise ValueError("plugin_modules must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            field.name: (
                list(value)
                if isinstance(value := getattr(self, field.name), tuple)
                else value
            )
            for field in fields(self)
        }
