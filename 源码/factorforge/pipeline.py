from __future__ import annotations

import json
import hashlib
import itertools
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from factorforge.artifacts import (
    ArtifactWriter,
    input_fingerprint,
    software_environment,
)
from factorforge.atoms import AtomCompiler, CompiledAtomLibrary
from factorforge.config import ResearchConfig
from factorforge.contracts import AtomSpec, DecisionRecord, PanelData
from factorforge.evaluation import (
    audit_values,
    benjamini_hochberg_total,
    evaluate_candidate,
    evaluate_frozen_oos,
    screen_candidate,
)
from factorforge.evaluation.negative_controls import selected_factor_negative_controls
from factorforge.formula import (
    CandidateGenerator,
    CudaFormulaEvaluator,
    FormulaEvaluator,
    verify_cuda_consistency,
)
from factorforge.governance import audit_mechanism_cards
from factorforge.ledger import ResearchLedger
from factorforge.monitoring import assess_decay
from factorforge.mechanisms import generate_card_candidates, load_mechanism_cards
from factorforge.plugins import load_plugin_records
from factorforge.pool_version import build_pool_version, read_parent_version
from factorforge.reporting import (
    build_factor_analysis,
    create_factor_charts,
    create_family_coverage_report,
    create_selected_factor_reports,
)
from factorforge.runtime import (
    DeviceGuard,
    RunJournal,
    TaskMonitor,
    preflight_plan,
    resolve_runtime_limits,
)
from factorforge.selection import IncrementalPoolSelector, PoolPolicy


class FactorDiscoveryPipeline:
    """Complete label-isolated candidate generation and incremental selection pipeline."""

    def __init__(self, config: ResearchConfig):
        config.validate()
        self.config = config
        self.output = Path(config.output_root).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self._compiled_atom_library: CompiledAtomLibrary | None = None
        self._card_audits = ()

    def _atom_definitions(self) -> dict[str, dict[str, object]]:
        if not self.config.atom_definitions_path:
            return {}
        return json.loads(
            Path(self.config.atom_definitions_path).read_text(encoding="utf-8")
        )

    def _compile_atom_library(self, panel: PanelData) -> CompiledAtomLibrary:
        if self._compiled_atom_library is not None:
            return self._compiled_atom_library
        definitions = self._atom_definitions()
        missing = sorted(set(panel.fields) - set(definitions)) if definitions else []
        if missing:
            raise ValueError(f"atom definition file is missing input fields: {missing}")
        if self.config.atom_compiler_enabled:
            library = AtomCompiler(
                definitions,
                self.config.rolling_windows,
                self.config.atom_families,
                self.config.maximum_compiled_atoms,
                self.config.residual_model_version,
            ).compile(panel)
        else:
            specs = self._legacy_atom_specs(panel, definitions)
            values = self._causal_fields(panel, specs)
            family_counts = {"source": len(specs)}
            library = CompiledAtomLibrary(specs, values, family_counts)
        self._compiled_atom_library = library
        return library

    def _legacy_atom_specs(
        self, panel: PanelData, definitions: dict[str, dict[str, object]]
    ) -> tuple[AtomSpec, ...]:
        return tuple(
            AtomSpec(
                atom_id=field,
                field=field,
                unit=str(definitions.get(field, {}).get("unit", "input_native")),
                description=str(
                    definitions.get(field, {}).get(
                        "description", f"Direct decision-time input field: {field}"
                    )
                ),
                available_lag=int(definitions.get(field, {}).get("available_lag", 0)),
            )
            for field in sorted(panel.fields)
        )

    def _atom_specs(self, panel: PanelData) -> tuple[AtomSpec, ...]:
        return self._compile_atom_library(panel).specs

    def build_label_blind_catalog(
        self, panel: PanelData, ledger: ResearchLedger
    ) -> int:
        """Generate and register formulas. This method has no label argument."""

        atom_specs = self._atom_specs(panel)
        generator = CandidateGenerator(
            atom_specs, self.config.rolling_windows, self.config.random_seed
        )
        count = ledger.candidate_count()
        if count > self.config.candidate_limit:
            raise ValueError(
                "existing catalog exceeds candidate_limit; use a new output directory"
            )
        if count == self.config.candidate_limit:
            return count
        pending = []
        plugin_records = load_plugin_records(
            self.config.plugin_modules,
            atom_specs,
            self.config.rolling_windows,
            self.config.random_seed,
        )
        cards = load_mechanism_cards(self.config.mechanism_cards_path)
        self._card_audits = audit_mechanism_cards(cards, atom_specs)
        blocking_cards = [
            audit.card_id
            for audit in self._card_audits
            if not (
                audit.structurally_complete
                and audit.falsifiable
                and audit.label_blind
            )
        ]
        if blocking_cards:
            raise ValueError(f"research card audit failed: {blocking_cards[:20]}")
        mechanism_records = generate_card_candidates(
            cards,
            atom_specs,
            self.config.random_seed,
            self.config.mechanism_candidates_per_card,
        )
        source = itertools.chain(
            mechanism_records,
            plugin_records,
            generator.stream(
                self.config.candidate_limit * 2,
                generation_batch="initial_empty_pool",
            ),
        )
        for record in source:
            pending.append(record)
            remaining = self.config.candidate_limit - count
            if len(pending) >= min(10_000, remaining):
                count += ledger.add_candidates(pending[:remaining])
                pending.clear()
                if count >= self.config.candidate_limit:
                    break
        if pending and count < self.config.candidate_limit:
            count += ledger.add_candidates(
                pending[: self.config.candidate_limit - count]
            )
        if count != self.config.candidate_limit:
            raise RuntimeError(
                f"candidate generator stopped at {count}; expected {self.config.candidate_limit}"
            )
        return count

    def _causal_fields(
        self, panel: PanelData, specs: tuple[AtomSpec, ...]
    ) -> dict[str, np.ndarray]:
        fields = {}
        for spec in specs:
            source = np.asarray(panel.fields[spec.field], dtype=float)
            if spec.available_lag < 0:
                raise ValueError(f"atom {spec.atom_id} has a negative available_lag")
            if spec.available_lag == 0:
                fields[spec.atom_id] = source
                continue
            shifted = np.full_like(source, np.nan)
            shifted[spec.available_lag :] = source[: -spec.available_lag]
            fields[spec.atom_id] = shifted
        return fields

    def _run_identity(
        self, panel: PanelData, labels: np.ndarray
    ) -> tuple[dict[str, object], dict[str, object], str]:
        input_identity = input_fingerprint(
            panel.timestamps, panel.assets, panel.fields, labels
        )
        contract_config = self.config.to_dict()
        contract_config["resume"] = False
        contract = {
            "configuration": contract_config,
            "complete_input_sha256": input_identity["complete_input_sha256"],
        }
        contract_sha = hashlib.sha256(
            json.dumps(contract, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return input_identity, contract, contract_sha

    def validate_run_identity(
        self, panel: PanelData, future_return: np.ndarray
    ) -> bool:
        """Fail before writing when an existing run cannot be resumed safely."""

        panel.validate()
        labels = np.asarray(future_return, dtype=float)
        if labels.shape != (len(panel.timestamps), len(panel.assets)):
            raise ValueError("future_return must match the panel geometry")
        for field_name, field_values in panel.fields.items():
            candidate = np.asarray(field_values, dtype=float)
            if candidate.shape == labels.shape and np.array_equal(
                candidate, labels, equal_nan=True
            ):
                raise ValueError(
                    f"input field {field_name!r} is identical to the evaluation label"
                )
        ledger_path = self.output / "research_ledger.sqlite"
        resumed = ledger_path.exists() and self.config.resume
        if ledger_path.exists() and not self.config.resume:
            raise FileExistsError(
                f"output already contains a research ledger: {ledger_path}. "
                "Choose a new output directory; existing evidence is never overwritten."
            )
        _, _, contract_sha = self._run_identity(panel, labels)
        contract_path = self.output / "RUN_CONTRACT.json"
        if resumed and not contract_path.is_file():
            raise ValueError("resume requires an existing RUN_CONTRACT.json")
        if contract_path.exists():
            previous = json.loads(contract_path.read_text(encoding="utf-8"))
            if previous.get("contract_sha256") != contract_sha:
                raise ValueError(
                    "resume contract mismatch: configuration or input panel changed"
                )
        return resumed

    def _validated_labels(
        self, panel: PanelData, future_return: np.ndarray
    ) -> np.ndarray:
        """Enforce the declared horizon and optionally verify labels from prices."""

        labels = np.asarray(future_return, dtype=float)
        if labels.shape != (len(panel.timestamps), len(panel.assets)):
            raise ValueError("future_return must match the panel geometry")
        if self.config.label_validation_mode != "recompute_and_verify":
            return labels
        price = np.asarray(panel.fields[self.config.label_price_field], dtype=float)
        offset = self.config.label_start_offset_periods
        horizon = self.config.label_horizon_periods
        expected = np.full_like(price, np.nan)
        stop = len(price) - offset - horizon
        if stop <= 0:
            raise ValueError("panel is too short to reconstruct the declared label")
        entry = price[offset : offset + stop]
        exit_price = price[offset + horizon : offset + horizon + stop]
        if self.config.label_return_type == "simple_return":
            reconstructed = np.divide(
                exit_price,
                entry,
                out=np.full_like(entry, np.nan),
                where=np.isfinite(entry) & np.isfinite(exit_price) & (entry > 0.0),
            ) - 1.0
        else:
            reconstructed = np.log(
                np.divide(
                    exit_price,
                    entry,
                    out=np.full_like(entry, np.nan),
                    where=np.isfinite(entry)
                    & np.isfinite(exit_price)
                    & (entry > 0.0)
                    & (exit_price > 0.0),
                )
            )
        expected[:stop] = reconstructed
        active = np.isfinite(expected) & np.isfinite(labels)
        if not np.any(active):
            raise ValueError("no finite labels can be verified against the price field")
        maximum_error = float(np.max(np.abs(expected[active] - labels[active])))
        if maximum_error > 1e-10:
            raise ValueError(
                "precomputed labels do not match the declared price clock; "
                f"maximum absolute error={maximum_error:.3g}"
            )
        return expected

    def run(self, panel: PanelData, future_return: np.ndarray) -> dict[str, object]:
        panel.validate()
        labels = self._validated_labels(panel, future_return)

        ledger_path = self.output / "research_ledger.sqlite"
        resumed = self.validate_run_identity(panel, labels)
        artifacts = ArtifactWriter(self.output)
        input_identity, contract, contract_sha = self._run_identity(panel, labels)
        fingerprint_path = self.output / "INPUT_FINGERPRINT.json"
        environment_path = self.output / "SOFTWARE_ENVIRONMENT.json"
        if not fingerprint_path.exists():
            artifacts.write_json("INPUT_FINGERPRINT.json", input_identity)
        if not environment_path.exists():
            artifacts.write_json("SOFTWARE_ENVIRONMENT.json", software_environment())
        contract_path = self.output / "RUN_CONTRACT.json"
        if not contract_path.exists():
            artifacts.write_json(
                "RUN_CONTRACT.json", {**contract, "contract_sha256": contract_sha}
            )
        artifacts.write_json(
            "LABEL_CONTRACT.json",
            {
                "horizon_periods": self.config.label_horizon_periods,
                "return_type": self.config.label_return_type,
                "start_offset_periods": self.config.label_start_offset_periods,
                "cost_status": self.config.label_cost_status,
                "validation_mode": self.config.label_validation_mode,
                "price_field": self.config.label_price_field,
                "portfolio_accounting": "daily_staggered_fixed_horizon_cohorts",
            },
        )
        fingerprints: set[str] = set()
        limits = resolve_runtime_limits(self.config)
        plan = preflight_plan(
            labels.shape,
            limits,
            self.config.evaluation_shortlist_size,
            self.config.target_pool_size,
            self.config.formula_cache_entries,
        )
        device_guard = DeviceGuard(limits, self.config.gpu_backend in {"auto", "cuda"})
        monitor = TaskMonitor(
            self.output,
            min(self.config.numeric_screening_budget, self.config.candidate_limit),
        )

        with RunJournal(self.output) as journal, ResearchLedger(ledger_path) as ledger:
            catalog_count = self.build_label_blind_catalog(panel, ledger)
            artifacts.write_json(
                "RESEARCH_CARD_AUDIT.json",
                {"cards": [asdict(audit) for audit in self._card_audits]},
            )
            atom_library = self._compile_atom_library(panel)
            atom_specs = atom_library.specs
            causal_fields = dict(atom_library.values)
            artifacts.write_json(
                "ATOM_LIBRARY_SUMMARY.json",
                {
                    "compiled_atom_count": len(atom_specs),
                    "family_counts": dict(atom_library.family_counts),
                    "compiler_enabled": self.config.atom_compiler_enabled,
                },
            )
            artifacts.write_json(
                "ATOM_LIBRARY_REGISTRY.json",
                {
                    "atoms": [asdict(spec) for spec in atom_specs],
                    "family_counts": dict(atom_library.family_counts),
                    "note": "Exact atom registry used by this run; later compiler changes must not replace it during replay.",
                },
            )
            evaluator = FormulaEvaluator(
                causal_fields, cache_entries=plan.formula_cache_entries
            )
            development_positions = np.flatnonzero(
                panel.timestamps <= pd.Timestamp(self.config.development_end, tz="UTC")
            )
            screening_rows = development_positions[-self.config.screening_max_rows :]
            asset_count = min(len(panel.assets), self.config.screening_max_assets)
            screening_assets = np.linspace(
                0, len(panel.assets) - 1, asset_count, dtype=int
            )
            screening_fields = {
                name: values[np.ix_(screening_rows, screening_assets)]
                for name, values in causal_fields.items()
            }
            screening_labels = labels[np.ix_(screening_rows, screening_assets)]
            screening_timestamps = panel.timestamps[screening_rows]
            screening_evaluator = FormulaEvaluator(
                screening_fields, cache_entries=plan.formula_cache_entries
            )
            timing_sample = ledger.sample_candidates(min(16, catalog_count))
            timing_started = time.perf_counter()
            for record in timing_sample:
                sample_values = screening_evaluator.evaluate_json(record.formula_json)
                audit_values(sample_values, self.config.minimum_coverage)
            seconds_per_candidate = (
                (time.perf_counter() - timing_started) / len(timing_sample)
                if timing_sample
                else 0.0
            )
            target_seconds = self.config.target_runtime_minutes * 60.0
            timed_capacity = (
                int(target_seconds * 0.35 / seconds_per_candidate)
                if seconds_per_candidate > 0.0
                else self.config.numeric_screening_budget
            )
            numeric_budget = min(
                catalog_count,
                self.config.numeric_screening_budget,
                max(self.config.evaluation_shortlist_size, timed_capacity),
            )
            monitor.total_items = numeric_budget
            ledger.record_structural_prefilter(
                numeric_budget, self.config.mechanism_numeric_quota_per_card
            )
            artifacts.write_json(
                "SCREENING_BUDGET_PLAN.json",
                {
                    "timing_sample_candidates": len(timing_sample),
                    "seconds_per_candidate": seconds_per_candidate,
                    "target_runtime_minutes": self.config.target_runtime_minutes,
                    "configured_numeric_budget": self.config.numeric_screening_budget,
                    "effective_numeric_budget": numeric_budget,
                    "safety_fraction": 0.35,
                    "budget_capacity_measurement_reads_labels": False,
                    "candidate_screening_reads_labels": True,
                },
            )
            artifacts.write_json("PREFLIGHT_PLAN.json", asdict(plan))
            backend_report = {
                "requested": self.config.gpu_backend,
                "active": "cpu",
                "consistency": None,
            }
            if self.config.gpu_backend in {"auto", "cuda"}:
                sample_batch = next(
                    ledger.iter_candidates(min(12, limits.batch_size)), []
                )
                consistency = verify_cuda_consistency(
                    causal_fields,
                    [item.formula_json for item in sample_batch],
                    limits.maximum_gpu_memory_fraction,
                )
                backend_report["consistency"] = consistency
                if consistency["available"] and consistency["passed"]:
                    evaluator = CudaFormulaEvaluator(
                        causal_fields,
                        limits.maximum_gpu_memory_fraction,
                        plan.formula_cache_entries,
                    )
                    backend_report["active"] = "cuda"
                elif self.config.gpu_backend == "cuda":
                    raise RuntimeError(f"CUDA consistency gate failed: {consistency}")
                else:
                    backend_report["fallback_reason"] = consistency["reason"]
            artifacts.write_json("EXECUTION_BACKEND.json", backend_report)

            fingerprints = ledger.accepted_metric_values(
                "LABEL_BLIND", "$.rank_path_sha256"
            )
            screened_count = ledger.stage_count("SCREENING")
            processed_count = ledger.stage_count("LABEL_BLIND")
            for batch_number, batch in enumerate(
                ledger.iter_candidates_pending_screening(limits.batch_size), start=1
            ):
                device_guard.check()
                values_batch = screening_evaluator.evaluate_many(
                    [record.formula_json for record in batch]
                )
                for record, values in zip(batch, values_batch, strict=True):
                    blind = audit_values(values, self.config.minimum_coverage)
                    if not blind.accepted:
                        ledger.add_decision(
                            DecisionRecord(
                                record.candidate_id,
                                "LABEL_BLIND",
                                False,
                                blind.reason_code,
                                "Candidate failed pre-label numeric checks.",
                                asdict(blind),
                            )
                        )
                        continue
                    if blind.rank_path_sha256 in fingerprints:
                        ledger.add_decision(
                            DecisionRecord(
                                record.candidate_id,
                                "LABEL_BLIND",
                                False,
                                "NUMERIC_INFORMATION_DUPLICATE",
                                "An earlier formula has the same orientation-free rank path.",
                                asdict(blind),
                            )
                        )
                        continue
                    fingerprints.add(blind.rank_path_sha256)
                    ledger.add_decision(
                        DecisionRecord(
                            record.candidate_id,
                            "LABEL_BLIND",
                            True,
                            "LABEL_BLIND_PASS",
                            "Formula and numeric identity are eligible for stage-one screening.",
                            asdict(blind),
                        )
                    )
                    try:
                        screening = screen_candidate(
                            record.candidate_id,
                            values,
                            screening_labels,
                            screening_timestamps,
                            self.config.development_end,
                            self.config.primary_holding_period,
                            self.config.screening_stride,
                            self.config.one_way_cost_bps,
                        )
                    except ValueError as error:
                        ledger.add_decision(
                            DecisionRecord(
                                record.candidate_id,
                                "SCREENING",
                                False,
                                "INVALID_SCREENING_EVIDENCE",
                                str(error),
                                {},
                            )
                        )
                        continue
                    passed = (
                        screening.metrics["positive_block_fraction"]
                        >= self.config.screening_minimum_positive_block_fraction
                    )
                    ledger.add_decision(
                        DecisionRecord(
                            record.candidate_id,
                            "SCREENING",
                            passed,
                            (
                                "SCREENING_PASS"
                                if passed
                                else "SCREENING_TIME_STABILITY_REJECT"
                            ),
                            "Stage-one uses a fixed development subsample and never reads the outer interval.",
                            screening.metrics,
                        )
                    )
                    screened_count += 1
                processed_count += len(batch)
                if batch_number % self.config.checkpoint_every_batches == 0:
                    ledger.commit()
                    telemetry = monitor.observe(processed_count)
                    journal.update(
                        "SCREENING",
                        batches_completed=batch_number,
                        candidates_screened=screened_count,
                        candidates_processed=processed_count,
                        candidates_cataloged=catalog_count,
                        telemetry=telemetry,
                    )

            monitor.observe(processed_count)

            screening_frame = ledger.screening_statistics()
            screening_frame.to_parquet(
                self.output / "完整预筛统计.parquet", index=False
            )
            shortlist = ledger.screening_shortlist(plan.shortlist_capacity)
            ledger.record_shortlist(shortlist, plan.shortlist_capacity)
            completed_development = ledger.stage_results(
                "DEVELOPMENT_CROSSFIT", accepted=True
            )
            evidences = []
            exact_fingerprints: set[str] = set()
            shortlist_records = [ledger.get_candidate(value) for value in shortlist]
            exact_values_batch = evaluator.evaluate_many(
                [record.formula_json for record in shortlist_records]
            )
            for index, (candidate_id, record, exact_values) in enumerate(
                zip(shortlist, shortlist_records, exact_values_batch, strict=True),
                start=1,
            ):
                device_guard.check()
                monitor.check_disk()
                exact_blind = audit_values(exact_values, self.config.minimum_coverage)
                if not exact_blind.accepted:
                    if not ledger.has_decision(candidate_id, "DEVELOPMENT_CROSSFIT"):
                        ledger.add_decision(
                            DecisionRecord(
                                candidate_id,
                                "DEVELOPMENT_CROSSFIT",
                                False,
                                "FULL_PANEL_LABEL_BLIND_REJECT",
                                "Full-panel numeric audit rejected the shortlist candidate.",
                                asdict(exact_blind),
                            )
                        )
                    continue
                if exact_blind.rank_path_sha256 in exact_fingerprints:
                    if not ledger.has_decision(candidate_id, "DEVELOPMENT_CROSSFIT"):
                        ledger.add_decision(
                            DecisionRecord(
                                candidate_id,
                                "DEVELOPMENT_CROSSFIT",
                                False,
                                "FULL_PANEL_NUMERIC_DUPLICATE",
                                "Full-panel rank path duplicates an earlier exact candidate.",
                                asdict(exact_blind),
                            )
                        )
                    continue
                exact_fingerprints.add(exact_blind.rank_path_sha256)
                try:
                    evidence = evaluate_candidate(
                        candidate_id,
                        exact_values,
                        labels,
                        panel.timestamps,
                        self.config.development_end,
                        self.config.periods_per_year,
                        self.config.one_way_cost_bps,
                        self.config.quantile_count,
                        self.config.primary_holding_period,
                    )
                except ValueError as error:
                    if not ledger.has_decision(candidate_id, "DEVELOPMENT_CROSSFIT"):
                        ledger.add_decision(
                            DecisionRecord(
                                candidate_id,
                                "DEVELOPMENT_CROSSFIT",
                                False,
                                "INVALID_DEVELOPMENT_EVIDENCE",
                                str(error),
                                {},
                            )
                        )
                    continue
                evidences.append(evidence)
                if candidate_id not in completed_development:
                    ledger.add_decision(
                        DecisionRecord(
                            candidate_id,
                            "DEVELOPMENT_CROSSFIT",
                            True,
                            "DEVELOPMENT_EVALUATED",
                            "Exact chronological cross-fit after the fixed stage-one shortlist.",
                            {**evidence.metrics, "p_value": evidence.p_value},
                        )
                    )
                if self.config.persist_values == "all_evaluated":
                    artifacts.write_values(
                        record, evidence, panel.timestamps, panel.assets
                    )
                if index % max(1, self.config.checkpoint_every_batches) == 0:
                    ledger.commit()
                    journal.update(
                        "EXACT_EVALUATION",
                        exact_completed=index,
                        exact_total=len(shortlist),
                    )

            p_value_ids = [item.candidate_id for item in evidences]
            device_guard.check()
            monitor.check_disk()
            p_values = [item.p_value for item in evidences]
            q_values_array = benjamini_hochberg_total(
                np.asarray(p_values, dtype=float), catalog_count
            )
            q_values = dict(zip(p_value_ids, q_values_array, strict=True))
            p_by_id = dict(zip(p_value_ids, p_values, strict=True))
            evidence_by_id = {item.candidate_id: item for item in evidences}
            for candidate_id, q_value in q_values.items():
                if ledger.has_decision(candidate_id, "MULTIPLE_TESTING"):
                    continue
                ledger.add_decision(
                    DecisionRecord(
                        candidate_id,
                        "MULTIPLE_TESTING",
                        q_value <= self.config.fdr_alpha,
                        (
                            "BH_FDR_PASS"
                            if q_value <= self.config.fdr_alpha
                            else "BH_FDR_NOT_PASSED"
                        ),
                        "Exact-stage BH conservatively counts the complete registered catalog as hypotheses.",
                        {
                            "p_value": float(p_by_id[candidate_id]),
                            "bh_q_value": float(q_value),
                            "alpha": self.config.fdr_alpha,
                            "total_hypotheses": catalog_count,
                        },
                    )
                )

            negative_control_rows = []
            control_pass: dict[str, bool] = {}
            for evidence in evidences:
                rows = selected_factor_negative_controls(
                    evidence,
                    labels,
                    panel.timestamps,
                    self.config.development_end,
                    self.config.periods_per_year,
                    self.config.one_way_cost_bps,
                    self.config.quantile_count,
                    self.config.primary_holding_period,
                )
                negative_control_rows.extend(rows)
                hard_controls = [
                    row
                    for row in rows
                    if row["control"]
                    in {"TRAINING_LABEL_BLOCK_PERMUTATION", "ASSET_EVENT_MISMATCH"}
                ]
                passed = bool(hard_controls) and all(
                    row["status"] == "COMPUTED"
                    and abs(float(row["rank_ic_mean"]))
                    <= self.config.negative_control_max_abs_rank_ic
                    and float(row["hac_p_value"])
                    >= self.config.negative_control_min_hac_p_value
                    for row in hard_controls
                )
                control_pass[evidence.candidate_id] = passed
                ledger.add_decision(
                    DecisionRecord(
                        evidence.candidate_id,
                        "NEGATIVE_CONTROL_GATE",
                        passed or not self.config.negative_control_gate_enabled,
                        (
                            "NEGATIVE_CONTROLS_PASS"
                            if passed
                            else "NEGATIVE_CONTROLS_NOT_PASSED"
                        ),
                        "Pre-registered controls are evaluated before pool construction.",
                        {
                            "gate_enabled": self.config.negative_control_gate_enabled,
                            "maximum_abs_rank_ic": self.config.negative_control_max_abs_rank_ic,
                            "minimum_hac_p_value": self.config.negative_control_min_hac_p_value,
                        },
                    )
                )
            pd.DataFrame(negative_control_rows).to_parquet(
                self.output / "负对照结果.parquet", index=False
            )
            eligible_evidences = [
                evidence
                for evidence in evidences
                if control_pass.get(evidence.candidate_id, False)
                or not self.config.negative_control_gate_enabled
            ]

            selector = IncrementalPoolSelector(
                PoolPolicy(
                    target_size=self.config.target_pool_size,
                    periods_per_year=self.config.periods_per_year,
                    minimum_positive_fold_fraction=self.config.minimum_positive_fold_fraction,
                    minimum_rank_ic=self.config.minimum_oof_rank_ic,
                    maximum_return_correlation=self.config.maximum_abs_rank_correlation,
                    fdr_alpha=self.config.fdr_alpha,
                    fdr_gate_enabled=self.config.fdr_gate_enabled,
                    confirmation_fraction=self.config.pool_confirmation_fraction,
                    maximum_value_rank_correlation=self.config.maximum_value_rank_correlation,
                    maximum_reconstruction_r2=self.config.maximum_reconstruction_r2,
                    development_tail_selection_enabled=self.config.development_tail_selection_enabled,
                )
            )
            selected, pool_decisions = selector.select(eligible_evidences, q_values)
            ablation_rows = []
            for tail_selection in (True, False):
                for negative_gate in (False, True):
                    ablation_input = [
                        evidence
                        for evidence in evidences
                        if not negative_gate
                        or control_pass.get(evidence.candidate_id, False)
                    ]
                    ablation_selector = IncrementalPoolSelector(
                        PoolPolicy(
                            target_size=self.config.target_pool_size,
                            periods_per_year=self.config.periods_per_year,
                            minimum_positive_fold_fraction=self.config.minimum_positive_fold_fraction,
                            minimum_rank_ic=self.config.minimum_oof_rank_ic,
                            maximum_return_correlation=self.config.maximum_abs_rank_correlation,
                            fdr_alpha=self.config.fdr_alpha,
                            fdr_gate_enabled=self.config.fdr_gate_enabled,
                            confirmation_fraction=self.config.pool_confirmation_fraction,
                            maximum_value_rank_correlation=self.config.maximum_value_rank_correlation,
                            maximum_reconstruction_r2=self.config.maximum_reconstruction_r2,
                            development_tail_selection_enabled=tail_selection,
                        )
                    )
                    ablation_selected, _ = ablation_selector.select(
                        ablation_input, q_values
                    )
                    ablation_rows.append(
                        {
                            "development_tail_selection_enabled": tail_selection,
                            "negative_control_gate_enabled": negative_gate,
                            "input_count": len(ablation_input),
                            "selected_count": len(ablation_selected),
                            "selected_candidate_ids": [
                                item.candidate_id for item in ablation_selected
                            ],
                            "mean_rank_ic": (
                                float(
                                    np.mean(
                                        [
                                            item.metrics["rank_ic_mean"]
                                            for item in ablation_selected
                                        ]
                                    )
                                )
                                if ablation_selected
                                else None
                            ),
                            "mean_positive_fold_fraction": (
                                float(
                                    np.mean(
                                        [
                                            item.metrics["positive_fold_fraction"]
                                            for item in ablation_selected
                                        ]
                                    )
                                )
                                if ablation_selected
                                else None
                            ),
                            "mean_annual_return": (
                                float(
                                    np.mean(
                                        [
                                            item.metrics["annual_return"]
                                            for item in ablation_selected
                                        ]
                                    )
                                )
                                if ablation_selected
                                else None
                            ),
                            "mean_sharpe": (
                                float(
                                    np.mean(
                                        [
                                            item.metrics["sharpe"]
                                            for item in ablation_selected
                                        ]
                                    )
                                )
                                if ablation_selected
                                else None
                            ),
                            "mean_max_drawdown": (
                                float(
                                    np.mean(
                                        [
                                            item.metrics["max_drawdown"]
                                            for item in ablation_selected
                                        ]
                                    )
                                )
                                if ablation_selected
                                else None
                            ),
                            "mean_calmar": (
                                float(
                                    np.mean(
                                        [
                                            item.metrics["calmar"]
                                            for item in ablation_selected
                                        ]
                                    )
                                )
                                if ablation_selected
                                else None
                            ),
                        }
                    )
            artifacts.write_json("SELECTION_ABLATION.json", ablation_rows)
            device_guard.check()
            monitor.check_disk()
            for decision in pool_decisions:
                if ledger.has_decision(decision.candidate_id, "INCREMENTAL_POOL"):
                    continue
                ledger.add_decision(
                    DecisionRecord(
                        decision.candidate_id,
                        "INCREMENTAL_POOL",
                        decision.accepted,
                        decision.reason_code,
                        "Candidate was judged against the already frozen pool, not by standalone beauty.",
                        {"step": decision.step, **decision.increments},
                    )
                )

            frozen_ids = [item.candidate_id for item in selected]
            freeze = {
                "project_name": self.config.project_name,
                "development_end": self.config.development_end,
                "oos_start": self.config.oos_start,
                "oos_end": self.config.oos_end,
                "selected_candidate_ids": frozen_ids,
                "directions": {
                    item.candidate_id: item.final_direction for item in selected
                },
                "selection_uses_oos": False,
                "configuration": self.config.to_dict(),
                "runtime_limits": asdict(limits),
                "preflight_plan": asdict(plan),
            }
            artifacts.write_json("FROZEN_POOL.json", freeze)
            pool_version = build_pool_version(
                frozen_ids,
                freeze["directions"],
                self.config.to_dict(),
                input_identity["complete_input_sha256"],
                read_parent_version(self.config.parent_pool_path),
            )
            artifacts.write_json("FACTOR_POOL_VERSION.json", pool_version)

            oos_rows = []
            if self.config.automatic_oos_evaluation:
                for evidence in selected:
                    device_guard.check()
                    monitor.check_disk()
                    path, metrics = evaluate_frozen_oos(
                        evidence,
                        labels,
                        panel.timestamps,
                        self.config.oos_start,
                        self.config.oos_end,
                        self.config.periods_per_year,
                        self.config.one_way_cost_bps,
                        self.config.primary_holding_period,
                    )
                    decay = assess_decay(
                        path, recent_length=max(5, min(63, len(path) // 3))
                    )
                    oos_rows.append(
                        {
                            "candidate_id": evidence.candidate_id,
                            **metrics,
                            **asdict(decay),
                        }
                    )
                    if not ledger.has_decision(evidence.candidate_id, "FROZEN_OOS"):
                        ledger.add_decision(
                            DecisionRecord(
                                evidence.candidate_id,
                                "FROZEN_OOS",
                                True,
                                "OOS_REPORTED_NOT_SELECTED",
                                "Frozen formula and direction were evaluated once without reselection.",
                                metrics,
                            )
                        )
            else:
                artifacts.write_json(
                    "OOS_EVALUATION_STATUS.json",
                    {
                        "status": "NOT_RUN_IN_DISCOVERY_PROCESS",
                        "reason": "automatic_oos_evaluation is disabled; use a separately governed evaluation run after freezing",
                    },
                )
            if self.config.persist_values == "selected_only":
                for evidence in selected:
                    artifacts.write_values(
                        ledger.get_candidate(evidence.candidate_id),
                        evidence,
                        panel.timestamps,
                        panel.assets,
                    )

            pd.DataFrame(oos_rows).to_parquet(
                self.output / "冻结池样本外结果.parquet", index=False
            )
            selected_records = {
                item.candidate_id: ledger.get_candidate(item.candidate_id)
                for item in selected
            }
            device_guard.check()
            monitor.check_disk()
            pd.DataFrame(
                [
                    {
                        "candidate_id": item.candidate_id,
                        "direction": item.final_direction,
                        "formula": ledger.get_candidate(item.candidate_id).formula_json,
                        "atoms": json.dumps(
                            ledger.get_candidate(item.candidate_id).atoms,
                            ensure_ascii=False,
                        ),
                        **item.metrics,
                        "p_value": item.p_value,
                        "bh_q_value": q_values[item.candidate_id],
                    }
                    for item in selected
                ]
            ).to_parquet(self.output / "冻结因子池.parquet", index=False)
            build_factor_analysis(selected, q_values, self.output)
            create_factor_charts(
                selected,
                self.output,
                panel.timestamps,
                self.config.periods_per_year,
            )
            create_selected_factor_reports(
                selected,
                q_values,
                selected_records,
                atom_specs,
                panel.timestamps,
                {row["candidate_id"]: row for row in oos_rows},
                self.output,
                self.config.evidence_provenance,
                {
                    "holding_period": self.config.primary_holding_period,
                    "periods_per_year": self.config.periods_per_year,
                    "one_way_cost_bps": self.config.one_way_cost_bps,
                    "label_horizon_periods": self.config.label_horizon_periods,
                    "label_return_type": self.config.label_return_type,
                    "label_start_offset_periods": self.config.label_start_offset_periods,
                    "label_cost_status": self.config.label_cost_status,
                    "label_validation_mode": self.config.label_validation_mode,
                },
            )
            coverage = ledger.family_coverage_statistics()
            coverage.to_parquet(self.output / "信息家族覆盖统计.parquet", index=False)
            create_family_coverage_report(coverage, self.output)
            backend_report["formula_cache"] = evaluator.cache_report()
            backend_report["screening_formula_cache"] = (
                screening_evaluator.cache_report()
            )
            backend_report["screening_geometry"] = {
                "rows": len(screening_rows),
                "assets": len(screening_assets),
                "numeric_budget": numeric_budget,
                "target_runtime_minutes": self.config.target_runtime_minutes,
            }
            artifacts.write_json("EXECUTION_BACKEND.json", backend_report)
            ledger.export(self.output)

            final_counts = {
                "structural_prefilter_count": ledger.stage_count(
                    "STRUCTURAL_PREFILTER", accepted=True
                ),
                "label_blind_unique_count": ledger.stage_count(
                    "LABEL_BLIND", accepted=True
                ),
                "screening_evaluated_count": ledger.stage_count("SCREENING"),
                "screening_pass_count": ledger.stage_count("SCREENING", accepted=True),
            }

        summary = {
            "catalog_candidates": catalog_count,
            **final_counts,
            "exact_evaluated_count": len(evidences),
            "bounded_shortlist_count": len(shortlist),
            "selected_count": len(selected),
            "oos_used_for_selection": False,
            "output_root": self.config.output_root,
            "runtime_limits": asdict(limits),
            "preflight_plan": asdict(plan),
            "resumed_from_ledger": resumed,
        }
        artifacts.write_json("RUN_SUMMARY.json", summary)
        artifacts.write_manifest()
        return summary
