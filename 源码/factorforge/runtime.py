from __future__ import annotations

import csv
import io
import json
import math
import os
import platform
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass

import psutil


@dataclass(frozen=True)
class MemorySnapshot:
    total_gb: float
    available_gb: float
    used_fraction: float


@dataclass(frozen=True)
class GPUStatus:
    name: str
    total_memory_mb: int
    used_memory_mb: int
    temperature_c: int
    utilization_percent: int

    @property
    def used_fraction(self) -> float:
        return (
            self.used_memory_mb / self.total_memory_mb if self.total_memory_mb else 0.0
        )


@dataclass(frozen=True)
class RuntimeLimits:
    profile: str
    batch_size: int
    workers: int
    maximum_memory_fraction: float
    maximum_gpu_memory_fraction: float
    maximum_gpu_temperature_c: int


@dataclass(frozen=True)
class PreflightPlan:
    panel_cells: int
    bytes_per_matrix: int
    formula_cache_entries: int
    shortlist_capacity: int
    estimated_peak_gb: float


PROFILE_LIMITS = {
    "safe": (64, 1, 0.60, 0.45, 76),
    "balanced": (256, 1, 0.72, 0.65, 79),
    "performance": (768, 1, 0.82, 0.78, 82),
}


def detect_gpu() -> GPUStatus | None:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,temperature.gpu,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=5, check=True
        )
        row = next(csv.reader(io.StringIO(completed.stdout.strip())))
        return GPUStatus(
            row[0].strip(), int(row[1]), int(row[2]), int(row[3]), int(row[4])
        )
    except (OSError, subprocess.SubprocessError, StopIteration, ValueError):
        return None


def resolve_runtime_limits(config) -> RuntimeLimits:
    if config.runtime_profile == "custom":
        return RuntimeLimits(
            "custom",
            config.batch_size,
            config.workers,
            config.max_memory_fraction,
            config.gpu_memory_fraction,
            config.gpu_temperature_limit_c,
        )
    batch, workers, memory, gpu_memory, temperature = PROFILE_LIMITS[
        config.runtime_profile
    ]
    return RuntimeLimits(
        config.runtime_profile,
        min(config.batch_size, batch),
        min(config.workers, workers, max(1, (os.cpu_count() or 2) // 2)),
        min(config.max_memory_fraction, memory),
        min(config.gpu_memory_fraction, gpu_memory),
        min(config.gpu_temperature_limit_c, temperature),
    )


def machine_report() -> dict[str, object]:
    memory = psutil.virtual_memory()
    gpu = detect_gpu()
    return {
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "total_memory_gb": round(memory.total / 1024**3, 2),
        "available_memory_gb": round(memory.available / 1024**3, 2),
        "gpu": asdict(gpu) if gpu else None,
        "profiles": {
            name: {
                "batch_size_cap": values[0],
                "worker_cap": values[1],
                "memory_cap": values[2],
                "gpu_memory_cap": values[3],
                "gpu_temperature_cap_c": values[4],
            }
            for name, values in PROFILE_LIMITS.items()
        },
    }


def preflight_plan(
    shape: tuple[int, int],
    limits: RuntimeLimits,
    requested_shortlist: int,
    target_pool: int,
    requested_cache: int,
) -> PreflightPlan:
    cells = int(shape[0] * shape[1])
    matrix_bytes = cells * 8
    memory = psutil.virtual_memory()
    used = int(memory.total - memory.available)
    usable_headroom = max(0, int(memory.total * limits.maximum_memory_fraction) - used)
    reserve = 768 * 1024**2
    budget = max(0, int((usable_headroom - reserve) * 0.85))
    cache = min(requested_cache, max(0, budget // max(matrix_bytes, 1) // 4))
    remaining = max(0, budget - cache * matrix_bytes)
    bytes_per_evidence = matrix_bytes * 3 + shape[0] * 8
    capacity = min(requested_shortlist, remaining // max(bytes_per_evidence, 1))
    if capacity < target_pool:
        raise MemoryError(
            f"preflight refused run: safe shortlist capacity={capacity}, target_pool_size={target_pool}. "
            "Use a smaller asset/time partition or a machine with more host memory."
        )
    estimate = (
        cache * matrix_bytes + capacity * bytes_per_evidence + reserve
    ) / 1024**3
    return PreflightPlan(
        cells, matrix_bytes, int(cache), int(capacity), round(estimate, 3)
    )


class MemoryGuard:
    """Fail closed before a new batch when system memory exceeds the fixed cap."""

    def __init__(self, maximum_used_fraction: float):
        if not 0.10 < maximum_used_fraction < 0.95:
            raise ValueError("maximum_used_fraction must be between 0.10 and 0.95")
        self.maximum_used_fraction = maximum_used_fraction

    def snapshot(self) -> MemorySnapshot:
        memory = psutil.virtual_memory()
        return MemorySnapshot(
            total_gb=memory.total / 1024**3,
            available_gb=memory.available / 1024**3,
            used_fraction=1.0 - memory.available / memory.total,
        )

    def check(self) -> MemorySnapshot:
        snapshot = self.snapshot()
        if snapshot.used_fraction > self.maximum_used_fraction:
            raise MemoryError(
                f"memory guard stopped before next batch: used={snapshot.used_fraction:.1%}, "
                f"limit={self.maximum_used_fraction:.1%}"
            )
        return snapshot


class DeviceGuard:
    """Fail closed between batches when either host memory or NVIDIA GPU is unsafe."""

    def __init__(self, limits: RuntimeLimits, gpu_enabled: bool):
        self.limits = limits
        self.memory = MemoryGuard(limits.maximum_memory_fraction)
        self.gpu_enabled = gpu_enabled

    def check(self) -> tuple[MemorySnapshot, GPUStatus | None]:
        memory = self.memory.check()
        gpu = detect_gpu() if self.gpu_enabled else None
        if gpu and gpu.used_fraction > self.limits.maximum_gpu_memory_fraction:
            raise MemoryError(
                f"GPU guard stopped before next batch: used={gpu.used_fraction:.1%}, "
                f"limit={self.limits.maximum_gpu_memory_fraction:.1%}"
            )
        if gpu and gpu.temperature_c > self.limits.maximum_gpu_temperature_c:
            raise RuntimeError(
                f"GPU temperature guard stopped before next batch: temperature={gpu.temperature_c}C, "
                f"limit={self.limits.maximum_gpu_temperature_c}C"
            )
        return memory, gpu


class RunJournal:
    """Atomic run state and a plain-text alert suitable for unattended offline jobs."""

    def __init__(self, output):
        from pathlib import Path

        self.root = Path(output)
        self.path = self.root / "RUN_STATE.json"
        self.started = time.time()

    def update(self, state: str, **details: object) -> None:
        payload = {
            "state": state,
            "updated_unix": time.time(),
            "elapsed_seconds": time.time() - self.started,
            **details,
        }
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(self.path)

    def __enter__(self):
        self.update("RUNNING", batches_completed=0)
        return self

    def __exit__(self, error_type, error, _traceback):
        if error is None:
            self.update("COMPLETED")
            return False
        self.update("FAILED", error_type=error_type.__name__, error_message=str(error))
        (self.root / "运行异常告警.txt").write_text(
            f"Run stopped safely.\nType: {error_type.__name__}\nMessage: {error}\n",
            encoding="utf-8",
        )
        return False


class TaskMonitor:
    """Low-overhead progress, ETA, disk and device telemetry for long runs."""

    def __init__(self, output, total_items: int, minimum_free_disk_gb: float = 2.0):
        from pathlib import Path

        self.root = Path(output)
        self.path = self.root / "运行遥测.csv"
        self.total_items = max(1, int(total_items))
        self.minimum_free_disk_gb = float(minimum_free_disk_gb)
        self.started = time.time()
        if not self.path.exists():
            self.path.write_text(
                "unix_time,completed,total,items_per_second,eta_seconds,memory_used_fraction,"
                "disk_free_gb,gpu_used_fraction,gpu_temperature_c,gpu_utilization_percent\n",
                encoding="utf-8",
            )

    def check_disk(self) -> float:
        disk_free = shutil.disk_usage(self.root).free / 1024**3
        if disk_free < self.minimum_free_disk_gb:
            raise OSError(
                f"disk guard stopped before next checkpoint: free={disk_free:.2f}GB, "
                f"minimum={self.minimum_free_disk_gb:.2f}GB"
            )
        return disk_free

    def observe(self, completed: int) -> dict[str, float | int | None]:
        elapsed = max(time.time() - self.started, 1e-9)
        rate = completed / elapsed
        eta = (
            max(0.0, (self.total_items - completed) / rate) if rate > 0.0 else math.inf
        )
        memory = psutil.virtual_memory()
        disk_free = self.check_disk()
        gpu = detect_gpu()
        row = {
            "completed": int(completed),
            "total": self.total_items,
            "items_per_second": rate,
            "eta_seconds": eta,
            "memory_used_fraction": 1.0 - memory.available / memory.total,
            "disk_free_gb": disk_free,
            "gpu_used_fraction": gpu.used_fraction if gpu else None,
            "gpu_temperature_c": gpu.temperature_c if gpu else None,
            "gpu_utilization_percent": gpu.utilization_percent if gpu else None,
        }
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            handle.write(
                f"{time.time():.3f},{row['completed']},{row['total']},{rate:.6f},{eta:.3f},"
                f"{row['memory_used_fraction']:.6f},{disk_free:.3f},"
                f"{'' if gpu is None else f'{gpu.used_fraction:.6f}'},"
                f"{'' if gpu is None else gpu.temperature_c},"
                f"{'' if gpu is None else gpu.utilization_percent}\n"
            )
        return row


def simulate_run(
    rows: int,
    assets: int,
    candidates: int,
    limits: RuntimeLimits,
    requested_shortlist: int,
    target_pool: int,
    requested_cache: int,
) -> dict[str, object]:
    catalog_storage_gb = candidates * 720 / 1024**3
    try:
        plan = preflight_plan(
            (int(rows), int(assets)),
            limits,
            requested_shortlist,
            target_pool,
            requested_cache,
        )
    except MemoryError as error:
        recommended_assets = int(assets)
        recommended_plan = None
        while recommended_assets > 1:
            recommended_assets = max(1, int(recommended_assets * 0.75))
            try:
                recommended_plan = preflight_plan(
                    (int(rows), recommended_assets),
                    limits,
                    requested_shortlist,
                    target_pool,
                    requested_cache,
                )
                break
            except MemoryError:
                continue
        return {
            "rows": int(rows),
            "assets": int(assets),
            "candidate_count": int(candidates),
            "runtime_limits": asdict(limits),
            "safe_to_start": False,
            "reason": str(error),
            "recommended_rows": int(rows),
            "recommended_assets": recommended_assets,
            "recommended_plan": (
                asdict(recommended_plan) if recommended_plan is not None else None
            ),
            "estimated_catalog_and_ledger_gb": round(catalog_storage_gb, 3),
            "execution_design": "bounded streaming; full candidate matrices are not retained",
        }
    return {
        "rows": int(rows),
        "assets": int(assets),
        "candidate_count": int(candidates),
        "runtime_limits": asdict(limits),
        "preflight_plan": asdict(plan),
        "estimated_catalog_and_ledger_gb": round(catalog_storage_gb, 3),
        "execution_design": "bounded streaming; full candidate matrices are not retained",
        "safe_to_start": True,
    }
