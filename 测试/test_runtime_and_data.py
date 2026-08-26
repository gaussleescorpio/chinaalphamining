from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from factorforge.adapters import LongFrameAdapter
from factorforge.data_quality import audit_long_frame
from factorforge.runtime import resolve_runtime_limits


def test_safe_profile_clamps_user_load() -> None:
    config = SimpleNamespace(
        runtime_profile="safe",
        batch_size=10_000,
        workers=20,
        max_memory_fraction=0.90,
        gpu_memory_fraction=0.90,
        gpu_temperature_limit_c=88,
    )
    limits = resolve_runtime_limits(config)
    assert limits.batch_size == 64
    assert limits.workers == 1
    assert limits.maximum_memory_fraction == 0.60


def test_membership_mask_prevents_backfilled_universe() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "symbol": ["A", "A"],
            "value": [1.0, 2.0],
            "eligible": [False, True],
        }
    )
    report = audit_long_frame(frame, "date", "symbol", ["value"], "eligible")
    assert report.inactive_rows == 1
    panel = LongFrameAdapter("date", "symbol", ["value"], "eligible").materialize(frame)
    assert pd.isna(panel.fields["value"][0, 0])
    assert panel.fields["value"][1, 0] == 2.0
