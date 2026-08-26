from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ResidualRole(str, Enum):
    """The economic job assigned to a residual candidate before evaluation."""

    ERROR_REPAIR = "error_repair"
    STATE_SPECIALIST = "state_specialist"
    TAIL_LOSS_REDUCER = "tail_loss_reducer"
    DIVERSIFIER = "diversifier"


@dataclass(frozen=True)
class FieldAvailability:
    name: str
    available_lag_sessions: int
    source_timestamp_field: str | None = None
    point_in_time_required: bool = True

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("field name is required")
        if self.available_lag_sessions < 0:
            raise ValueError(f"negative availability lag for {self.name}")


@dataclass(frozen=True)
class AShareResidualConfig:
    """Frozen defaults for daily cash A-share residual research."""

    decision_clock: str = "T_CLOSE"
    entry_offset_sessions: int = 1
    holding_period_sessions: int = 5
    purge_sessions: int = 6
    development_start: str = "2018-01-01"
    development_end: str = "2024-12-31"
    orientation_end: str = "2020-12-31"
    stage_one_start: str = "2021-01-01"
    stage_one_end: str = "2022-12-31"
    stage_two_start: str = "2023-01-01"
    minimum_cross_section: int = 100
    minimum_days_per_block: int = 80
    minimum_positive_block_fraction: float = 0.60
    maximum_pool_score_correlation: float = 0.70
    ridge_alpha: float = 10.0
    random_seed: int = 20260823

    def validate(self) -> None:
        if self.decision_clock != "T_CLOSE":
            raise ValueError("the first release supports T-close decisions only")
        if self.entry_offset_sessions < 1:
            raise ValueError("A-share close signals cannot enter before T+1")
        required_purge = self.entry_offset_sessions + self.holding_period_sessions
        if self.purge_sessions < required_purge:
            raise ValueError(f"purge_sessions must be at least {required_purge}")
        if self.minimum_cross_section < 30:
            raise ValueError("minimum cross-section is too small")
        if not self.orientation_end < self.stage_one_start <= self.stage_one_end < self.stage_two_start:
            raise ValueError("orientation and evaluation stages are not ordered")
        if not 0.0 < self.minimum_positive_block_fraction <= 1.0:
            raise ValueError("positive block fraction is invalid")
        if not 0.0 < self.maximum_pool_score_correlation < 1.0:
            raise ValueError("pool correlation cap is invalid")
