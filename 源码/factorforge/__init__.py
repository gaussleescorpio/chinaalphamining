"""FactorForge Max public interface."""

from .config import ResearchConfig
from .contracts import AtomSpec, CandidateRecord, DecisionRecord, PanelData
from .pipeline import FactorDiscoveryPipeline

__all__ = [
    "AtomSpec",
    "CandidateRecord",
    "DecisionRecord",
    "FactorDiscoveryPipeline",
    "PanelData",
    "ResearchConfig",
]

__version__ = "1.0.0"
