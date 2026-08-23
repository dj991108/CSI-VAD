"""CSI-VAD single-video inference package."""

from .config import PipelineConfig
from .schemas import BranchResult

__all__ = ["BranchResult", "PipelineConfig"]

__version__ = "1.0.0"
