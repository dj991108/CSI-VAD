from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


PAPER_FRAME_BUDGETS = (32, 64, 128, 256)
SCORE_AGGREGATIONS = ("max", "mean", "noisy_or")
LABEL_AGGREGATIONS = ("or", "majority", "and")


@dataclass
class PipelineConfig:
    """Validated runtime settings for one CSI-VAD inference run."""

    work_dir: Path = Path(".csi_vad_work")
    num_frames: int = 32
    score_aggregation: str = "noisy_or"
    label_aggregation: str = "or"
    dtype: str = "bf16"
    attention_implementation: str = "sdpa"
    device: str = "cuda"
    local_files_only: bool = False

    vision_language_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    vision_language_revision: str = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
    language_model: str = "Qwen/Qwen2.5-7B-Instruct"
    language_model_revision: str = "a09a35458c702b33eeacc393d103063234e8bc28"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

    max_pixels: int = 360 * 420
    max_new_tokens: int = 512
    caption_max_new_tokens: int = 124
    embedding_batch_size: int = 64
    embedding_device: str = "cpu"

    object_fps: float = 10.0
    detection_confidence: float = 0.4
    detection_resize_long_side: int = 576
    detector_weights: Path = Path("weights/rf-detr-medium.pth")
    reid_weights: Path = Path("weights/osnet_x0_25_msmt17.pt")

    temporal_frame_interval: int = 16
    temporal_merge_similarity: float = 0.92
    temporal_dedup_similarity: float = 0.95
    temporal_boundary_similarity: float = 0.70
    temporal_smooth_window: int = 5
    temporal_boundary_debounce: int = 1
    temporal_min_boundary_gap: int = 0
    temporal_max_segments: int = 20

    interest_classes: tuple[str, ...] = field(
        default_factory=lambda: (
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "cat",
            "dog",
            "backpack",
            "handbag",
            "baseball bat",
            "bottle",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "chair",
            "tv",
            "laptop",
            "cell phone",
        )
    )

    def __post_init__(self) -> None:
        self.work_dir = Path(self.work_dir).expanduser()
        self.detector_weights = Path(self.detector_weights).expanduser()
        self.reid_weights = Path(self.reid_weights).expanduser()
        if self.num_frames not in PAPER_FRAME_BUDGETS:
            raise ValueError(
                f"num_frames must be one of {PAPER_FRAME_BUDGETS}, got {self.num_frames}"
            )
        if self.score_aggregation not in SCORE_AGGREGATIONS:
            raise ValueError(f"score_aggregation must be one of {SCORE_AGGREGATIONS}")
        if self.label_aggregation not in LABEL_AGGREGATIONS:
            raise ValueError(f"label_aggregation must be one of {LABEL_AGGREGATIONS}")
        if self.dtype not in ("bf16", "fp16"):
            raise ValueError("dtype must be 'bf16' or 'fp16'")
        if self.device != "cuda":
            raise ValueError(
                "The official CSI-VAD inference path requires device='cuda'"
            )
        if self.object_fps <= 0:
            raise ValueError("object_fps must be positive")
        if not 0.0 <= self.detection_confidence <= 1.0:
            raise ValueError("detection_confidence must be within [0, 1]")
        if self.temporal_frame_interval <= 0:
            raise ValueError("temporal_frame_interval must be positive")
        if self.temporal_max_segments <= 0:
            raise ValueError("temporal_max_segments must be positive")

    def public_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values.pop("work_dir")
        values.pop("detector_weights")
        values.pop("reid_weights")
        for key in (
            "vision_language_model",
            "vision_language_revision",
            "language_model",
            "language_model_revision",
            "embedding_model",
            "embedding_revision",
        ):
            values.pop(key)
        values["interest_classes"] = list(self.interest_classes)
        values["models"] = {
            "vision_language": {
                "id": self.vision_language_model,
                "revision": self.vision_language_revision,
            },
            "language": {
                "id": self.language_model,
                "revision": self.language_model_revision,
            },
            "embedding": {
                "id": self.embedding_model,
                "revision": self.embedding_revision,
            },
            "detector": {"name": "RF-DETR Medium", "package": "rfdetr==1.4.0.post0"},
            "tracker": {"name": "StrongSORT", "package": "boxmot==16.0.9"},
        }
        return values
