from pathlib import Path

import pytest

from csi_vad.config import PipelineConfig


def test_defaults_match_public_release_contract(tmp_path: Path) -> None:
    cfg = PipelineConfig(work_dir=tmp_path / "work")

    assert cfg.num_frames == 32
    assert cfg.object_fps == 10.0
    assert cfg.detection_confidence == 0.4
    assert cfg.detection_resize_long_side == 576
    assert cfg.score_aggregation == "noisy_or"
    assert cfg.label_aggregation == "or"
    assert cfg.attention_implementation == "sdpa"
    assert cfg.dtype == "bf16"


@pytest.mark.parametrize("num_frames", [0, -1, 33])
def test_frame_budget_must_be_a_paper_budget(num_frames: int, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="num_frames"):
        PipelineConfig(num_frames=num_frames, work_dir=tmp_path)


def test_public_config_excludes_local_paths(tmp_path: Path) -> None:
    cfg = PipelineConfig(work_dir=tmp_path / "private")

    public = cfg.public_dict()

    assert "work_dir" not in public
    assert public["num_frames"] == 32
    assert public["models"]["vision_language"]["revision"]
