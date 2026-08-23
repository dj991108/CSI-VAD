from pathlib import Path

import cv2
import numpy as np

from csi_vad.video_io import (
    environment_keyframe_indices,
    probe_video,
    sample_video_frames,
    uniform_indices,
)


def _write_video(path: Path, frames: int = 5, fps: float = 5.0) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (32, 24))
    assert writer.isOpened()
    for index in range(frames):
        frame = np.full((24, 32, 3), index * 30, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_uniform_indices_include_first_and_last() -> None:
    assert uniform_indices(total_frames=10, count=4) == [0, 3, 6, 9]


def test_uniform_indices_repeat_short_video_frames() -> None:
    assert uniform_indices(total_frames=2, count=4) == [0, 0, 0, 1]


def test_environment_keyframes_use_first_middle_and_last() -> None:
    assert environment_keyframe_indices(total_frames=10) == [0, 5, 9]


def test_probe_and_sample_synthetic_video(tmp_path: Path) -> None:
    video = tmp_path / "sample.mp4"
    _write_video(video)

    metadata = probe_video(video)
    paths = sample_video_frames(video, tmp_path / "frames", count=3)

    assert metadata.frame_count == 5
    assert metadata.width == 32
    assert metadata.height == 24
    assert len(paths) == 3
    assert all(path.exists() for path in paths)
