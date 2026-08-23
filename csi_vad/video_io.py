from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoMetadata:
    fps: float
    frame_count: int
    width: int
    height: int
    duration_sec: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def uniform_indices(total_frames: int, count: int) -> list[int]:
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if count <= 0:
        raise ValueError("count must be positive")
    if count == 1:
        return [0]
    return [int(i * (total_frames - 1) / (count - 1)) for i in range(count)]


def environment_keyframe_indices(total_frames: int) -> list[int]:
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    return [0, total_frames // 2, total_frames - 1]


def probe_video(video_path: str | Path) -> VideoMetadata:
    import cv2

    path = Path(video_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"video file not found: {path}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(
            "invalid video metadata: "
            f"fps={fps}, frames={frame_count}, size={width}x{height}"
        )
    return VideoMetadata(
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        duration_sec=frame_count / fps,
    )


def _sample_video_indices(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    indices: list[int],
    filenames: list[str],
) -> list[Path]:
    import cv2

    path = Path(video_path).expanduser().resolve()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    expected = [destination / filename for filename in filenames]
    if all(item.is_file() for item in expected):
        return expected

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    try:
        for index, output_path in zip(indices, expected):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"failed to decode frame {index} from {path}")
            if not cv2.imwrite(str(output_path), frame):
                raise RuntimeError(f"failed to write sampled frame: {output_path}")
    finally:
        capture.release()
    return expected


def sample_video_frames(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    count: int,
    prefix: str = "frame",
) -> list[Path]:
    metadata = probe_video(video_path)
    return _sample_video_indices(
        video_path,
        output_dir,
        indices=uniform_indices(metadata.frame_count, count),
        filenames=[f"{prefix}_{position:04d}.jpg" for position in range(count)],
    )


def sample_environment_keyframes(
    video_path: str | Path, output_dir: str | Path
) -> list[Path]:
    metadata = probe_video(video_path)
    return _sample_video_indices(
        video_path,
        output_dir,
        indices=environment_keyframe_indices(metadata.frame_count),
        filenames=["0000_first.jpg", "0001_middle.jpg", "0002_last.jpg"],
    )


def sample_video_frames_by_interval(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    interval: int,
) -> list[Path]:
    if interval <= 0:
        raise ValueError("interval must be positive")
    path = Path(video_path).expanduser().resolve()
    metadata = probe_video(path)
    indices = list(range(0, metadata.frame_count, interval)) or [0]
    return _sample_video_indices(
        path,
        output_dir,
        indices=indices,
        filenames=[f"temporal_{index:06d}.jpg" for index in indices],
    )


def sample_paths(paths: list[Path], count: int) -> list[Path]:
    if not paths:
        raise ValueError("cannot sample an empty path list")
    return [paths[index] for index in uniform_indices(len(paths), count)]
