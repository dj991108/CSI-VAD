from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..config import PipelineConfig


RFDETR_COCO_CATEGORIES: dict[int, str] = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}


def build_interest_class_ids(names: Iterable[str]) -> set[int]:
    mapping = {
        name: category_id for category_id, name in RFDETR_COCO_CATEGORIES.items()
    }
    unknown = sorted(set(names) - mapping.keys())
    if unknown:
        raise ValueError("unknown COCO class names: " + ", ".join(unknown))
    return {mapping[name] for name in names}


def normalize_detections(
    detections: Any,
    *,
    original_size: tuple[int, int],
    inference_size: tuple[int, int],
    interest_class_ids: set[int],
    topk: int | None,
) -> np.ndarray:
    if detections is None:
        return np.zeros((0, 6), dtype=np.float32)
    class_ids = getattr(detections, "class_id", None)
    confidence = getattr(detections, "confidence", None)
    boxes = getattr(detections, "xyxy", None)
    if isinstance(detections, dict):
        class_ids = detections.get("class_id", class_ids)
        confidence = detections.get("confidence", confidence)
        boxes = detections.get("xyxy", boxes)
    if class_ids is None or confidence is None or boxes is None:
        return np.zeros((0, 6), dtype=np.float32)
    classes = np.asarray(class_ids, dtype=np.int32).reshape(-1)
    scores = np.asarray(confidence, dtype=np.float32).reshape(-1)
    xyxy = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if not len(xyxy):
        return np.zeros((0, 6), dtype=np.float32)
    keep = np.asarray(
        [int(value) in interest_class_ids for value in classes], dtype=bool
    )
    xyxy, scores, classes = xyxy[keep], scores[keep], classes[keep]
    if topk is not None and len(scores) > topk:
        order = np.argsort(-scores)[:topk]
        xyxy, scores, classes = xyxy[order], scores[order], classes[order]
    if not len(xyxy):
        return np.zeros((0, 6), dtype=np.float32)
    original_h, original_w = original_size
    inference_h, inference_w = inference_size
    xyxy[:, (0, 2)] *= float(original_w) / float(inference_w)
    xyxy[:, (1, 3)] *= float(original_h) / float(inference_h)
    return np.concatenate(
        [xyxy, scores[:, None], classes.astype(np.float32)[:, None]], axis=1
    ).astype(np.float32)


def _resize_keep_aspect(frame: np.ndarray, long_side: int) -> np.ndarray:
    import cv2

    height, width = frame.shape[:2]
    if max(height, width) <= long_side:
        return frame
    scale = long_side / float(max(height, width))
    return cv2.resize(
        frame,
        (max(2, round(width * scale)), max(2, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _track_color(track_id: int) -> tuple[int, int, int]:
    import cv2

    hue = int(((track_id * 2654435761) & 0xFFFFFFFF) % 180)
    hsv = np.uint8([[[hue, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _draw_tracks(frame: np.ndarray, tracks: np.ndarray) -> np.ndarray:
    import cv2

    output = frame.copy()
    height, width = output.shape[:2]
    for track in tracks if tracks is not None else []:
        x1, y1, x2, y2 = track[:4]
        track_id = int(track[4])
        point1 = (
            max(0, min(width - 1, round(float(x1)))),
            max(0, min(height - 1, round(float(y1)))),
        )
        point2 = (
            max(0, min(width - 1, round(float(x2)))),
            max(0, min(height - 1, round(float(y2)))),
        )
        cv2.rectangle(output, point1, point2, _track_color(track_id), 1)
    return output


def _extract_at_fps(video_path: Path, output_dir: Path, fps: float) -> list[Path]:
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("frame_*.jpg"))
    marker = output_dir / "complete.json"
    if existing and marker.is_file():
        return existing
    for path in existing:
        path.unlink()
    marker.unlink(missing_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_fps <= 0 or frame_count <= 0:
        capture.release()
        raise RuntimeError("invalid video metadata for object frame extraction")
    duration = frame_count / source_fps
    sample_count = max(1, int(np.ceil(duration * fps)))
    indices = sorted(
        {
            min(frame_count - 1, int(position * source_fps / fps))
            for position in range(sample_count)
        }
    )
    paths: list[Path] = []
    try:
        for output_index, frame_index in enumerate(indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"failed to decode object frame {frame_index}")
            path = output_dir / f"frame_{output_index:06d}.jpg"
            if not cv2.imwrite(str(path), frame):
                raise RuntimeError(f"failed to write object frame: {path}")
            paths.append(path)
    finally:
        capture.release()
    marker.write_text(json.dumps({"frames": len(paths), "fps": fps}), encoding="utf-8")
    return paths


class ObjectContextBuilder:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def build(self, video_path: Path, work_dir: Path) -> list[Path]:
        import cv2
        import torch
        from boxmot import StrongSort
        from PIL import Image
        from rfdetr import RFDETRMedium

        missing_weights = [
            path
            for path in (self.config.detector_weights, self.config.reid_weights)
            if not path.is_file()
        ]
        if missing_weights:
            raise FileNotFoundError(
                "required weights not found: "
                + ", ".join(str(path) for path in missing_weights)
                + ". "
                "Run `python scripts/download_models.py`."
            )
        frames_dir = work_dir / "frames_10fps"
        overlay_dir = work_dir / "overlays"
        frame_paths = _extract_at_fps(video_path, frames_dir, self.config.object_fps)
        marker = overlay_dir / "complete.json"
        overlays = sorted(overlay_dir.glob("overlay_*.jpg"))
        if marker.is_file() and len(overlays) == len(frame_paths):
            return overlays
        overlay_dir.mkdir(parents=True, exist_ok=True)
        for path in overlays:
            path.unlink()
        marker.unlink(missing_ok=True)

        interest_ids = build_interest_class_ids(self.config.interest_classes)
        detector = RFDETRMedium(pretrain_weights=str(self.config.detector_weights))
        tracker = StrongSort(
            reid_weights=self.config.reid_weights,
            device="0",
            half=self.config.dtype == "fp16",
        )
        output_paths: list[Path] = []
        try:
            for index, frame_path in enumerate(frame_paths):
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    raise RuntimeError(f"failed to read object frame: {frame_path}")
                resized = _resize_keep_aspect(
                    frame, self.config.detection_resize_long_side
                )
                detections = detector.predict(
                    Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)),
                    threshold=self.config.detection_confidence,
                )
                array = normalize_detections(
                    detections,
                    original_size=frame.shape[:2],
                    inference_size=resized.shape[:2],
                    interest_class_ids=interest_ids,
                    topk=None,
                )
                tracks = tracker.update(array, frame)
                output_path = overlay_dir / f"overlay_{index:06d}.jpg"
                if not cv2.imwrite(str(output_path), _draw_tracks(frame, tracks)):
                    raise RuntimeError(f"failed to write object overlay: {output_path}")
                output_paths.append(output_path)
        finally:
            del tracker
            del detector
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        marker.write_text(json.dumps({"frames": len(output_paths)}), encoding="utf-8")
        return output_paths
