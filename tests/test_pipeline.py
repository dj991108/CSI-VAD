from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from csi_vad.config import PipelineConfig
from csi_vad.pipeline import Pipeline
from csi_vad.schemas import BranchResult


def _write_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (32, 24))
    assert writer.isOpened()
    for index in range(8):
        writer.write(np.full((24, 32, 3), index * 20, dtype=np.uint8))
    writer.release()


def test_pipeline_writes_final_result_without_context_payloads(tmp_path: Path) -> None:
    events: list[str] = []
    video = tmp_path / "sample.mp4"
    output = tmp_path / "result.json"
    _write_video(video)

    class FakeManager:
        def release(self, name: str) -> None:
            events.append(f"release:{name}")

        def release_all(self) -> None:
            events.append("release:all")

    class FakeObjectBuilder:
        def build(self, video_path: Path, work_dir: Path) -> list[Path]:
            events.append("object_context")
            work_dir.mkdir(parents=True, exist_ok=True)
            paths = []
            for index in range(2):
                path = work_dir / f"overlay_{index}.jpg"
                cv2.imwrite(str(path), np.zeros((24, 32, 3), dtype=np.uint8))
                paths.append(path)
            return paths

    class FakeEnvironmentBuilder:
        def build(self, keyframes: list[Path]) -> dict[str, object]:
            events.append("environment_context")
            return {
                "place": "road",
                "daytime": "day",
                "environment_type": "outdoor road",
                "typical_situation": {"visible_infrastructure": ["road"]},
            }

    class FakeVisualRecognizer:
        def caption(self, path: Path) -> str:
            return f"caption {path.stem}"

        def environment(self, paths: list[Path], context: dict) -> BranchResult:
            events.append("environment_recognition")
            return BranchResult(0, 0.2, "normal road activity")

        def object(self, paths: list[Path]) -> BranchResult:
            events.append("object_recognition")
            return BranchResult(1, 0.8, "tracked physical conflict")

    class FakeTemporalBuilder:
        def caption_and_segment(
            self, paths: list[Path], recognizer: object
        ) -> list[list[str]]:
            events.append("temporal_captioning")
            return [["person approaches", "physical conflict begins"]]

        def summarize(self, groups: list[list[str]], recognizer: object) -> dict:
            events.append("temporal_summary")
            return {"scene_segments": [{"segment_id": 1, "scene_state": "conflict"}]}

    class FakeTemporalRecognizer:
        def recognize(self, context: dict) -> BranchResult:
            events.append("temporal_recognition")
            return BranchResult(1, 0.7, "event changes into conflict")

    pipeline = Pipeline(
        PipelineConfig(work_dir=tmp_path / "work"),
        manager=FakeManager(),
        object_builder=FakeObjectBuilder(),
        environment_builder=FakeEnvironmentBuilder(),
        visual_recognizer=FakeVisualRecognizer(),
        temporal_builder=FakeTemporalBuilder(),
        temporal_recognizer=FakeTemporalRecognizer(),
    )

    result = pipeline.run(video, output)

    assert output.is_file()
    assert result["prediction"]["label"] == 1
    assert result["prediction"]["score"] == 0.952
    assert "environment_context" not in result
    assert "temporal_context" not in result
    assert events == [
        "object_context",
        "environment_context",
        "temporal_captioning",
        "environment_recognition",
        "object_recognition",
        "release:vision_language",
        "temporal_summary",
        "temporal_recognition",
        "release:language",
        "release:all",
    ]
    assert set(result["timings_sec"]) == {
        "object_context",
        "environment_context",
        "temporal_captioning",
        "environment_recognition",
        "object_recognition",
        "temporal_summary",
        "temporal_recognition",
        "total",
    }
    assert result["cache_hits"] == []

    events.clear()
    resumed = pipeline.run(video, output)

    assert resumed["cache_hits"] == [
        "environment_context",
        "temporal_captioning",
        "environment_recognition",
        "object_recognition",
        "temporal_summary",
        "temporal_recognition",
    ]
    assert set(resumed["timings_sec"]) == set(result["timings_sec"])
    assert events == [
        "object_context",
        "release:vision_language",
        "release:language",
        "release:all",
    ]


def test_manifest_fingerprints_weight_contents_and_prompts(tmp_path: Path) -> None:
    video = tmp_path / "sample.mp4"
    detector = tmp_path / "detector.pth"
    reid = tmp_path / "reid.pt"
    video.write_bytes(b"video")
    detector.write_bytes(b"detector-v1")
    reid.write_bytes(b"reid-v1")
    config = PipelineConfig(
        work_dir=tmp_path / "work",
        detector_weights=detector,
        reid_weights=reid,
    )
    pipeline = Pipeline(config)
    metadata = SimpleNamespace(to_dict=lambda: {"frame_count": 1})

    first = pipeline._manifest(video, metadata)
    detector.write_bytes(b"detector-v2")
    second = pipeline._manifest(video, metadata)

    assert first["cache_schema_version"] == 1
    assert len(first["implementation_sha256"]) == 64
    assert len(first["prompt_sha256"]) == 64
    assert (
        first["weights"]["detector"]["sha256"]
        != second["weights"]["detector"]["sha256"]
    )
