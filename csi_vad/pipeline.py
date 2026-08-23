from __future__ import annotations

import hashlib
import json
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .contexts.environment import EnvironmentContextBuilder
from .contexts.object_tracking import ObjectContextBuilder
from .contexts.temporal import TemporalContextBuilder
from .model_manager import ModelManager
from .model_specs import file_sha256, package_source_sha256, weight_identity
from .prompts import prompt_fingerprint
from .recognition import TemporalRecognizer, VisualRecognizer
from .schemas import BranchResult, build_final_result
from .video_io import (
    probe_video,
    sample_environment_keyframes,
    sample_paths,
    sample_video_frames,
    sample_video_frames_by_interval,
)


CACHE_SCHEMA_VERSION = 1
TIMING_STAGES = (
    "object_context",
    "environment_context",
    "temporal_captioning",
    "environment_recognition",
    "object_recognition",
    "temporal_summary",
    "temporal_recognition",
)


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


class Pipeline:
    def __init__(
        self,
        config: PipelineConfig,
        *,
        manager: Any | None = None,
        object_builder: Any | None = None,
        environment_builder: Any | None = None,
        visual_recognizer: Any | None = None,
        temporal_builder: Any | None = None,
        temporal_recognizer: Any | None = None,
    ):
        self.config = config
        self.manager = manager or ModelManager(config)
        self.object_builder = object_builder or ObjectContextBuilder(config)
        self.environment_builder = environment_builder or EnvironmentContextBuilder(
            self.manager, config
        )
        self.visual_recognizer = visual_recognizer or VisualRecognizer(
            self.manager, config
        )
        self.temporal_builder = temporal_builder or TemporalContextBuilder(
            self.manager, config
        )
        self.temporal_recognizer = temporal_recognizer or TemporalRecognizer(
            self.manager, config
        )

    def _manifest(self, video: Path, metadata: Any) -> dict[str, Any]:
        stat = video.stat()
        public_config = self.config.public_dict()
        fingerprint = hashlib.sha256(
            json.dumps(public_config, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "implementation_sha256": package_source_sha256(),
            "prompt_sha256": prompt_fingerprint(),
            "video": {
                "path": str(video),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": file_sha256(video),
            },
            "video_metadata": metadata.to_dict(),
            "config_sha256": fingerprint,
            "weights": {
                "detector": weight_identity(self.config.detector_weights),
                "reid": weight_identity(self.config.reid_weights),
            },
        }

    def _check_manifest(self, manifest: dict[str, Any]) -> None:
        path = self.config.work_dir / "manifest.json"
        previous = _read_json(path)
        if previous is not None and previous != manifest:
            raise RuntimeError(
                f"work directory belongs to a different video or configuration: {path}. "
                "Use another --work-dir or pass --clean-workdir."
            )
        if previous is None:
            _atomic_write_json(path, manifest)

    @staticmethod
    def _load_branch(path: Path) -> BranchResult | None:
        value = _read_json(path)
        if not isinstance(value, dict):
            return None
        try:
            return BranchResult(
                label=int(value["label"]),
                score=float(value["score"]),
                explanation=str(value["explanation"]),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def run(self, video_path: str | Path, output_path: str | Path) -> dict[str, Any]:
        started = time.perf_counter()
        timings = {stage: 0.0 for stage in TIMING_STAGES}
        cache_hits: list[str] = []
        video = Path(video_path).expanduser().resolve()
        output = Path(output_path).expanduser().resolve()
        metadata = probe_video(video)
        self.config.work_dir.mkdir(parents=True, exist_ok=True)
        self._check_manifest(self._manifest(video, metadata))

        context_dir = self.config.work_dir / "contexts"
        branch_dir = self.config.work_dir / "branches"
        sampled_dir = self.config.work_dir / "sampled_frames"
        context_dir.mkdir(parents=True, exist_ok=True)
        branch_dir.mkdir(parents=True, exist_ok=True)

        environment_context_path = context_dir / "environment.json"
        caption_groups_path = context_dir / "temporal_caption_groups.json"
        temporal_context_path = context_dir / "temporal.json"

        try:
            stage_started = time.perf_counter()
            overlays = self.object_builder.build(video, self.config.work_dir / "object")
            timings["object_context"] = time.perf_counter() - stage_started

            visual_frames = sample_video_frames(
                video,
                sampled_dir / f"visual_{self.config.num_frames}",
                count=self.config.num_frames,
            )
            keyframes = sample_environment_keyframes(video, sampled_dir / "environment")
            temporal_frames = sample_video_frames_by_interval(
                video,
                sampled_dir / f"temporal_{self.config.temporal_frame_interval}",
                interval=self.config.temporal_frame_interval,
            )

            environment_context = _read_json(environment_context_path)
            if not isinstance(environment_context, dict):
                stage_started = time.perf_counter()
                environment_context = self.environment_builder.build(keyframes)
                _atomic_write_json(environment_context_path, environment_context)
                timings["environment_context"] = time.perf_counter() - stage_started
            else:
                cache_hits.append("environment_context")

            caption_groups = _read_json(caption_groups_path)
            if not isinstance(caption_groups, list):
                stage_started = time.perf_counter()
                caption_groups = self.temporal_builder.caption_and_segment(
                    temporal_frames, self.visual_recognizer
                )
                _atomic_write_json(caption_groups_path, caption_groups)
                timings["temporal_captioning"] = time.perf_counter() - stage_started
            else:
                cache_hits.append("temporal_captioning")

            environment_result = self._load_branch(branch_dir / "environment.json")
            if environment_result is None:
                stage_started = time.perf_counter()
                environment_result = self.visual_recognizer.environment(
                    visual_frames, environment_context
                )
                _atomic_write_json(
                    branch_dir / "environment.json", environment_result.to_dict()
                )
                timings["environment_recognition"] = time.perf_counter() - stage_started
            else:
                cache_hits.append("environment_recognition")

            object_result = self._load_branch(branch_dir / "object.json")
            if object_result is None:
                stage_started = time.perf_counter()
                object_result = self.visual_recognizer.object(
                    sample_paths(list(overlays), self.config.num_frames)
                )
                _atomic_write_json(branch_dir / "object.json", object_result.to_dict())
                timings["object_recognition"] = time.perf_counter() - stage_started
            else:
                cache_hits.append("object_recognition")

            self.manager.release("vision_language")

            temporal_context = _read_json(temporal_context_path)
            if not isinstance(temporal_context, dict):
                stage_started = time.perf_counter()
                temporal_context = self.temporal_builder.summarize(
                    caption_groups, self.temporal_recognizer
                )
                _atomic_write_json(temporal_context_path, temporal_context)
                timings["temporal_summary"] = time.perf_counter() - stage_started
            else:
                cache_hits.append("temporal_summary")

            temporal_result = self._load_branch(branch_dir / "time.json")
            if temporal_result is None:
                stage_started = time.perf_counter()
                temporal_result = self.temporal_recognizer.recognize(temporal_context)
                _atomic_write_json(branch_dir / "time.json", temporal_result.to_dict())
                timings["temporal_recognition"] = time.perf_counter() - stage_started
            else:
                cache_hits.append("temporal_recognition")

            self.manager.release("language")
            timings["total"] = time.perf_counter() - started

            branches = OrderedDict(
                environment=environment_result,
                object=object_result,
                time=temporal_result,
            )
            result = build_final_result(
                video={"name": video.name, "path": str(video), **metadata.to_dict()},
                config=self.config.public_dict(),
                branches=branches,
                score_aggregation=self.config.score_aggregation,
                label_aggregation=self.config.label_aggregation,
                timings=timings,
                cache_hits=cache_hits,
            )
            _atomic_write_json(output, result)
            return result
        finally:
            self.manager.release_all()
