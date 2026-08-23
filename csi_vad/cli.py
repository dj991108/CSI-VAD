from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Sequence

from .config import (
    LABEL_AGGREGATIONS,
    PAPER_FRAME_BUDGETS,
    SCORE_AGGREGATIONS,
    PipelineConfig,
)
from .pipeline import Pipeline
from .preflight import run_preflight
from .video_io import probe_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CSI-VAD inference for one local video."
    )
    parser.add_argument("--video", type=Path, required=True, help="Input video path")
    parser.add_argument(
        "--output", type=Path, required=True, help="Final result JSON path"
    )
    parser.add_argument(
        "--num-frames", type=int, choices=PAPER_FRAME_BUDGETS, default=32
    )
    parser.add_argument(
        "--score-aggregation", choices=SCORE_AGGREGATIONS, default="noisy_or"
    )
    parser.add_argument("--label-aggregation", choices=LABEL_AGGREGATIONS, default="or")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--clean-workdir", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the video and print the effective configuration without loading models",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    video = args.video.expanduser().resolve()
    output = args.output.expanduser().resolve()
    work_dir = (
        args.work_dir.expanduser().resolve()
        if args.work_dir
        else output.parent / ".csi_vad_work" / video.stem
    )
    weights_dir = args.weights_dir.expanduser().resolve()
    try:
        config = PipelineConfig(
            work_dir=work_dir,
            num_frames=args.num_frames,
            score_aggregation=args.score_aggregation,
            label_aggregation=args.label_aggregation,
            dtype=args.dtype,
            local_files_only=args.local_files_only,
            detector_weights=weights_dir / "rf-detr-medium.pth",
            reid_weights=weights_dir / "osnet_x0_25_msmt17.pt",
        )
        metadata = probe_video(video)
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "video": {"path": str(video), **metadata.to_dict()},
                        "output": str(output),
                        "work_dir": str(work_dir),
                        "config": config.public_dict(),
                    },
                    indent=2,
                )
            )
            return 0
        if args.clean_workdir and work_dir.exists():
            shutil.rmtree(work_dir)
        run_preflight(config)
        result = Pipeline(config).run(video, output)
        print(json.dumps(result["prediction"], indent=2))
        print(f"Result: {output}")
        return 0
    except Exception as exc:
        print(f"CSI-VAD failed: {exc}", file=sys.stderr)
        return 1
