#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from csi_vad.config import PipelineConfig  # noqa: E402
from csi_vad.model_specs import version_matches  # noqa: E402, F401
from csi_vad.preflight import PreflightError, run_preflight  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the CSI-VAD runtime environment."
    )
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument("--skip-weights", action="store_true")
    parser.add_argument("--skip-model-cache", action="store_true")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    weights_dir = args.weights_dir.expanduser().resolve()
    config = PipelineConfig(
        dtype=args.dtype,
        detector_weights=weights_dir / "rf-detr-medium.pth",
        reid_weights=weights_dir / "osnet_x0_25_msmt17.pt",
    )
    try:
        run_preflight(
            config,
            check_weights=not args.skip_weights,
            check_model_cache=not args.skip_model_cache,
        )
    except PreflightError as exc:
        print(exc, file=sys.stderr)
        return 1
    print("CSI-VAD environment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
