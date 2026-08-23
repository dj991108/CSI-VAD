#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from csi_vad.model_specs import (  # noqa: E402
    REID_WEIGHT,
    RF_DETR_WEIGHT,
    WeightSpec,
    validate_weight_file,
)

HF_SNAPSHOTS = {
    "vision-language": (
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "cc594898137f460bfe9f0759e9844b3ce807cfb5",
    ),
    "language": (
        "Qwen/Qwen2.5-7B-Instruct",
        "a09a35458c702b33eeacc393d103063234e8bc28",
    ),
    "embedding": (
        "sentence-transformers/all-MiniLM-L6-v2",
        "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    ),
}


def download_url(spec: WeightSpec, destination: Path) -> None:
    if (
        validate_weight_file(
            destination,
            expected_sha256=spec.sha256,
            expected_size=spec.size_bytes,
        )
        is None
    ):
        print(f"Found: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    print(f"Downloading {spec.url} -> {destination}")
    urllib.request.urlretrieve(spec.url, temporary)
    failure = validate_weight_file(
        temporary,
        expected_sha256=spec.sha256,
        expected_size=spec.size_bytes,
    )
    if failure:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(failure)
    os.replace(temporary, destination)


def download_reid(destination: Path) -> None:
    if (
        validate_weight_file(
            destination,
            expected_sha256=REID_WEIGHT.sha256,
            expected_size=REID_WEIGHT.size_bytes,
        )
        is None
    ):
        print(f"Found: {destination}")
        return
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "gdown is not installed; install requirements.txt first"
        ) from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    print(f"Downloading StrongSORT ReID weights -> {destination}")
    temporary.unlink(missing_ok=True)
    gdown.download(REID_WEIGHT.url, str(temporary), quiet=False, fuzzy=True)
    failure = validate_weight_file(
        temporary,
        expected_sha256=REID_WEIGHT.sha256,
        expected_size=REID_WEIGHT.size_bytes,
    )
    if failure:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(failure)
    os.replace(temporary, destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download CSI-VAD runtime models.")
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument(
        "--components",
        nargs="+",
        choices=(*HF_SNAPSHOTS, "detector", "reid"),
        default=[*HF_SNAPSHOTS, "detector", "reid"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = set(args.components)
    if selected.intersection(HF_SNAPSHOTS):
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface-hub is not installed; install requirements.txt first"
            ) from exc
        for component, (repo_id, revision) in HF_SNAPSHOTS.items():
            if component in selected:
                print(f"Prefetching {repo_id}@{revision}")
                snapshot_download(repo_id=repo_id, revision=revision)
    weights_dir = args.weights_dir.expanduser().resolve()
    if "detector" in selected:
        download_url(RF_DETR_WEIGHT, weights_dir / RF_DETR_WEIGHT.filename)
    if "reid" in selected:
        download_reid(weights_dir / REID_WEIGHT.filename)
    print("Model preparation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
