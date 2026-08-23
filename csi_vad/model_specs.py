from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WeightSpec:
    filename: str
    url: str
    size_bytes: int
    sha256: str


RF_DETR_WEIGHT = WeightSpec(
    filename="rf-detr-medium.pth",
    url="https://storage.googleapis.com/rfdetr/medium_coco/checkpoint_best_regular.pth",
    size_bytes=404_992_918,
    sha256="749ff6071828aaffac63e204c4f4135ed3d6cdae4d702e086c360edc3b5768c8",
)
REID_WEIGHT = WeightSpec(
    filename="osnet_x0_25_msmt17.pt",
    url="https://drive.google.com/uc?id=1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF",
    size_bytes=3_057_863,
    sha256="6f57607fed9f502b9efed546108132ee715df5a5b6e6932c6269bacb47f59f99",
)

EXPECTED_DISTRIBUTIONS = {
    "torch": "2.6.0",
    "torchvision": "0.21.0",
    "transformers": "4.49.0",
    "accelerate": "1.4.0",
    "huggingface-hub": "0.29.3",
    "qwen-vl-utils": "0.0.8",
    "sentence-transformers": "5.2.0",
    "rfdetr": "1.4.0.post0",
    "boxmot": "16.0.9",
    "opencv-python": "4.11.0.86",
    "numpy": "1.26.4",
    "Pillow": "11.1.0",
    "gdown": "5.2.0",
}

IMPORT_CHECKS = (
    "torch",
    "torchvision",
    "transformers",
    "accelerate",
    "huggingface_hub",
    "qwen_vl_utils",
    "sentence_transformers",
    "rfdetr",
    "boxmot",
    "cv2",
    "PIL",
)


def version_matches(found: str, expected: str) -> bool:
    return found.split("+", maxsplit=1)[0] == expected


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_source_sha256() -> str:
    package_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        digest.update(path.relative_to(package_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def validate_weight_file(
    path: Path, *, expected_sha256: str, expected_size: int
) -> str | None:
    if not path.is_file():
        return f"missing weight file: {path}"
    if path.stat().st_size != expected_size:
        return (
            f"weight size mismatch: {path} "
            f"(expected {expected_size}, found {path.stat().st_size})"
        )
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_sha256:
        return (
            f"weight SHA-256 mismatch: {path} "
            f"(expected {expected_sha256}, found {actual_sha256})"
        )
    return None


def weight_identity(path: Path) -> dict[str, object]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        return {"path": str(resolved), "exists": False}
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "exists": True,
        "size_bytes": stat.st_size,
        "sha256": file_sha256(resolved),
    }
