from __future__ import annotations

import importlib
import importlib.metadata
import sys
from typing import Iterable

from .config import PipelineConfig
from .model_specs import (
    EXPECTED_DISTRIBUTIONS,
    IMPORT_CHECKS,
    REID_WEIGHT,
    RF_DETR_WEIGHT,
    validate_weight_file,
    version_matches,
)


class PreflightError(RuntimeError):
    pass


def _check_python_and_packages() -> list[str]:
    failures: list[str] = []
    if sys.version_info[:2] != (3, 11):
        failures.append(f"Python 3.11 required, found {sys.version.split()[0]}")
    for distribution, expected in EXPECTED_DISTRIBUTIONS.items():
        try:
            found = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            failures.append(f"missing package: {distribution}=={expected}")
            continue
        if not version_matches(found, expected):
            failures.append(f"{distribution}: expected {expected}, found {found}")
    for module in IMPORT_CHECKS:
        try:
            importlib.import_module(module)
        except Exception as exc:
            failures.append(f"failed to import {module}: {exc}")
    return failures


def _check_cuda(dtype: str) -> list[str]:
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return ["CUDA is not available to PyTorch"]
    if dtype == "bf16" and not torch.cuda.is_bf16_supported():
        return ["the selected GPU does not support bfloat16; use --dtype fp16"]
    return []


def _check_weights(config: PipelineConfig) -> list[str]:
    failures = []
    checks = (
        (config.detector_weights, RF_DETR_WEIGHT),
        (config.reid_weights, REID_WEIGHT),
    )
    for path, spec in checks:
        failure = validate_weight_file(
            path,
            expected_sha256=spec.sha256,
            expected_size=spec.size_bytes,
        )
        if failure:
            failures.append(failure)
    return failures


def _check_hf_cache(config: PipelineConfig) -> list[str]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return []
    models = (
        (config.vision_language_model, config.vision_language_revision),
        (config.language_model, config.language_model_revision),
        (config.embedding_model, config.embedding_revision),
    )
    failures = []
    for repo_id, revision in models:
        try:
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                local_files_only=True,
            )
        except Exception as exc:
            failures.append(f"missing local model snapshot {repo_id}@{revision}: {exc}")
    return failures


def _format_failures(failures: Iterable[str]) -> str:
    return "CSI-VAD preflight failed:\n" + "\n".join(
        f"- {failure}" for failure in failures
    )


def run_preflight(
    config: PipelineConfig,
    *,
    check_weights: bool = True,
    check_model_cache: bool | None = None,
) -> None:
    failures = [*_check_python_and_packages(), *_check_cuda(config.dtype)]
    if check_weights:
        failures.extend(_check_weights(config))
    should_check_cache = (
        config.local_files_only if check_model_cache is None else check_model_cache
    )
    if should_check_cache:
        failures.extend(_check_hf_cache(config))
    if failures:
        raise PreflightError(_format_failures(failures))
