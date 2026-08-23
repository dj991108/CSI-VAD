from pathlib import Path

import pytest

from csi_vad.config import PipelineConfig
from csi_vad.preflight import PreflightError, run_preflight


def test_preflight_reports_all_failures(monkeypatch, tmp_path: Path) -> None:
    config = PipelineConfig(
        work_dir=tmp_path / "work",
        detector_weights=tmp_path / "missing-detector.pth",
        reid_weights=tmp_path / "missing-reid.pt",
        local_files_only=True,
    )
    monkeypatch.setattr(
        "csi_vad.preflight._check_python_and_packages",
        lambda: ["missing package: rfdetr"],
    )
    monkeypatch.setattr(
        "csi_vad.preflight._check_cuda", lambda dtype: ["CUDA unavailable"]
    )
    monkeypatch.setattr(
        "csi_vad.preflight._check_weights", lambda value: ["bad detector weight"]
    )
    monkeypatch.setattr(
        "csi_vad.preflight._check_hf_cache", lambda value: ["missing HF snapshot"]
    )

    with pytest.raises(PreflightError) as exc_info:
        run_preflight(config)

    message = str(exc_info.value)
    assert "missing package: rfdetr" in message
    assert "CUDA unavailable" in message
    assert "bad detector weight" in message
    assert "missing HF snapshot" in message


def test_preflight_can_skip_weights_and_model_cache(
    monkeypatch, tmp_path: Path
) -> None:
    config = PipelineConfig(work_dir=tmp_path / "work")
    monkeypatch.setattr("csi_vad.preflight._check_python_and_packages", lambda: [])
    monkeypatch.setattr("csi_vad.preflight._check_cuda", lambda dtype: [])

    def fail_if_called(*args, **kwargs):
        raise AssertionError("optional check should have been skipped")

    monkeypatch.setattr("csi_vad.preflight._check_weights", fail_if_called)
    monkeypatch.setattr("csi_vad.preflight._check_hf_cache", fail_if_called)

    run_preflight(config, check_weights=False, check_model_cache=False)
