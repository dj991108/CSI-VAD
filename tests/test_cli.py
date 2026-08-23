from pathlib import Path

import cv2
import numpy as np

from csi_vad.cli import main


def _write_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (16, 16))
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
    writer.release()


def test_cli_dry_run_validates_video_without_loading_models(
    tmp_path: Path, capsys
) -> None:
    video = tmp_path / "sample.mp4"
    output = tmp_path / "result.json"
    _write_video(video)

    exit_code = main(
        [
            "--video",
            str(video),
            "--output",
            str(output),
            "--num-frames",
            "256",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert '"num_frames": 256' in captured.out
    assert '"frame_count": 3' in captured.out
    assert not output.exists()


def test_cli_runs_preflight_before_constructing_pipeline(
    monkeypatch, tmp_path: Path
) -> None:
    video = tmp_path / "sample.mp4"
    output = tmp_path / "result.json"
    _write_video(video)
    events: list[str] = []

    monkeypatch.setattr(
        "csi_vad.cli.run_preflight", lambda config: events.append("preflight")
    )

    class FakePipeline:
        def __init__(self, config) -> None:
            events.append("pipeline_init")

        def run(self, video_path: Path, output_path: Path) -> dict:
            events.append("pipeline_run")
            return {"prediction": {"label": 0, "score": 0.1}}

    monkeypatch.setattr("csi_vad.cli.Pipeline", FakePipeline)

    assert main(["--video", str(video), "--output", str(output)]) == 0
    assert events == ["preflight", "pipeline_init", "pipeline_run"]
