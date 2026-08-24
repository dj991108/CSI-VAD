from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_documents_minimal_runnable_command() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "python infer.py --video" in readme
    assert "--num-frames FRAME_BUDGET" in readme
    assert "python scripts/download_models.py" in readme
    assert "python scripts/verify_environment.py" in readme
    assert "https://arxiv.org/abs/2607.19077" in readme


def test_environment_pins_public_runtime_versions() -> None:
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")

    for pin in (
        "torch==2.6.0",
        "torchvision==0.21.0",
        "transformers==4.49.0",
        "qwen-vl-utils[decord]==0.0.8",
        "rfdetr==1.4.0.post0",
        "boxmot==16.0.9",
        "sentence-transformers==5.2.0",
    ):
        assert pin in requirements


def test_license_scope_and_third_party_notices_are_explicit() -> None:
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")

    assert license_text.startswith("MIT License")
    assert "BoxMOT" in notices and "AGPL-3.0" in notices
    assert "StrongSORT" in notices and "GPL-3.0" in notices
    assert "RF-DETR" in notices and "Apache-2.0" in notices
    assert "Platform Model License 1.0" in notices
    assert "not covered by the CSI-VAD MIT License" in notices


def test_project_page_is_not_vendored_in_code_repository() -> None:
    assert not (ROOT / "docs").exists()


def test_generated_and_private_artifacts_are_ignored() -> None:
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

    for pattern in ("__pycache__/", ".csi_vad_work/", "weights/", "outputs/"):
        assert pattern in ignore


def test_public_text_files_do_not_expose_private_server_paths() -> None:
    private_markers = ("/home1/", "/home/irteam/", "/data/", "/gpfs/", "qc25501")
    public_files = [
        ROOT / "README.md",
        ROOT / "environment.yml",
        ROOT / "requirements.txt",
    ]

    for path in public_files:
        text = path.read_text(encoding="utf-8")
        assert not any(marker in text for marker in private_markers), path
