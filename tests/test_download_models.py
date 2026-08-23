import hashlib
from pathlib import Path

from csi_vad.model_specs import WeightSpec
from scripts.download_models import download_url


def test_download_url_replaces_invalid_existing_file(
    monkeypatch, tmp_path: Path
) -> None:
    payload = b"verified weight"
    spec = WeightSpec(
        filename="weight.bin",
        url="https://example.invalid/weight.bin",
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    destination = tmp_path / spec.filename
    destination.write_bytes(b"wrong")

    def fake_retrieve(url: str, path: Path) -> None:
        assert url == spec.url
        Path(path).write_bytes(payload)

    monkeypatch.setattr(
        "scripts.download_models.urllib.request.urlretrieve", fake_retrieve
    )

    download_url(spec, destination)

    assert destination.read_bytes() == payload


def test_download_url_rejects_checksum_mismatch(monkeypatch, tmp_path: Path) -> None:
    payload = b"verified weight"
    spec = WeightSpec(
        filename="weight.bin",
        url="https://example.invalid/weight.bin",
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    destination = tmp_path / spec.filename

    def fake_retrieve(url: str, path: Path) -> None:
        Path(path).write_bytes(b"tampered weight")

    monkeypatch.setattr(
        "scripts.download_models.urllib.request.urlretrieve", fake_retrieve
    )

    try:
        download_url(spec, destination)
    except RuntimeError as exc:
        assert "weight size mismatch" in str(exc) or "SHA-256 mismatch" in str(exc)
    else:
        raise AssertionError("invalid download was accepted")

    assert not destination.exists()
