import hashlib
from pathlib import Path

from csi_vad.model_specs import file_sha256, validate_weight_file


def test_weight_validation_uses_exact_sha256(tmp_path: Path) -> None:
    path = tmp_path / "weight.bin"
    path.write_bytes(b"known weight content")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()

    assert file_sha256(path) == expected
    assert (
        validate_weight_file(path, expected_sha256=expected, expected_size=20) is None
    )

    path.write_bytes(b"tampered weight content")
    error = validate_weight_file(path, expected_sha256=expected, expected_size=23)

    assert error is not None
    assert "SHA-256 mismatch" in error
