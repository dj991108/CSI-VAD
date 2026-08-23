from scripts.verify_environment import version_matches


def test_cuda_wheel_local_suffix_matches_pinned_base_version() -> None:
    assert version_matches("2.6.0+cu126", "2.6.0")
    assert version_matches("0.21.0+cu126", "0.21.0")


def test_different_base_version_does_not_match() -> None:
    assert not version_matches("2.7.0+cu126", "2.6.0")
