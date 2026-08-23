from pathlib import Path

from csi_vad.config import PipelineConfig
from csi_vad.model_manager import ModelManager


def test_release_removes_registered_backend(tmp_path: Path) -> None:
    manager = ModelManager(PipelineConfig(work_dir=tmp_path))
    first = object()
    second = object()
    manager.register("fake", first, second)

    assert manager.has("fake")

    manager.release("fake")

    assert not manager.has("fake")


def test_release_all_clears_multiple_backends(tmp_path: Path) -> None:
    manager = ModelManager(PipelineConfig(work_dir=tmp_path))
    manager.register("one", object())
    manager.register("two", object())

    manager.release_all()

    assert not manager.has("one")
    assert not manager.has("two")
