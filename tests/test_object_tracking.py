from types import SimpleNamespace

import numpy as np

from csi_vad.contexts.object_tracking import (
    RFDETR_COCO_CATEGORIES,
    build_interest_class_ids,
    normalize_detections,
)


def test_interest_class_names_map_to_stable_coco_ids() -> None:
    assert len(RFDETR_COCO_CATEGORIES) == 80
    assert build_interest_class_ids(["person", "car", "knife", "cell phone"]) == {
        1,
        3,
        49,
        77,
    }


def test_normalize_detections_preserves_sparse_one_based_ids() -> None:
    detections = SimpleNamespace(
        class_id=np.asarray([1, 4, 49, 90]),
        confidence=np.asarray([0.9, 0.8, 0.7, 0.6]),
        xyxy=np.asarray(
            [[1, 2, 5, 6], [2, 3, 4, 5], [3, 4, 6, 8], [1, 1, 2, 2]],
            dtype=np.float32,
        ),
    )

    result = normalize_detections(
        detections,
        original_size=(20, 40),
        inference_size=(10, 20),
        interest_class_ids={1, 49},
        topk=None,
    )

    np.testing.assert_allclose(
        result,
        [[2, 4, 10, 12, 0.9, 1], [6, 8, 12, 16, 0.7, 49]],
    )


def test_normalize_empty_detection_result() -> None:
    assert normalize_detections(
        None,
        original_size=(20, 40),
        inference_size=(20, 40),
        interest_class_ids={1},
        topk=None,
    ).shape == (0, 6)
