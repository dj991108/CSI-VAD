import numpy as np

from csi_vad.contexts.temporal import (
    deduplicate_captions,
    detect_boundaries,
    merge_consecutive_captions,
    reduce_segments,
)


def test_detect_boundaries_from_similarity_drop() -> None:
    embeddings = np.asarray(
        [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]],
        dtype=np.float32,
    )
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    assert detect_boundaries(embeddings, threshold=0.7, smooth_window=1) == [2]


def test_detect_boundaries_does_not_invent_edge_cuts() -> None:
    embeddings = np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (10, 1))

    assert detect_boundaries(embeddings, threshold=0.7, smooth_window=5) == []


def test_merge_consecutive_captions_tracks_source_ranges() -> None:
    captions = ["a person walks", "a person is walking", "a car arrives"]
    embeddings = np.asarray([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    merged, ranges = merge_consecutive_captions(captions, embeddings, threshold=0.92)

    assert merged == ["a person walks", "a car arrives"]
    assert ranges == [(0, 1), (2, 2)]


def test_deduplicate_captions_removes_semantic_duplicates() -> None:
    captions = ["one", "one again", "different"]
    embeddings = np.asarray([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    assert deduplicate_captions(captions, embeddings, threshold=0.95) == [
        "one",
        "different",
    ]


def test_reduce_segments_uniformly_and_renumbers() -> None:
    segments = [{"segment_id": i + 1, "scene_state": str(i)} for i in range(30)]

    reduced = reduce_segments(segments, max_segments=20)

    assert len(reduced) == 20
    assert [item["segment_id"] for item in reduced] == list(range(1, 21))
    assert reduced[0]["scene_state"] == "0"
    assert reduced[-1]["scene_state"] == "29"
