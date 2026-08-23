from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) <= 1:
        return values
    width = min(int(window), len(values))
    kernel = np.ones(width, dtype=np.float32) / float(width)
    left = (width - 1) // 2
    right = width - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def detect_boundaries(
    embeddings: np.ndarray,
    *,
    threshold: float,
    smooth_window: int,
    debounce: int = 1,
    min_gap: int = 0,
) -> list[int]:
    matrix = np.asarray(embeddings, dtype=np.float32)
    if len(matrix) < 2:
        return []
    similarities = np.sum(matrix[:-1] * matrix[1:], axis=1)
    low = _moving_average(similarities, smooth_window) < threshold
    cuts: list[int] = []
    run_length = 0
    for index, is_low in enumerate(low):
        if is_low:
            run_length += 1
            if run_length == max(1, int(debounce)):
                cuts.append(index + 1)
        else:
            run_length = 0
    if min_gap > 0 and cuts:
        filtered = [cuts[0]]
        for cut in cuts[1:]:
            if cut - filtered[-1] > min_gap:
                filtered.append(cut)
        cuts = filtered
    return cuts


def merge_consecutive_captions(
    captions: list[str], embeddings: np.ndarray, *, threshold: float
) -> tuple[list[str], list[tuple[int, int]]]:
    if not captions:
        return [], []
    matrix = np.asarray(embeddings, dtype=np.float32)
    if len(matrix) != len(captions):
        raise ValueError("caption and embedding counts must match")
    merged = [captions[0]]
    ranges = [(0, 0)]
    representative = captions[0]
    for index in range(1, len(captions)):
        similarity = float(np.dot(matrix[index - 1], matrix[index]))
        normalized = " ".join(captions[index].lower().split())
        representative_normalized = " ".join(representative.lower().split())
        if normalized == representative_normalized or similarity >= threshold:
            ranges[-1] = (ranges[-1][0], index)
            if len(captions[index]) < len(representative):
                representative = captions[index]
                merged[-1] = representative
            continue
        representative = captions[index]
        merged.append(representative)
        ranges.append((index, index))
    return merged, ranges


def deduplicate_captions(
    captions: list[str], embeddings: np.ndarray, *, threshold: float
) -> list[str]:
    matrix = np.asarray(embeddings, dtype=np.float32)
    if len(matrix) != len(captions):
        raise ValueError("caption and embedding counts must match")
    kept_captions: list[str] = []
    kept_embeddings: list[np.ndarray] = []
    for caption, embedding in zip(captions, matrix):
        if kept_embeddings:
            similarities = np.dot(np.stack(kept_embeddings), embedding)
            if float(np.max(similarities)) >= threshold:
                continue
        kept_captions.append(caption)
        kept_embeddings.append(embedding)
    return kept_captions


def _rounded_uniform_indices(total: int, count: int) -> list[int]:
    if count == 1:
        return [0]
    chosen: list[int] = []
    for position in range(count):
        index = int(round(position * (total - 1) / (count - 1)))
        if index not in chosen:
            chosen.append(index)
    if len(chosen) < count:
        for index in range(total):
            if index not in chosen:
                chosen.append(index)
            if len(chosen) == count:
                break
        chosen.sort()
    return chosen


def reduce_segments(
    segments: Iterable[dict[str, Any]], *, max_segments: int
) -> list[dict[str, Any]]:
    values = [dict(segment) for segment in segments]
    if max_segments <= 0:
        raise ValueError("max_segments must be positive")
    if len(values) > max_segments:
        values = [
            values[index]
            for index in _rounded_uniform_indices(len(values), max_segments)
        ]
    for index, value in enumerate(values, start=1):
        value["segment_id"] = index
    return values


class TemporalContextBuilder:
    """Builds event-aware caption groups and summarizes them with the text model."""

    def __init__(self, manager: Any, config: Any):
        self.manager = manager
        self.config = config

    def caption_and_segment(
        self, frame_paths: list[Any], visual_recognizer: Any
    ) -> list[list[str]]:
        captions = [visual_recognizer.caption(path) for path in frame_paths]
        if not captions:
            raise RuntimeError("temporal captioning produced no captions")
        embedder = self.manager.load_embedder()
        embeddings = embedder.encode(
            captions,
            batch_size=self.config.embedding_batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        merged, _ = merge_consecutive_captions(
            captions, embeddings, threshold=self.config.temporal_merge_similarity
        )
        merged_embeddings = embedder.encode(
            merged,
            batch_size=self.config.embedding_batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        cuts = detect_boundaries(
            merged_embeddings,
            threshold=self.config.temporal_boundary_similarity,
            smooth_window=self.config.temporal_smooth_window,
            debounce=self.config.temporal_boundary_debounce,
            min_gap=self.config.temporal_min_boundary_gap,
        )
        boundaries = [0, *cuts, len(merged)]
        groups: list[list[str]] = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            group = merged[start:end]
            group_embeddings = merged_embeddings[start:end]
            if group:
                groups.append(
                    deduplicate_captions(
                        group,
                        group_embeddings,
                        threshold=self.config.temporal_dedup_similarity,
                    )
                )
        return groups

    def summarize(
        self, groups: list[list[str]], temporal_recognizer: Any
    ) -> dict[str, Any]:
        segments = [
            {"segment_id": index, "scene_state": temporal_recognizer.summarize(group)}
            for index, group in enumerate(groups, start=1)
        ]
        return {
            "scene_segments": reduce_segments(
                segments, max_segments=self.config.temporal_max_segments
            )
        }
