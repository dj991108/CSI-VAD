from __future__ import annotations

from math import prod
from typing import Iterable


def _three(values: Iterable[float | int], name: str) -> list[float | int]:
    items = list(values)
    if len(items) != 3:
        raise ValueError(f"{name} aggregation requires exactly three branch values")
    return items


def aggregate_scores(scores: Iterable[float]) -> dict[str, float]:
    values = [float(value) for value in _three(scores, "score")]
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("branch scores must be within [0, 1]")
    return {
        "max": max(values),
        "mean": sum(values) / 3.0,
        "noisy_or": 1.0 - prod(1.0 - value for value in values),
    }


def aggregate_labels(labels: Iterable[int]) -> dict[str, int]:
    values = [int(value) for value in _three(labels, "label")]
    if any(value not in (0, 1) for value in values):
        raise ValueError("branch labels must be either 0 or 1")
    positives = sum(values)
    return {
        "or": int(positives >= 1),
        "majority": int(positives >= 2),
        "and": int(positives >= 3),
    }
