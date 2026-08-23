from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .aggregation import aggregate_labels, aggregate_scores


@dataclass(frozen=True)
class BranchResult:
    label: int
    score: float
    explanation: str

    def __post_init__(self) -> None:
        if self.label not in (0, 1):
            raise ValueError("label must be 0 or 1")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be within [0, 1]")
        if not self.explanation.strip():
            raise ValueError("explanation must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_final_result(
    *,
    video: Mapping[str, Any],
    config: Mapping[str, Any],
    branches: Mapping[str, BranchResult],
    score_aggregation: str,
    label_aggregation: str,
    timings: Mapping[str, float],
    cache_hits: list[str] | None = None,
) -> dict[str, Any]:
    expected = ("environment", "object", "time")
    if tuple(branches.keys()) != expected:
        raise ValueError(f"branches must be ordered as {expected}")

    scores = aggregate_scores(branch.score for branch in branches.values())
    labels = aggregate_labels(branch.label for branch in branches.values())
    if score_aggregation not in scores:
        raise ValueError(f"unknown score aggregation: {score_aggregation}")
    if label_aggregation not in labels:
        raise ValueError(f"unknown label aggregation: {label_aggregation}")

    return {
        "video": dict(video),
        "config": dict(config),
        "branches": {name: value.to_dict() for name, value in branches.items()},
        "aggregations": {"scores": scores, "labels": labels},
        "prediction": {
            "score": scores[score_aggregation],
            "score_aggregation": score_aggregation,
            "label": labels[label_aggregation],
            "label_aggregation": label_aggregation,
        },
        "timings_sec": dict(timings),
        "cache_hits": list(cache_hits or []),
    }
