import pytest

from csi_vad.aggregation import aggregate_labels, aggregate_scores
from csi_vad.schemas import BranchResult, build_final_result


def test_score_aggregation_formulas() -> None:
    scores = [0.2, 0.5, 0.8]

    result = aggregate_scores(scores)

    assert result["max"] == pytest.approx(0.8)
    assert result["mean"] == pytest.approx(0.5)
    assert result["noisy_or"] == pytest.approx(0.92)


def test_label_aggregation_thresholds() -> None:
    result = aggregate_labels([1, 0, 1])

    assert result == {"or": 1, "majority": 1, "and": 0}


def test_aggregation_rejects_incomplete_branch_sets() -> None:
    with pytest.raises(ValueError, match="exactly three"):
        aggregate_scores([0.1, 0.2])


def test_final_result_selects_requested_defaults() -> None:
    branches = {
        "environment": BranchResult(1, 0.6, "environment evidence"),
        "object": BranchResult(0, 0.2, "object evidence"),
        "time": BranchResult(1, 0.7, "temporal evidence"),
    }

    result = build_final_result(
        video={"name": "sample.mp4"},
        config={"num_frames": 32},
        branches=branches,
        score_aggregation="noisy_or",
        label_aggregation="or",
        timings={"total": 1.0},
    )

    assert result["prediction"]["score"] == pytest.approx(0.904)
    assert result["prediction"]["label"] == 1
    assert result["aggregations"]["scores"]["mean"] == pytest.approx(0.5)
    assert result["branches"]["object"]["explanation"] == "object evidence"
