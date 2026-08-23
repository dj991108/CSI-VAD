import pytest

from csi_vad.parsing import ModelOutputError, parse_branch_response, parse_json_object


def test_parse_direct_branch_json() -> None:
    result = parse_branch_response(
        '{"label":"1","score":"0.8","explanation":"visible fighting"}'
    )

    assert result.label == 1
    assert result.score == pytest.approx(0.8)
    assert result.explanation == "visible fighting"


def test_parse_fenced_json_and_round_score_to_prompt_grid() -> None:
    result = parse_branch_response(
        '```json\n{"label":0,"score":0.24,"explanation":"ordinary activity"}\n```'
    )

    assert result.label == 0
    assert result.score == pytest.approx(0.2)


def test_parse_json_object_from_surrounding_text() -> None:
    result = parse_json_object('answer: {"place":"road","daytime":"day"} end')

    assert result == {"place": "road", "daytime": "day"}


@pytest.mark.parametrize(
    "raw",
    [
        "",
        '{"label":2,"score":0.5,"explanation":"invalid label"}',
        '{"label":1,"explanation":"missing score"}',
    ],
)
def test_invalid_branch_output_raises(raw: str) -> None:
    with pytest.raises(ModelOutputError):
        parse_branch_response(raw)
