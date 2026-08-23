from __future__ import annotations

import json
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from .schemas import BranchResult


class ModelOutputError(ValueError):
    """Raised when a model response violates the public output contract."""


def _strip_code_fence(text: str) -> str:
    value = text.strip()
    if not value.startswith("```"):
        return value
    first_newline = value.find("\n")
    if first_newline < 0:
        return value
    value = value[first_newline + 1 :]
    if value.rstrip().endswith("```"):
        value = value.rstrip()[:-3]
    return value.strip()


def _balanced_json_substring(text: str) -> str | None:
    start = text.find("{")
    while start >= 0:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
        start = text.find("{", start + 1)
    return None


def parse_json_object(raw: str) -> dict[str, Any]:
    text = _strip_code_fence(str(raw or "").replace("\u200b", " "))
    if not text:
        raise ModelOutputError("model returned an empty response")
    for candidate in (text, _balanced_json_substring(text)):
        if candidate is None:
            continue
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ModelOutputError("model response does not contain a valid JSON object")


def _normalize_score(value: Any) -> float:
    try:
        score = Decimal(str(value))
    except Exception as exc:
        raise ModelOutputError("score is not numeric") from exc
    if score < 0 or score > 1:
        raise ModelOutputError("score must be within [0, 1]")
    return float(score.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def parse_branch_response(raw: str) -> BranchResult:
    value = parse_json_object(raw)
    try:
        label = int(str(value["label"]).strip())
        explanation = str(value["explanation"]).strip()
        score = _normalize_score(value["score"])
    except KeyError as exc:
        raise ModelOutputError(f"missing required key: {exc.args[0]}") from exc
    except (TypeError, ValueError) as exc:
        raise ModelOutputError("label must be 0 or 1") from exc
    try:
        return BranchResult(label=label, score=score, explanation=explanation)
    except ValueError as exc:
        raise ModelOutputError(str(exc)) from exc
