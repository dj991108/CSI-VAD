from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ..config import PipelineConfig
from ..model_manager import ModelManager
from ..parsing import ModelOutputError, parse_json_object
from ..prompts import (
    ENVIRONMENT_CONTEXT_SYSTEM_PROMPT,
    ENVIRONMENT_CONTEXT_USER_PROMPT,
)
from ..recognition import _file_uri, generate_vision


class EnvironmentContextBuilder:
    REQUIRED_KEYS = {"place", "daytime", "environment_type", "typical_situation"}

    def __init__(self, manager: ModelManager, config: PipelineConfig):
        self.manager = manager
        self.config = config

    def build(self, keyframes: Sequence[Path]) -> dict[str, Any]:
        if len(keyframes) != 3:
            raise ValueError("environment context requires exactly three keyframes")
        model, processor = self.manager.load_vision_language()
        content: list[dict[str, Any]] = [
            {
                "type": "image",
                "image": _file_uri(path),
                "max_pixels": self.config.max_pixels,
            }
            for path in keyframes
        ]
        content.append({"type": "text", "text": ENVIRONMENT_CONTEXT_USER_PROMPT})
        raw = generate_vision(
            model,
            processor,
            system_prompt=ENVIRONMENT_CONTEXT_SYSTEM_PROMPT,
            content=content,
            max_new_tokens=self.config.max_new_tokens,
        )
        context = parse_json_object(raw)
        missing = self.REQUIRED_KEYS - context.keys()
        if missing:
            raise ModelOutputError(
                "environment context is missing required keys: "
                + ", ".join(sorted(missing))
            )
        return context
