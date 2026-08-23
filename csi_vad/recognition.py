from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .config import PipelineConfig
from .model_manager import ModelManager
from .parsing import parse_branch_response
from .prompts import (
    CAPTION_PROMPT,
    DECISION_USER_PROMPT,
    ENVIRONMENT_RECOGNITION_SYSTEM_PROMPT,
    OBJECT_RECOGNITION_SYSTEM_PROMPT,
    SCENE_SUMMARY_SYSTEM_PROMPT,
    TEMPORAL_RECOGNITION_SYSTEM_PROMPT,
    branch_user_prompt,
)
from .schemas import BranchResult


def _file_uri(path: str | Path) -> str:
    return Path(path).expanduser().resolve().as_uri()


def _input_device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return "cuda"


def generate_vision(
    model: Any,
    processor: Any,
    *,
    system_prompt: str | None,
    content: list[dict[str, Any]],
    max_new_tokens: int,
) -> str:
    import torch

    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:
        raise RuntimeError("qwen-vl-utils is not installed") from exc
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(_input_device(model))
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    trimmed = [
        output[len(source) :] for source, output in zip(inputs.input_ids, generated)
    ]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


def generate_text(
    model: Any,
    tokenizer: Any,
    *,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int,
) -> str:
    import torch

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(_input_device(model)) for key, value in inputs.items()}
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    continuation = output[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(continuation, skip_special_tokens=True).strip()


class VisualRecognizer:
    def __init__(self, manager: ModelManager, config: PipelineConfig):
        self.manager = manager
        self.config = config

    def _recognize(
        self, frame_paths: Sequence[Path], *, system_prompt: str, user_prompt: str
    ) -> BranchResult:
        model, processor = self.manager.load_vision_language()
        video_item = {
            "type": "video",
            "video": [_file_uri(path) for path in frame_paths],
            "fps": 1.0,
            "max_pixels": self.config.max_pixels,
        }
        raw = generate_vision(
            model,
            processor,
            system_prompt=system_prompt,
            content=[video_item, {"type": "text", "text": user_prompt}],
            max_new_tokens=self.config.max_new_tokens,
        )
        return parse_branch_response(raw)

    def environment(
        self, frame_paths: Sequence[Path], environment_context: dict[str, Any]
    ) -> BranchResult:
        return self._recognize(
            frame_paths,
            system_prompt=ENVIRONMENT_RECOGNITION_SYSTEM_PROMPT,
            user_prompt=branch_user_prompt("Environment_Context", environment_context),
        )

    def object(self, overlay_paths: Sequence[Path]) -> BranchResult:
        return self._recognize(
            overlay_paths,
            system_prompt=OBJECT_RECOGNITION_SYSTEM_PROMPT,
            user_prompt=DECISION_USER_PROMPT,
        )

    def caption(self, image_path: Path) -> str:
        model, processor = self.manager.load_vision_language()
        return generate_vision(
            model,
            processor,
            system_prompt=None,
            content=[
                {
                    "type": "image",
                    "image": _file_uri(image_path),
                    "max_pixels": self.config.max_pixels,
                },
                {"type": "text", "text": CAPTION_PROMPT},
            ],
            max_new_tokens=self.config.caption_max_new_tokens,
        )


class TemporalRecognizer:
    def __init__(self, manager: ModelManager, config: PipelineConfig):
        self.manager = manager
        self.config = config

    def summarize(self, captions: Sequence[str]) -> str:
        model, tokenizer = self.manager.load_language()
        return generate_text(
            model,
            tokenizer,
            system_prompt=SCENE_SUMMARY_SYSTEM_PROMPT,
            user_prompt=(
                "Segment frame captions (newline-separated):\n"
                + "\n".join(captions)
                + "\n\nWrite a paragraph describing this segment."
            ),
            max_new_tokens=self.config.max_new_tokens,
        )

    def recognize(self, temporal_context: dict[str, Any]) -> BranchResult:
        model, tokenizer = self.manager.load_language()
        raw = generate_text(
            model,
            tokenizer,
            system_prompt=TEMPORAL_RECOGNITION_SYSTEM_PROMPT,
            user_prompt=branch_user_prompt("Temporal_Event_Context", temporal_context),
            max_new_tokens=self.config.max_new_tokens,
        )
        return parse_branch_response(raw)
