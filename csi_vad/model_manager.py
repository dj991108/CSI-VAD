from __future__ import annotations

import gc
from typing import Any

from .config import PipelineConfig


class ModelManager:
    """Lazily loads heavyweight backends and releases them by stage."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._backends: dict[str, tuple[Any, ...]] = {}

    def register(self, name: str, *objects: Any) -> tuple[Any, ...]:
        if not objects:
            raise ValueError("at least one backend object is required")
        self._backends[name] = tuple(objects)
        return self._backends[name]

    def has(self, name: str) -> bool:
        return name in self._backends

    def require_cuda(self) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is not installed; create the documented environment"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. The official CSI-VAD inference path requires an NVIDIA GPU."
            )

    def _torch_dtype(self) -> Any:
        import torch

        return torch.bfloat16 if self.config.dtype == "bf16" else torch.float16

    def load_vision_language(self) -> tuple[Any, Any]:
        if self.has("vision_language"):
            return self._backends["vision_language"]  # type: ignore[return-value]
        self.require_cuda()
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "transformers with Qwen2.5-VL support is not installed"
            ) from exc
        processor = AutoProcessor.from_pretrained(
            self.config.vision_language_model,
            revision=self.config.vision_language_revision,
            use_fast=True,
            local_files_only=self.config.local_files_only,
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config.vision_language_model,
            revision=self.config.vision_language_revision,
            torch_dtype=self._torch_dtype(),
            attn_implementation=self.config.attention_implementation,
            device_map="auto",
            local_files_only=self.config.local_files_only,
        ).eval()
        self.register("vision_language", model, processor)
        return model, processor

    def load_language(self) -> tuple[Any, Any]:
        if self.has("language"):
            return self._backends["language"]  # type: ignore[return-value]
        self.require_cuda()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers is not installed") from exc
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.language_model,
            revision=self.config.language_model_revision,
            use_fast=True,
            local_files_only=self.config.local_files_only,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.config.language_model,
            revision=self.config.language_model_revision,
            torch_dtype=self._torch_dtype(),
            device_map="auto",
            local_files_only=self.config.local_files_only,
        ).eval()
        self.register("language", model, tokenizer)
        return model, tokenizer

    def load_embedder(self) -> Any:
        if self.has("embedding"):
            return self._backends["embedding"][0]
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is not installed") from exc
        model = SentenceTransformer(
            self.config.embedding_model,
            revision=self.config.embedding_revision,
            device=self.config.embedding_device,
            local_files_only=self.config.local_files_only,
        )
        self.register("embedding", model)
        return model

    def release(self, name: str) -> None:
        self._backends.pop(name, None)
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def release_all(self) -> None:
        for name in list(self._backends):
            self.release(name)
