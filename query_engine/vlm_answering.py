"""Configurable image-question answering adapter backed by Transformers.

This module deliberately does not choose a competition model. The AIC data and
queries should be used to benchmark candidate VLMs before selecting a default.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .answering import AnswerEvidence, AnswerResult


class TransformersImageAnswerExtractor:
    """Run a configurable Transformers vision-language model on one frame."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        max_new_tokens: int = 64,
        prompt_template: str = (
            "Answer the question using only the image. Question: {question}"
        ),
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be empty")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0")
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.prompt_template = prompt_template
        self._processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:  # pragma: no cover - optional ML runtime
            raise RuntimeError(
                "TransformersImageAnswerExtractor requires the optional 'ml' dependencies."
            ) from exc

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = AutoModelForImageTextToText.from_pretrained(self.model_name)
        self._model.to(device)
        self._model.eval()
        self.device = device

    def answer(self, evidence: AnswerEvidence) -> AnswerResult:
        frame_path = Path(evidence.frame_path)
        if not frame_path.is_file():
            return AnswerResult("", None, "evidence_unavailable")

        self._load()
        assert self._processor is not None
        assert self._model is not None
        assert self._torch is not None

        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - optional ML runtime
            raise RuntimeError("Pillow is required for VLM answer extraction") from exc

        image = Image.open(frame_path).convert("RGB")
        prompt = self.prompt_template.format(question=evidence.question)
        inputs = self._processor(images=image, text=prompt, return_tensors="pt")
        inputs = {
            name: value.to(self.device) if hasattr(value, "to") else value
            for name, value in inputs.items()
        }

        with self._torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        answer = self._processor.batch_decode(
            output_ids, skip_special_tokens=True
        )[0].strip()
        return AnswerResult(answer, None, "completed" if answer else "empty_answer")
