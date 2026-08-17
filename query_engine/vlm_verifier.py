"""Optional VLM verification for difficult action/relation candidates.

This is deliberately a second-stage verifier, not a corpus-wide retriever.  A
strong VLM is useful for distinctions such as ``riding`` vs ``standing next
to`` or ``repairing`` vs ``using`` that object presence cannot resolve.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Protocol


class VisualVerifier(Protocol):
    def verify(self, image: Any, query: str) -> float:
        """Return a normalized [0, 1] semantic-match score."""
        ...


@dataclass(frozen=True)
class VLMVerifierConfig:
    enabled: bool = False
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    candidate_limit: int = 5
    weight: float = 0.10

    def __post_init__(self) -> None:
        if self.candidate_limit <= 0:
            raise ValueError("candidate_limit must be > 0")
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError("weight must be in [0, 1]")


class Qwen25VLVerifier:
    """Lazy Transformers adapter around Qwen2.5-VL-Instruct.

    The model is only loaded when ``verify`` is first called.  The prompt asks
    for a JSON probability and a short rationale, but only the probability is
    consumed by the ranking pipeline.
    """

    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct", *, device: str | None = None) -> None:
        self.model_id = model_id
        self.device = device
        self._processor = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:  # pragma: no cover - optional ML dependency
            raise RuntimeError(
                "Qwen2.5-VL requires the optional ML dependencies. Install the 'ml' extra."
            ) from exc

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map="auto",
        )
        self._model.eval()
        self.device = str(next(self._model.parameters()).device)

    @staticmethod
    def _parse_score(text: str) -> float:
        # Prefer a JSON object, then fall back to a bounded decimal.
        try:
            match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
            if match:
                payload = json.loads(match.group(0))
                value = float(payload.get("score", 0.0))
                return max(0.0, min(1.0, value))
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
        numbers = re.findall(r"(?:0(?:\.\d+)?|1(?:\.0+)?)", text)
        return float(numbers[-1]) if numbers else 0.0

    def verify(self, image: Any, query: str) -> float:
        if image is None:
            return 0.0
        if not query.strip():
            raise ValueError("query must not be empty")
        self._load()
        import torch

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": (
                            "Judge whether this image depicts the event described by the query. "
                            "Focus on actions and relations, not just object presence. "
                            "Return JSON only: {\"score\": number between 0 and 1, "
                            "\"reason\": \"short reason\"}.\nQuery: " + query
                        ),
                    },
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text], images=[image], padding=True, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items() if hasattr(v, "to")}
        with torch.inference_mode():
            generated = self._model.generate(**inputs, max_new_tokens=96)
        prompt_len = inputs["input_ids"].shape[1]
        decoded = self._processor.batch_decode(
            generated[:, prompt_len:], skip_special_tokens=True
        )[0]
        return self._parse_score(decoded)


def build_vlm_verifier(config: VLMVerifierConfig) -> VisualVerifier | None:
    if not config.enabled:
        return None
    return Qwen25VLVerifier(model_id=config.model_id)
