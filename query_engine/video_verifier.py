"""Optional video-window verification with InternVideo3.

This adapter is intentionally late-stage. It is not a replacement for the
BTC CLIP/FAISS candidate generator. InternVideo3 is used when a small set of
bounded source-video windows needs stronger temporal/action reasoning.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Protocol


class VideoVerifier(Protocol):
    def verify(self, video_path: str, query: str, *, fps: float = 2.0) -> float:
        """Return a normalized [0, 1] event-match score."""
        ...


@dataclass(frozen=True)
class VideoVerifierConfig:
    enabled: bool = False
    model_id: str = "yanziang/InternVideo3-8B-Instruct"
    candidate_limit: int = 3
    weight: float = 0.10
    fps: float = 2.0

    def __post_init__(self) -> None:
        if self.candidate_limit <= 0:
            raise ValueError("candidate_limit must be > 0")
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError("weight must be in [0, 1]")
        if self.fps <= 0:
            raise ValueError("fps must be > 0")


class InternVideo3Verifier:
    """Lazy Transformers adapter for InternVideo3-8B-Instruct.

    The adapter deliberately exposes only a normalized verification score to
    the retrieval engine. Prompting and output parsing are kept here so the
    ranking layer does not depend on a particular VLM implementation.
    """

    def __init__(self, model_id: str = "yanziang/InternVideo3-8B-Instruct") -> None:
        self.model_id = model_id
        self._processor = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "InternVideo3 requires the optional ML dependencies. Install the 'ml-video' extra."
            ) from exc
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="auto",
            trust_remote_code=True,
        )
        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )
        self._model.eval()

    @staticmethod
    def _parse_score(text: str) -> float:
        """Parse only an explicitly labelled normalized score.

        We intentionally do not take the last arbitrary number in model text:
        prompts and explanations can contain unrelated numbers, which would
        otherwise silently corrupt ranking.
        """
        match = re.search(
            r"(?:score|probability|confidence)\s*[:=]\s*(0(?:\.\d+)?|1(?:\.0+)?)",
            text,
            re.I,
        )
        if match is None:
            raise ValueError("video verifier output does not contain a labelled score")
        return max(0.0, min(1.0, float(match.group(1))))

    def verify(self, video_path: str, query: str, *, fps: float = 2.0) -> float:
        if not video_path:
            return 0.0
        if not query.strip():
            raise ValueError("query must not be empty")
        if fps <= 0:
            raise ValueError("fps must be > 0")
        self._load()
        import torch

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "fps": fps,
                    "min_pixels": 128 * 2 * 32 * 32,
                    "max_pixels": 256 * 2 * 32 * 32,
                },
                {
                    "type": "text",
                    "text": (
                        "Verify whether this video depicts the event in the query. "
                        "Focus on action, interaction, temporal evidence and relations. "
                        "Return exactly one normalized score in the form `score: <number>` "
                        "where the number is between 0 and 1.\n"
                        f"Query: {query}"
                    ),
                },
            ],
        }]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            fps=fps,
            return_tensors="pt",
        ).to(self._model.device)
        with torch.inference_mode():
            output = self._model.generate(**inputs, max_new_tokens=32, use_cache=True)
        generated_ids = [o[len(i):] for i, o in zip(inputs.input_ids, output)]
        text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return self._parse_score(text)


def build_video_verifier(config: VideoVerifierConfig) -> VideoVerifier | None:
    if not config.enabled:
        return None
    return InternVideo3Verifier(model_id=config.model_id)
