"""Model-pluggable visual semantic reranking.

The canonical retrieval path keeps the BTC-provided CLIP/FAISS index as the
high-recall candidate generator.  This module adds an optional second-stage
vision-language encoder.  SigLIP2 is the first supported backend because it
is explicitly intended for image-text retrieval and semantic understanding.
The adapter is lazy: importing the query engine does not download or load a
model, so CI and CPU-only baseline runs remain lightweight.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np

from .query_understanding import QuerySpec


class ImageTextScorer(Protocol):
    def score_images(self, images: Sequence[Any], text: str) -> np.ndarray:
        """Return one semantic image-text score per image."""
        ...


@dataclass(frozen=True)
class SemanticRerankConfig:
    enabled: bool = False
    model_id: str = "google/siglip2-base-patch16-256"
    candidate_limit: int = 50
    weight: float = 0.15

    def __post_init__(self) -> None:
        if self.candidate_limit <= 0:
            raise ValueError("candidate_limit must be > 0")
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError("weight must be in [0, 1]")


class SigLIP2ImageTextScorer:
    """Lazy Hugging Face Transformers adapter for SigLIP2."""

    def __init__(self, model_id: str = "google/siglip2-base-patch16-256", *, device: str | None = None) -> None:
        self.model_id = model_id
        self.device = device
        self._processor = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:  # pragma: no cover - optional ML dependency
            raise RuntimeError(
                "SigLIP2 requires the optional ML dependencies. Install the 'ml' extra."
            ) from exc

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModel.from_pretrained(self.model_id)
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(device)
        self._model.eval()
        self.device = device

    def score_images(self, images: Sequence[Any], text: str) -> np.ndarray:
        if not images:
            return np.empty((0,), dtype=np.float32)
        if not text.strip():
            raise ValueError("text must not be empty")
        self._load()

        import torch

        inputs = self._processor(
            text=[text] * len(images),
            images=list(images),
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items() if hasattr(value, "to")}
        with torch.inference_mode():
            outputs = self._model(**inputs)
            # SigLIP exposes pairwise image/text logits. Sigmoid converts them
            # to independent compatibility probabilities in [0, 1].
            scores = torch.sigmoid(outputs.logits_per_image).diagonal()
        return scores.detach().float().cpu().numpy()


def build_semantic_reranker(
    config: SemanticRerankConfig,
) -> ImageTextScorer | None:
    """Build the configured semantic backend without loading it eagerly."""
    if not config.enabled:
        return None
    return SigLIP2ImageTextScorer(model_id=config.model_id)


def semantic_text(spec: QuerySpec) -> str:
    """Use the full natural-language query for visual semantic alignment.

    Structured fields are auxiliary metadata; the model should see the full
    phrase so relations such as ``person riding motorcycle`` are preserved.
    """
    return spec.text.strip()
