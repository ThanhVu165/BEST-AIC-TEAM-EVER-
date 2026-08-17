"""CLIP ViT-B/32 adapter for text and source-frame embeddings."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np


class CLIPTextEncoder:
    """Encode text and images with one shared OpenAI CLIP ViT-B/32 model.

    The text ``encode`` API remains compatible with the existing retriever. The
    image methods are used only by the fine temporal stage, which lets the
    system compare dense source-video frames against the query without loading
    a second copy of CLIP.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        *,
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._tokenizer: Any | None = None
        self._image_processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None

    def _load(self) -> None:
        if self._model is not None:
            return

        try:
            import torch
            from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizerFast
        except ImportError as exc:  # pragma: no cover - depends on optional ML extra
            raise RuntimeError(
                "CLIPTextEncoder requires the optional 'ml' dependencies "
                "(torch and transformers)."
            ) from exc

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._torch = torch
        self._tokenizer = CLIPTokenizerFast.from_pretrained(self.model_name)
        self._image_processor = CLIPImageProcessor.from_pretrained(self.model_name)
        self._model = CLIPModel.from_pretrained(self.model_name)
        self._model.to(device)
        self._model.eval()
        self.device = device

    def encode(self, text: str) -> np.ndarray:
        text = text.strip()
        if not text:
            raise ValueError("text must not be empty")

        self._load()
        assert self._torch is not None
        assert self._tokenizer is not None
        assert self._model is not None

        tokens = self._tokenizer(
            [text],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokens = {name: value.to(self.device) for name, value in tokens.items()}

        with self._torch.inference_mode():
            text_outputs = self._model.text_model(**tokens)
            pooled = text_outputs.pooler_output
            features = self._model.text_projection(pooled)
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        return features[0].detach().cpu().numpy().astype(np.float32, copy=False)

    def encode_images(self, images: Sequence[Any], *, batch_size: int = 16) -> np.ndarray:
        """Encode RGB image arrays/PIL images into the same normalized CLIP space."""
        if not images:
            return np.empty((0, 0), dtype=np.float32)
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        self._load()
        assert self._torch is not None
        assert self._image_processor is not None
        assert self._model is not None

        chunks: list[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            batch = list(images[start : start + batch_size])
            inputs = self._image_processor(images=batch, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            with self._torch.inference_mode():
                features = self._model.vision_model(pixel_values=pixel_values)
                pooled = features.pooler_output
                projected = self._model.visual_projection(pooled)
                projected = projected / projected.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            chunks.append(projected.detach().cpu().numpy().astype(np.float32, copy=False))
        return np.concatenate(chunks, axis=0)

    def encode_image(self, image: Any) -> np.ndarray:
        """Encode one RGB image."""
        return self.encode_images([image], batch_size=1)[0]
