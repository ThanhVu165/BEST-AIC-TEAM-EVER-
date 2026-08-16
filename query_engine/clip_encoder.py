"""CLIP text encoder adapter for BTC ViT-B/32 frame features.

The adapter is lazy: importing the Query Engine does not require model weights.
At inference time it uses the Hugging Face OpenAI CLIP ViT-B/32 checkpoint and
L2-normalizes the text embedding so it can be compared with normalized image
features using inner-product/cosine retrieval.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class CLIPTextEncoder:
    """Encode text with OpenAI CLIP ViT-B/32 using a replaceable model adapter."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        *,
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None

    def _load(self) -> None:
        if self._model is not None:
            return

        try:
            import torch
            from transformers import CLIPModel, CLIPTokenizerFast
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
            # Recent transformers releases may return a ModelOutput from
            # get_text_features() instead of the projected tensor expected by
            # this adapter. Build the CLIP text embedding explicitly so the
            # output remains the same 512-D projected space as BTC's ViT-B/32
            # image features.
            text_outputs = self._model.text_model(**tokens)
            pooled = text_outputs.pooler_output
            features = self._model.text_projection(pooled)
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        return features[0].detach().cpu().numpy().astype(np.float32, copy=False)
