"""FAISS adapter with explicit internal-id -> source-frame mapping."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class FAISSFrameStore:
    """Load and query a deterministic frame-level FAISS index."""

    def __init__(self, index_path: str | Path, mapping_path: str | Path):
        self.index_path = Path(index_path)
        self.mapping_path = Path(mapping_path)
        self.index = None
        self.mapping: list[dict[str, Any]] = []

    def load(self) -> None:
        if not self.index_path.is_file():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        if not self.mapping_path.is_file():
            raise FileNotFoundError(f"FAISS mapping not found: {self.mapping_path}")

        import faiss  # type: ignore

        self.index = faiss.read_index(str(self.index_path))
        payload = json.loads(self.mapping_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise TypeError("FAISS mapping must be a JSON list")
        self.mapping = payload
        if len(self.mapping) != self.index.ntotal:
            raise ValueError(
                f"FAISS mapping length {len(self.mapping)} != index size {self.index.ntotal}"
            )
        for internal_id, item in enumerate(self.mapping):
            if not isinstance(item, dict):
                raise TypeError(f"Mapping entry {internal_id} is not an object")
            if "video_id" not in item or "frame_id" not in item:
                raise ValueError(
                    f"Mapping entry {internal_id} must contain video_id and frame_id"
                )
            int(item["frame_id"])

    def search(self, vector: np.ndarray, top_k: int) -> list[dict[str, Any]]:
        if self.index is None:
            raise RuntimeError("FAISS index is not loaded")
        if top_k <= 0:
            return []

        query = np.asarray(vector, dtype=np.float32)
        if query.ndim == 1:
            query = query[None, :]
        if query.ndim != 2 or query.shape[0] != 1:
            raise ValueError("query vector must have shape (dimension,) or (1, dimension)")
        if query.shape[1] != self.index.d:
            raise ValueError(
                f"query dimension {query.shape[1]} != index dimension {self.index.d}"
            )
        if not np.isfinite(query).all():
            raise ValueError("query vector contains NaN or infinity")

        distances, ids = self.index.search(query, min(top_k, self.index.ntotal))
        results: list[dict[str, Any]] = []
        for distance, internal_id in zip(distances[0], ids[0]):
            if internal_id < 0:
                continue
            item = dict(self.mapping[int(internal_id)])
            item["score"] = float(distance)
            item["faiss_id"] = int(internal_id)
            results.append(item)
        return results
