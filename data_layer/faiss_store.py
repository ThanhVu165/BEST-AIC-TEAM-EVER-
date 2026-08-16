"""FAISS adapter with an explicit internal-id -> frame mapping.

FAISS is optional at import time so the rest of the repository remains
installable before indexes are built.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class FAISSFrameStore:
    def __init__(self, index_path: str | Path, mapping_path: str | Path):
        self.index_path = Path(index_path)
        self.mapping_path = Path(mapping_path)
        self.index = None
        self.mapping: list[dict[str, Any]] = []

    def load(self) -> None:
        import faiss  # type: ignore

        self.index = faiss.read_index(str(self.index_path))
        self.mapping = json.loads(self.mapping_path.read_text(encoding="utf-8"))
        if len(self.mapping) != self.index.ntotal:
            raise ValueError(
                f"FAISS mapping length {len(self.mapping)} != index size {self.index.ntotal}"
            )

    def search(self, vector: np.ndarray, top_k: int) -> list[dict[str, Any]]:
        if self.index is None:
            raise RuntimeError("FAISS index is not loaded")
        query = np.asarray(vector, dtype=np.float32)
        if query.ndim == 1:
            query = query[None, :]
        distances, ids = self.index.search(query, top_k)
        results: list[dict[str, Any]] = []
        for distance, internal_id in zip(distances[0], ids[0]):
            if internal_id < 0:
                continue
            item = dict(self.mapping[internal_id])
            item["score"] = float(distance)
            item["faiss_id"] = int(internal_id)
            results.append(item)
        return results
