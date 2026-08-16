"""Baseline candidate retrieval for the AIC 2026 Query Engine.

This module intentionally contains only query-side orchestration. It consumes
DataStore.search_clip() and never accesses SQLite/FAISS implementation details.

The baseline retrieves frame-level candidates and aggregates them to video-level
hypotheses while retaining multiple alternatives for Top-k evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from data_layer.datastore import DataStore


class QueryEmbedder(Protocol):
    """Encode a normalized natural-language query into the corpus embedding space."""

    def encode(self, text: str) -> np.ndarray:  # pragma: no cover - protocol
        ...


@dataclass(frozen=True)
class RetrievalHit:
    video_id: str
    frame_id: int
    score: float
    faiss_id: int | None = None
    sources: tuple[str, ...] = ("clip",)


class ClipCandidateRetriever:
    """Retrieve and aggregate CLIP frame hits into ranked video candidates."""

    def __init__(
        self,
        datastore: DataStore,
        embedder: QueryEmbedder,
        *,
        frame_top_k: int = 200,
        video_top_k: int = 50,
    ) -> None:
        if frame_top_k <= 0:
            raise ValueError("frame_top_k must be > 0")
        if video_top_k <= 0:
            raise ValueError("video_top_k must be > 0")
        self.datastore = datastore
        self.embedder = embedder
        self.frame_top_k = frame_top_k
        self.video_top_k = video_top_k

    def retrieve(self, query_text: str) -> list[RetrievalHit]:
        text = query_text.strip()
        if not text:
            raise ValueError("query_text must not be empty")

        vector = np.asarray(self.embedder.encode(text), dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError("embedder must return a non-empty 1D vector")

        raw_hits = self.datastore.search_clip(vector, self.frame_top_k)
        frame_hits = [self._normalize_hit(item) for item in raw_hits]
        return self._aggregate_by_video(frame_hits)

    @staticmethod
    def _normalize_hit(item: dict[str, Any]) -> RetrievalHit:
        try:
            video_id = str(item["video_id"])
            frame_id = int(item["frame_id"])
            score = float(item["score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid retrieval hit: {item!r}") from exc

        faiss_id = item.get("faiss_id")
        if faiss_id is not None:
            faiss_id = int(faiss_id)
        return RetrievalHit(
            video_id=video_id,
            frame_id=frame_id,
            score=score,
            faiss_id=faiss_id,
        )

    def _aggregate_by_video(self, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        """Keep the strongest frame per video, then rank videos by that score.

        Keeping the best frame is deliberately conservative for the first
        baseline: it preserves an interpretable frame hypothesis and avoids
        over-counting videos that happen to have many visually similar frames.
        """
        best: dict[str, RetrievalHit] = {}
        for hit in hits:
            previous = best.get(hit.video_id)
            if previous is None or hit.score > previous.score:
                best[hit.video_id] = hit

        ranked = sorted(best.values(), key=lambda item: item.score, reverse=True)
        return ranked[: self.video_top_k]
