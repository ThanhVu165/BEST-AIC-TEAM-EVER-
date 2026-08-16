"""Baseline candidate retrieval for the AIC 2026 Query Engine.

This module contains query-side orchestration only. It consumes the shared
``DataStore`` interface and never accesses SQLite/FAISS implementation details.

Two retrieval views are intentionally kept separate:

* frame retrieval is needed by KIS because a correct video can contain many
  visually similar frames and the first frame hit is not necessarily inside
  the ground-truth event interval;
* video aggregation is needed for QA/TRAKE candidate generation.

This separation avoids throwing away useful frame hypotheses too early.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from data_layer.datastore import DataStore


class QueryEmbedder(Protocol):
    """Encode natural-language text into the corpus embedding space."""

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
    """Retrieve CLIP frame hits and expose frame/video ranked hypotheses."""

    def __init__(
        self,
        datastore: DataStore,
        embedder: QueryEmbedder,
        *,
        frame_top_k: int = 200,
        video_top_k: int = 50,
        max_frames_per_video: int = 3,
    ) -> None:
        if frame_top_k <= 0:
            raise ValueError("frame_top_k must be > 0")
        if video_top_k <= 0:
            raise ValueError("video_top_k must be > 0")
        if max_frames_per_video <= 0:
            raise ValueError("max_frames_per_video must be > 0")
        self.datastore = datastore
        self.embedder = embedder
        self.frame_top_k = frame_top_k
        self.video_top_k = video_top_k
        self.max_frames_per_video = max_frames_per_video

    def retrieve(self, query_text: str) -> list[RetrievalHit]:
        """Return globally ranked frame hypotheses.

        Exact duplicate ``(video_id, frame_id)`` hits are removed, but multiple
        frames from the same video are deliberately retained for KIS.
        """
        text = query_text.strip()
        if not text:
            raise ValueError("query_text must not be empty")

        vector = np.asarray(self.embedder.encode(text), dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError("embedder must return a non-empty 1D vector")

        raw_hits = self.datastore.search_clip(vector, self.frame_top_k)
        frame_hits = [self._normalize_hit(item) for item in raw_hits]
        return self._rank_frames(frame_hits)

    def retrieve_videos(self, query_text: str) -> list[RetrievalHit]:
        """Aggregate frame evidence into one strongest hypothesis per video."""
        frame_hits = self.retrieve(query_text)
        best: dict[str, RetrievalHit] = {}
        for hit in frame_hits:
            previous = best.get(hit.video_id)
            if previous is None or hit.score > previous.score:
                best[hit.video_id] = hit

        ranked = sorted(best.values(), key=lambda item: item.score, reverse=True)
        return ranked[: self.video_top_k]

    def _rank_frames(self, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        unique: dict[tuple[str, int], RetrievalHit] = {}
        for hit in hits:
            key = (hit.video_id, hit.frame_id)
            previous = unique.get(key)
            if previous is None or hit.score > previous.score:
                unique[key] = hit

        per_video: dict[str, int] = {}
        ranked: list[RetrievalHit] = []
        for hit in sorted(unique.values(), key=lambda item: item.score, reverse=True):
            count = per_video.get(hit.video_id, 0)
            if count >= self.max_frames_per_video:
                continue
            ranked.append(hit)
            per_video[hit.video_id] = count + 1
        return ranked

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
