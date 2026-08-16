"""Candidate retrieval and optional auxiliary object-evidence reranking."""
from __future__ import annotations

import re
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
    object_score: float = 0.0
    retrieval_score: float | None = None


class ClipCandidateRetriever:
    """Retrieve frame hypotheses and optionally fuse object evidence."""

    def __init__(
        self,
        datastore: DataStore,
        embedder: QueryEmbedder,
        *,
        frame_top_k: int = 200,
        video_top_k: int = 50,
        object_weight: float = 0.10,
    ) -> None:
        if frame_top_k <= 0:
            raise ValueError("frame_top_k must be > 0")
        if video_top_k <= 0:
            raise ValueError("video_top_k must be > 0")
        if not 0.0 <= object_weight < 1.0:
            raise ValueError("object_weight must be in [0, 1)")
        self.datastore = datastore
        self.embedder = embedder
        self.frame_top_k = frame_top_k
        self.video_top_k = video_top_k
        self.object_weight = object_weight

    def retrieve(self, query_text: str) -> list[RetrievalHit]:
        """Return frame hypotheses without collapsing multiple frames per video."""
        text = query_text.strip()
        if not text:
            raise ValueError("query_text must not be empty")

        vector = np.asarray(self.embedder.encode(text), dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError("embedder must return a non-empty 1D vector")
        raw_hits = self.datastore.search_clip(vector, self.frame_top_k)
        frame_hits = [self._normalize_hit(item) for item in raw_hits]
        return self._rerank_with_objects(frame_hits, text)

    def retrieve_videos(self, query_text: str) -> list[RetrievalHit]:
        """Aggregate frame evidence into one strongest hypothesis per video."""
        frame_hits = self.retrieve(query_text)
        best: dict[str, RetrievalHit] = {}
        for hit in frame_hits:
            previous = best.get(hit.video_id)
            if previous is None or hit.score > previous.score:
                best[hit.video_id] = hit

        ranked = sorted(
            best.values(),
            key=lambda item: (-item.score, item.video_id, item.frame_id),
        )
        return ranked[: self.video_top_k]

    def _rerank_with_objects(
        self,
        hits: list[RetrievalHit],
        query_text: str,
    ) -> list[RetrievalHit]:
        query_tokens = _tokens(query_text)
        reranked: list[RetrievalHit] = []
        getter = getattr(self.datastore, "get_objects", None)
        for hit in hits:
            object_score = 0.0
            record = getter(hit.video_id, hit.frame_id) if getter is not None else None
            if record is not None and query_tokens:
                for detection in record.objects:
                    label_tokens = _tokens(detection.label)
                    if label_tokens and label_tokens.issubset(query_tokens):
                        object_score = max(object_score, float(detection.confidence))

            retrieval_score = hit.retrieval_score
            if retrieval_score is None:
                retrieval_score = hit.score
            fused = (
                (1.0 - self.object_weight) * retrieval_score
                + self.object_weight * object_score
            )
            sources = list(hit.sources)
            if object_score > 0.0:
                sources.append("objects")
            reranked.append(
                RetrievalHit(
                    video_id=hit.video_id,
                    frame_id=hit.frame_id,
                    score=fused,
                    faiss_id=hit.faiss_id,
                    sources=tuple(sources),
                    object_score=object_score,
                    retrieval_score=retrieval_score,
                )
            )
        return self._rank_frames(reranked)

    @staticmethod
    def _rank_frames(hits: list[RetrievalHit]) -> list[RetrievalHit]:
        unique: dict[tuple[str, int], RetrievalHit] = {}
        for hit in hits:
            key = (hit.video_id, hit.frame_id)
            previous = unique.get(key)
            if previous is None or hit.score > previous.score:
                unique[key] = hit

        return sorted(
            unique.values(),
            key=lambda item: (-item.score, item.video_id, item.frame_id),
        )

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
            retrieval_score=score,
        )


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[\w-]+", text.casefold())
        if len(token) > 1
    }
