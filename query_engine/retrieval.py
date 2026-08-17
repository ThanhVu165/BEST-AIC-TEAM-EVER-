"""Candidate retrieval and inspectable multimodal evidence collection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol
import re

import numpy as np

from data_layer.datastore import DataStore

from .query_understanding import QuerySpec


class QueryEmbedder(Protocol):
    def encode(self, text: str) -> np.ndarray:  # pragma: no cover - protocol
        ...


@dataclass(frozen=True)
class RetrievalHit:
    video_id: str
    frame_id: int
    score: float
    keyframe_n: int | None = None
    faiss_id: int | None = None
    sources: tuple[str, ...] = ("clip",)
    object_score: float = 0.0
    metadata_score: float = 0.0
    ocr_score: float = 0.0
    asr_score: float = 0.0
    retrieval_score: float | None = None

    @property
    def raw_score(self) -> float:
        return float(self.retrieval_score if self.retrieval_score is not None else self.score)


class ClipCandidateRetriever:
    """Generate CLIP candidates; auxiliary signals remain raw evidence."""

    def __init__(self, datastore: DataStore, embedder: QueryEmbedder, *, frame_top_k: int = 5000,
                 video_top_k: int = 100, object_weight: float = 0.05, metadata_weight: float = 0.03,
                 ocr_weight: float = 0.02, asr_weight: float = 0.02) -> None:
        if frame_top_k <= 0 or video_top_k <= 0:
            raise ValueError("frame_top_k and video_top_k must be > 0")
        weights = (object_weight, metadata_weight, ocr_weight, asr_weight)
        if any(not 0.0 <= weight < 1.0 for weight in weights):
            raise ValueError("auxiliary weights must be in [0, 1)")
        if sum(weights) >= 0.5:
            raise ValueError("auxiliary evidence weights must remain below 0.5 total")
        self.datastore = datastore
        self.embedder = embedder
        self.frame_top_k = frame_top_k
        self.video_top_k = video_top_k
        self.object_weight = object_weight
        self.metadata_weight = metadata_weight
        self.ocr_weight = ocr_weight
        self.asr_weight = asr_weight

    def retrieve(self, query: QuerySpec | str) -> list[RetrievalHit]:
        text = (query.text if isinstance(query, QuerySpec) else query).strip()
        if not text:
            raise ValueError("query_text must not be empty")
        vector = np.asarray(self.embedder.encode(text), dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError("embedder must return a non-empty 1D vector")
        raw_hits = self.datastore.search_clip(vector, self.frame_top_k)
        return self._collect_auxiliary_evidence([self._normalize_hit(item) for item in raw_hits], text)

    def retrieve_videos(self, query: QuerySpec | str) -> list[RetrievalHit]:
        frame_hits = self.retrieve(query)
        grouped: dict[str, list[RetrievalHit]] = {}
        for hit in frame_hits:
            grouped.setdefault(hit.video_id, []).append(hit)
        representatives: list[tuple[RetrievalHit, float]] = []
        for hits in grouped.values():
            ordered = sorted(hits, key=lambda item: (-item.raw_score, item.frame_id, item.keyframe_n or 0))
            best = ordered[0]
            support = float(np.mean([item.raw_score for item in ordered[:3]]))
            representatives.append((best, 0.97 * best.raw_score + 0.03 * support))
        representatives.sort(key=lambda item: (-item[1], item[0].video_id, item[0].frame_id, item[0].keyframe_n or 0))
        return [item[0] for item in representatives[: self.video_top_k]]

    def _collect_auxiliary_evidence(self, hits: list[RetrievalHit], query_text: str) -> list[RetrievalHit]:
        query_tokens = _tokens(query_text)
        get_objects = getattr(self.datastore, "get_objects", None)
        get_ocr = getattr(self.datastore, "get_ocr", None)
        get_metadata = getattr(self.datastore, "get_metadata", None)
        get_asr = getattr(self.datastore, "get_asr", None)
        enriched: list[RetrievalHit] = []
        for hit in hits:
            object_score = self._object_score(hit, query_tokens, get_objects)
            metadata_score = self._metadata_score(hit, query_tokens, get_metadata)
            ocr_score = self._ocr_score(hit, query_tokens, get_ocr)
            asr_score = self._asr_score(hit, query_tokens, get_asr)
            sources = list(hit.sources)
            for name, score in (("objects", object_score), ("metadata", metadata_score), ("ocr", ocr_score), ("asr", asr_score)):
                if score > 0.0:
                    sources.append(name)
            enriched.append(RetrievalHit(video_id=hit.video_id, frame_id=hit.frame_id, score=hit.raw_score,
                keyframe_n=hit.keyframe_n, faiss_id=hit.faiss_id, sources=tuple(dict.fromkeys(sources)),
                object_score=object_score, metadata_score=metadata_score, ocr_score=ocr_score,
                asr_score=asr_score, retrieval_score=hit.raw_score))
        return self._rank_frames(enriched)

    @staticmethod
    def _object_score(hit: RetrievalHit, query_tokens: set[str], getter: Any) -> float:
        if getter is None or hit.keyframe_n is None or not query_tokens:
            return 0.0
        record = getter(hit.video_id, hit.keyframe_n)
        if record is None:
            return 0.0
        best = 0.0
        for detection in record.objects:
            labels = _tokens(detection.label)
            if labels and labels.issubset(query_tokens):
                best = max(best, float(detection.confidence))
        return best

    @staticmethod
    def _metadata_score(hit: RetrievalHit, query_tokens: set[str], getter: Any) -> float:
        if getter is None or not query_tokens:
            return 0.0
        payload = getter(hit.video_id)
        if not payload:
            return 0.0
        text_parts: list[str] = []
        for key in ("title", "description", "keywords", "author", "channel_id"):
            value = payload.get(key)
            if isinstance(value, list):
                text_parts.extend(str(item) for item in value)
            elif value is not None:
                text_parts.append(str(value))
        available = _tokens(" ".join(text_parts))
        return len(query_tokens & available) / len(query_tokens) if available else 0.0

    @staticmethod
    def _ocr_score(hit: RetrievalHit, query_tokens: set[str], getter: Any) -> float:
        if getter is None or not query_tokens or hit.keyframe_n is None:
            return 0.0
        records = getter(hit.video_id, hit.keyframe_n)
        if not records:
            return 0.0
        available = _tokens(" ".join(record.text for record in records))
        return len(query_tokens & available) / len(query_tokens) if available else 0.0

    @staticmethod
    def _asr_score(hit: RetrievalHit, query_tokens: set[str], getter: Any) -> float:
        if getter is None or not query_tokens:
            return 0.0
        records = getter(hit.video_id)
        if not records:
            return 0.0
        available = _tokens(" ".join(record.text for record in records))
        return len(query_tokens & available) / len(query_tokens) if available else 0.0

    @staticmethod
    def _rank_frames(hits: list[RetrievalHit]) -> list[RetrievalHit]:
        unique: dict[tuple[str, int, int | None], RetrievalHit] = {}
        for hit in hits:
            key = (hit.video_id, hit.frame_id, hit.keyframe_n)
            previous = unique.get(key)
            if previous is None or hit.raw_score > previous.raw_score:
                unique[key] = hit
        return sorted(unique.values(), key=lambda item: (-item.raw_score, item.video_id, item.frame_id, item.keyframe_n or 0))

    @staticmethod
    def _normalize_hit(item: dict[str, Any]) -> RetrievalHit:
        try:
            video_id = str(item["video_id"])
            frame_id = int(item["frame_id"])
            score = float(item["score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid retrieval hit: {item!r}") from exc
        keyframe_n = item.get("keyframe_n")
        faiss_id = item.get("faiss_id")
        return RetrievalHit(video_id=video_id, frame_id=frame_id, score=score,
            keyframe_n=int(keyframe_n) if keyframe_n is not None else None,
            faiss_id=int(faiss_id) if faiss_id is not None else None, retrieval_score=score)


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[\w-]+", text.casefold()) if len(token) > 1}
