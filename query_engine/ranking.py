"""Inspectable multimodal reranking and final candidate ordering."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RankingEvidence:
    video_id: str
    frame_id: int
    retrieval_score: float
    object_score: float = 0.0
    metadata_score: float = 0.0
    ocr_score: float = 0.0
    asr_score: float = 0.0
    temporal_score: float = 0.0
    semantic_score: float = 0.0
    video_verification_score: float = 0.0
    semantic_weight: float = 0.02
    video_verification_weight: float = 0.0
    sources: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not 0.0 <= self.semantic_weight <= 1.0:
            raise ValueError("semantic_weight must be in [0, 1]")
        if not 0.0 <= self.video_verification_weight <= 1.0:
            raise ValueError("video_verification_weight must be in [0, 1]")
        if self.semantic_weight + self.video_verification_weight > 1.0:
            raise ValueError("semantic_weight + video_verification_weight must be <= 1")

    @property
    def fused_score(self) -> float:
        """Fuse independent evidence exactly once."""
        semantic_weight = self.semantic_weight
        video_weight = self.video_verification_weight
        base_weight = 1.0 - semantic_weight - video_weight
        canonical_base = (
            0.82 * self.retrieval_score
            + 0.05 * self.object_score
            + 0.04 * self.metadata_score
            + 0.02 * self.ocr_score
            + 0.02 * self.asr_score
            + 0.03 * self.temporal_score
        ) / 0.98
        return (
            base_weight * canonical_base
            + semantic_weight * self.semantic_score
            + video_weight * self.video_verification_score
        )


def rerank_candidates(candidates: Iterable[RankingEvidence], *, limit: int = 100) -> list[RankingEvidence]:
    if limit <= 0:
        return []
    unique: dict[tuple[str, int], RankingEvidence] = {}
    for candidate in candidates:
        key = (candidate.video_id, candidate.frame_id)
        previous = unique.get(key)
        if previous is None or candidate.fused_score > previous.fused_score:
            unique[key] = candidate
    return sorted(unique.values(), key=lambda item: (-item.fused_score, item.video_id, item.frame_id))[:limit]


def diversify_candidates(candidates: Iterable[RankingEvidence], *, limit: int = 100, max_per_video: int | None = None) -> list[RankingEvidence]:
    """Select top-ranked candidates while enforcing an optional per-video cap.

    The cap is a hard constraint. We do not append deferred candidates after
    the first pass because doing so would silently violate ``max_per_video``.
    """
    if limit <= 0:
        return []
    items = sorted(candidates, key=lambda item: (-item.fused_score, item.video_id, item.frame_id))
    if max_per_video is None:
        return items[:limit]
    if max_per_video <= 0:
        return []

    selected: list[RankingEvidence] = []
    counts: dict[str, int] = {}
    for item in items:
        if len(selected) >= limit:
            break
        count = counts.get(item.video_id, 0)
        if count >= max_per_video:
            continue
        selected.append(item)
        counts[item.video_id] = count + 1
    return selected
