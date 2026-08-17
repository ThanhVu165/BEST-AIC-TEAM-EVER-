"""Inspect-able multimodal reranking and final candidate ordering."""
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
    sources: tuple[str, ...] = field(default_factory=tuple)

    @property
    def fused_score(self) -> float:
        """Default transparent fusion for the baseline reranking stage.

        Retrieval remains dominant. Additional evidence is additive and bounded;
        model-specific learned weights can later replace this function without
        changing the data contract.
        """
        return (
            0.80 * self.retrieval_score
            + 0.05 * self.object_score
            + 0.05 * self.metadata_score
            + 0.03 * self.ocr_score
            + 0.03 * self.asr_score
            + 0.02 * self.temporal_score
            + 0.02 * self.semantic_score
        )


def rerank_candidates(
    candidates: Iterable[RankingEvidence],
    *,
    limit: int = 100,
) -> list[RankingEvidence]:
    """Rerank candidates with deterministic tie-breaking and deduplication."""
    if limit <= 0:
        return []

    unique: dict[tuple[str, int], RankingEvidence] = {}
    for candidate in candidates:
        key = (candidate.video_id, candidate.frame_id)
        previous = unique.get(key)
        if previous is None or candidate.fused_score > previous.fused_score:
            unique[key] = candidate

    ranked = sorted(
        unique.values(),
        key=lambda item: (
            -item.fused_score,
            item.video_id,
            item.frame_id,
        ),
    )
    return ranked[:limit]


def diversify_candidates(
    candidates: Iterable[RankingEvidence],
    *,
    limit: int = 100,
    max_per_video: int | None = None,
) -> list[RankingEvidence]:
    """Create a final ranked pool while preventing one video from dominating it.

    The first pass preserves global score order. A later pass fills remaining
    slots with high-scoring candidates from underrepresented videos.
    """
    if limit <= 0:
        return []
    items = sorted(
        candidates,
        key=lambda item: (-item.fused_score, item.video_id, item.frame_id),
    )
    if max_per_video is None:
        return items[:limit]
    if max_per_video <= 0:
        return []

    selected: list[RankingEvidence] = []
    counts: dict[str, int] = {}
    deferred: list[RankingEvidence] = []

    for item in items:
        count = counts.get(item.video_id, 0)
        if count < max_per_video and len(selected) < limit:
            selected.append(item)
            counts[item.video_id] = count + 1
        else:
            deferred.append(item)

    if len(selected) < limit:
        for item in deferred:
            if len(selected) >= limit:
                break
            selected.append(item)

    return selected
