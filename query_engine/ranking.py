from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class RankingEvidence:
    video_id: str
    frame_id: int
    retrieval_score: float
    object_score: float = 0.0
    metadata_score: float = 0.0

    @property
    def fused_score(self) -> float:
        """Fuse optional auxiliary evidence without overriding retrieval."""
        return (
            0.92 * self.retrieval_score
            + 0.06 * self.object_score
            + 0.02 * self.metadata_score
        )


def rerank_candidates(
    candidates: Iterable[RankingEvidence],
    *,
    limit: int = 100,
) -> list[RankingEvidence]:
    """Deterministically rerank candidates using inspectable score components."""
    if limit <= 0:
        return []
    return sorted(
        candidates,
        key=lambda item: (
            -item.fused_score,
            item.video_id,
            item.frame_id,
        ),
    )[:limit]
