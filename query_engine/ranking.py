from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class RankingEvidence:
    video_id: str
    frame_id: int
    retrieval_score: float
    object_score: float = 0.0
    metadata_score: float = 0.0

    @property
    def fused_score(self) -> float:
        # Keep the baseline interpretable. Auxiliary signals are optional and
        # cannot overwhelm the primary cross-modal retrieval evidence.
        return (
            0.70 * self.retrieval_score
            + 0.20 * self.object_score
            + 0.10 * self.metadata_score
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
