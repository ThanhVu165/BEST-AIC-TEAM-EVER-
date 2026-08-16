"""Task-agnostic retrieval metrics for offline candidate evaluation.

These metrics intentionally measure only candidate retrieval recall. They are
not presented as the official AIC scoring implementation because the official
query/ground-truth format is not yet available in the repository.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any


def recall_at_k(
    ranked_candidates: Sequence[Any],
    relevant_video_ids: Iterable[str],
    k: int,
) -> float:
    """Return 1.0 when a relevant video appears in the first ``k`` candidates."""
    if k <= 0:
        raise ValueError("k must be > 0")
    relevant = {str(video_id) for video_id in relevant_video_ids}
    if not relevant:
        raise ValueError("relevant_video_ids must not be empty")

    for candidate in ranked_candidates[:k]:
        video_id = _video_id(candidate)
        if video_id in relevant:
            return 1.0
    return 0.0


def recall_curve(
    ranked_candidates: Sequence[Any],
    relevant_video_ids: Iterable[str],
    ks: Sequence[int] = (1, 5, 20, 50, 100),
) -> dict[int, float]:
    """Compute a Recall@K curve for the requested cutoffs."""
    return {
        int(k): recall_at_k(ranked_candidates, relevant_video_ids, int(k))
        for k in ks
    }


def _video_id(candidate: Any) -> str:
    if isinstance(candidate, dict):
        try:
            return str(candidate["video_id"])
        except KeyError as exc:
            raise ValueError(f"Candidate has no video_id: {candidate!r}") from exc

    try:
        return str(candidate.video_id)
    except AttributeError as exc:
        raise ValueError(f"Candidate has no video_id: {candidate!r}") from exc
