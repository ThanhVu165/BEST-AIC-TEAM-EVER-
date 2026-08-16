from __future__ import annotations

from collections.abc import Iterable


def recall_at_k(
    ranked_video_ids: Iterable[str],
    relevant_video_ids: set[str],
    k: int,
) -> float:
    """Return binary video-level Recall@K for one query."""
    if k <= 0 or not relevant_video_ids:
        return 0.0
    return float(bool(set(ranked_video_ids) & relevant_video_ids and k > 0)) if any(
        video_id in relevant_video_ids for video_id in list(ranked_video_ids)[:k]
    ) else 0.0


def recall_at_ks(
    ranked_video_ids: Iterable[str],
    relevant_video_ids: set[str],
    ks: tuple[int, ...] = (1, 5, 20, 50, 100),
) -> dict[int, float]:
    ranked = list(ranked_video_ids)
    return {k: recall_at_k(ranked, relevant_video_ids, k) for k in ks}
