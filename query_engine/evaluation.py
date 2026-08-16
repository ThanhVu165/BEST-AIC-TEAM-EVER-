from __future__ import annotations

from typing import Any, Iterable, Sequence


DEFAULT_KS = (1, 5, 20, 50, 100)


def recall_at_k(
    ranked_candidates: Sequence[Any],
    relevant_video_ids: Iterable[str],
    k: int,
) -> float:
    """Return binary video-level Recall@K for one query."""
    if k <= 0:
        raise ValueError("k must be > 0")
    relevant = {str(video_id) for video_id in relevant_video_ids}
    if not relevant:
        raise ValueError("relevant_video_ids must not be empty")

    return float(
        any(_video_id(candidate) in relevant for candidate in ranked_candidates[:k])
    )


def recall_at_ks(
    ranked_candidates: Sequence[Any],
    relevant_video_ids: Iterable[str],
    ks: Sequence[int] = DEFAULT_KS,
) -> dict[int, float]:
    """Compute the competition-oriented Recall@1/5/20/50/100 curve."""
    return {
        int(k): recall_at_k(ranked_candidates, relevant_video_ids, int(k))
        for k in ks
    }


def mean_recall_at_ks(
    ranked_lists: Iterable[Sequence[Any]],
    relevant_sets: Iterable[Iterable[str]],
    ks: Sequence[int] = DEFAULT_KS,
) -> dict[int, float]:
    """Average Recall@K over a query collection."""
    rows = [
        recall_at_ks(ranked, relevant, ks)
        for ranked, relevant in zip(ranked_lists, relevant_sets)
    ]
    if not rows:
        return {int(k): 0.0 for k in ks}
    return {int(k): sum(row[int(k)] for row in rows) / len(rows) for k in ks}


def _video_id(candidate: Any) -> str:
    if isinstance(candidate, str):
        return candidate
    if isinstance(candidate, dict):
        try:
            return str(candidate["video_id"])
        except KeyError as exc:
            raise ValueError(f"Candidate has no video_id: {candidate!r}") from exc

    try:
        return str(candidate.video_id)
    except AttributeError as exc:
        raise ValueError(f"Candidate has no video_id: {candidate!r}") from exc
