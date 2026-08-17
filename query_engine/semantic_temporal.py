"""Semantic-aware temporal alignment primitives.

The canonical TRAKE DP must optimize the same evidence used by semantic
reranking. This module keeps the dynamic-programming implementation independent
from a particular VLM/encoder by accepting precomputed semantic scores.
"""
from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

from .temporal import FrameEvidence, TemporalCandidate


def select_semantic_ordered_event_frames(
    events: Sequence[Sequence[FrameEvidence]],
    semantic_scores: Sequence[dict[tuple[str, int], float]],
    *,
    semantic_weight: float = 0.15,
    max_candidates_per_event: int = 100,
    allow_same_frame: bool = False,
) -> list[TemporalCandidate]:
    """Select one frame per event using retrieval + semantic evidence.

    Objective per hypothesis:
        (1 - semantic_weight) * retrieval_score
        + semantic_weight * semantic_score

    The DP then maximizes the cumulative objective under strict temporal order.
    Missing semantic scores fall back to retrieval-only evidence; they never
    fabricate a semantic score.
    """
    if not events or len(events) != len(semantic_scores):
        return []
    if not 0.0 <= semantic_weight <= 1.0:
        raise ValueError("semantic_weight must be in [0, 1]")
    if max_candidates_per_event <= 0:
        return []

    candidates: list[list[FrameEvidence]] = []
    local_scores: list[dict[tuple[str, int], float]] = []
    for event, scores in zip(events, semantic_scores):
        ranked = list(event[:max_candidates_per_event])
        if not ranked:
            return []
        ranked.sort(key=lambda item: (-float(item.retrieval_score), item.frame_id, item.keyframe_n or 0))
        candidates.append(ranked)
        local_scores.append(scores)

    def objective(event_idx: int, item: FrameEvidence) -> float:
        retrieval = max(0.0, min(1.0, float(item.retrieval_score)))
        semantic = local_scores[event_idx].get((item.video_id, item.frame_id))
        if semantic is None:
            return retrieval
        semantic = max(0.0, min(1.0, float(semantic)))
        return (1.0 - semantic_weight) * retrieval + semantic_weight * semantic

    dp: list[list[float]] = [[objective(0, item) for item in candidates[0]]]
    parent: list[list[int | None]] = [[None] * len(candidates[0])]

    for event_idx in range(1, len(candidates)):
        current_scores: list[float] = []
        current_parent: list[int | None] = []
        for current in candidates[event_idx]:
            best_score = float("-inf")
            best_parent: int | None = None
            for previous_idx, previous in enumerate(candidates[event_idx - 1]):
                valid = current.frame_id >= previous.frame_id if allow_same_frame else current.frame_id > previous.frame_id
                previous_score = dp[event_idx - 1][previous_idx]
                if not valid or not isfinite(previous_score):
                    continue
                score = previous_score + objective(event_idx, current)
                if score > best_score:
                    best_score, best_parent = score, previous_idx
                elif score == best_score and best_parent is not None:
                    if previous.frame_id < candidates[event_idx - 1][best_parent].frame_id:
                        best_parent = previous_idx
            current_scores.append(best_score)
            current_parent.append(best_parent)
        dp.append(current_scores)
        parent.append(current_parent)

    final_idx = max(range(len(candidates[-1])), key=lambda idx: (dp[-1][idx], -candidates[-1][idx].frame_id, -idx))
    if not isfinite(dp[-1][final_idx]):
        return []

    indices = [final_idx]
    for event_idx in range(len(candidates) - 1, 0, -1):
        previous = parent[event_idx][indices[-1]]
        if previous is None:
            return []
        indices.append(previous)
    indices.reverse()

    selected: list[TemporalCandidate] = []
    for rank, (event, idx) in enumerate(zip(candidates, indices), start=1):
        item = event[idx]
        selected.append(
            TemporalCandidate(
                video_id=item.video_id,
                frame_id=item.frame_id,
                keyframe_n=item.keyframe_n,
                timestamp=item.timestamp,
                score=objective(rank - 1, item),
                rank=rank,
            )
        )
    return selected
