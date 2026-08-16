from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class FrameEvidence:
    """A retrieved source frame with optional timestamp evidence."""

    video_id: str
    frame_id: int
    keyframe_n: int | None
    timestamp: float | None
    retrieval_score: float


@dataclass(frozen=True)
class TemporalCandidate:
    """A temporally grounded frame/window hypothesis."""

    video_id: str
    frame_id: int
    keyframe_n: int | None
    timestamp: float | None
    score: float
    rank: int


def _safe_score(value: float) -> float:
    value = float(value)
    return value if isfinite(value) else float("-inf")


def select_semantic_keyframes(
    frames: Sequence[FrameEvidence],
    *,
    max_candidates: int = 100,
) -> list[TemporalCandidate]:
    """Select deterministic semantic-frame hypotheses from retrieved evidence."""
    if max_candidates <= 0:
        return []

    ordered = sorted(
        frames,
        key=lambda item: (
            -_safe_score(item.retrieval_score),
            item.video_id,
            item.frame_id,
            item.keyframe_n or 0,
        ),
    )
    return [
        TemporalCandidate(
            video_id=item.video_id,
            frame_id=item.frame_id,
            keyframe_n=item.keyframe_n,
            timestamp=item.timestamp,
            score=_safe_score(item.retrieval_score),
            rank=rank,
        )
        for rank, item in enumerate(ordered[:max_candidates], start=1)
    ]


def select_ordered_event_frames(
    events: Sequence[Sequence[FrameEvidence]],
    *,
    max_candidates_per_event: int = 100,
    allow_same_frame: bool = True,
) -> list[TemporalCandidate]:
    """Choose one frame per event while respecting temporal event order.

    This is a small dynamic-programming aligner intended for TRAKE candidate
    generation. It does not invent frames: every returned frame must already be
    present in the retrieval candidates. The objective is the sum of retrieval
    scores, with a deterministic tie-break toward earlier frames.

    If no temporally valid path exists, the function falls back to the strongest
    candidate of each event rather than silently dropping the video.
    """
    if not events or max_candidates_per_event <= 0:
        return []

    candidates: list[list[FrameEvidence]] = []
    for event in events:
        ranked = sorted(
            event,
            key=lambda item: (
                -_safe_score(item.retrieval_score),
                item.frame_id,
                item.keyframe_n or 0,
            ),
        )[:max_candidates_per_event]
        candidates.append(ranked)

    if any(not event for event in candidates):
        return [
            TemporalCandidate(
                video_id=event[0].video_id,
                frame_id=event[0].frame_id,
                keyframe_n=event[0].keyframe_n,
                timestamp=event[0].timestamp,
                score=_safe_score(event[0].retrieval_score),
                rank=1,
            )
            for event in candidates
            if event
        ]

    # dp[i][j] = best cumulative score ending at candidate j of event i.
    dp: list[list[float]] = [[_safe_score(item.retrieval_score) for item in candidates[0]]]
    parent: list[list[int | None]] = [[None] * len(candidates[0])]

    for event_idx in range(1, len(candidates)):
        current_scores: list[float] = []
        current_parent: list[int | None] = []
        for current in candidates[event_idx]:
            best_score = float("-inf")
            best_parent: int | None = None
            for previous_idx, previous in enumerate(candidates[event_idx - 1]):
                valid = current.frame_id >= previous.frame_id if allow_same_frame else current.frame_id > previous.frame_id
                if not valid or not isfinite(dp[event_idx - 1][previous_idx]):
                    continue
                score = dp[event_idx - 1][previous_idx] + _safe_score(current.retrieval_score)
                if (
                    score > best_score
                    or (
                        score == best_score
                        and best_parent is not None
                        and previous.frame_id < candidates[event_idx - 1][best_parent].frame_id
                    )
                ):
                    best_score = score
                    best_parent = previous_idx
            current_scores.append(best_score)
            current_parent.append(best_parent)
        dp.append(current_scores)
        parent.append(current_parent)

    final_idx = max(
        range(len(candidates[-1])),
        key=lambda idx: (
            dp[-1][idx],
            -candidates[-1][idx].frame_id,
            -idx,
        ),
    )
    if not isfinite(dp[-1][final_idx]):
        return [
            TemporalCandidate(
                video_id=event[0].video_id,
                frame_id=event[0].frame_id,
                keyframe_n=event[0].keyframe_n,
                timestamp=event[0].timestamp,
                score=_safe_score(event[0].retrieval_score),
                rank=1,
            )
            for event in candidates
        ]

    selected_indices = [final_idx]
    for event_idx in range(len(candidates) - 1, 0, -1):
        previous_idx = parent[event_idx][selected_indices[-1]]
        assert previous_idx is not None
        selected_indices.append(previous_idx)
    selected_indices.reverse()

    selected: list[TemporalCandidate] = []
    for rank, (event, idx) in enumerate(zip(candidates, selected_indices), start=1):
        item = event[idx]
        selected.append(
            TemporalCandidate(
                video_id=item.video_id,
                frame_id=item.frame_id,
                keyframe_n=item.keyframe_n,
                timestamp=item.timestamp,
                score=_safe_score(item.retrieval_score),
                rank=rank,
            )
        )
    return selected


def group_into_temporal_windows(
    frames: Sequence[FrameEvidence],
    *,
    max_gap_frames: int = 10,
) -> list[list[FrameEvidence]]:
    """Group retrieved frames into deterministic local temporal windows."""
    if max_gap_frames < 0:
        raise ValueError("max_gap_frames must be >= 0")
    ordered = sorted(
        frames,
        key=lambda item: (item.video_id, item.frame_id, item.keyframe_n or 0),
    )
    windows: list[list[FrameEvidence]] = []
    current: list[FrameEvidence] = []
    previous_video: str | None = None
    previous_frame: int | None = None

    for item in ordered:
        contiguous = (
            bool(current)
            and item.video_id == previous_video
            and previous_frame is not None
            and item.frame_id - previous_frame <= max_gap_frames
        )
        if not contiguous:
            if current:
                windows.append(current)
            current = [item]
        else:
            current.append(item)
        previous_video = item.video_id
        previous_frame = item.frame_id

    if current:
        windows.append(current)
    return windows


def align_event_sequence(
    events: Iterable[Sequence[FrameEvidence]],
    *,
    max_candidates_per_event: int = 100,
) -> list[list[TemporalCandidate]]:
    """Align each event independently while retaining top-k hypotheses."""
    return [
        select_semantic_keyframes(event, max_candidates=max_candidates_per_event)
        for event in events
    ]
