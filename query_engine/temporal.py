from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Sequence


@dataclass(frozen=True)
class FrameEvidence:
    """A retrieved source frame with optional timestamp evidence."""

    video_id: str
    frame_id: int
    timestamp: float | None
    retrieval_score: float


@dataclass(frozen=True)
class TemporalCandidate:
    """A temporally grounded frame/window hypothesis."""

    video_id: str
    frame_id: int
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
    """Select deterministic semantic-frame hypotheses.

    Until a learned temporal grounder is available, the only legitimate score
    is the retrieval evidence. The function therefore never infers a boundary
    or synthesizes a frame ID; it only ranks already retrieved source frames.
    """
    if max_candidates <= 0:
        return []

    ordered = sorted(
        frames,
        key=lambda item: (
            -_safe_score(item.retrieval_score),
            item.video_id,
            item.frame_id,
        ),
    )
    return [
        TemporalCandidate(
            video_id=item.video_id,
            frame_id=item.frame_id,
            timestamp=item.timestamp,
            score=_safe_score(item.retrieval_score),
            rank=rank,
        )
        for rank, item in enumerate(ordered[:max_candidates], start=1)
    ]


def group_into_temporal_windows(
    frames: Sequence[FrameEvidence],
    *,
    max_gap_frames: int = 10,
) -> list[list[FrameEvidence]]:
    """Group retrieved frames into deterministic local temporal windows.

    This is useful for later temporal grounding and for diagnostics. It does
    not claim that a window is a ground-truth event interval.
    """
    if max_gap_frames < 0:
        raise ValueError("max_gap_frames must be >= 0")
    ordered = sorted(frames, key=lambda item: (item.video_id, item.frame_id))
    windows: list[list[FrameEvidence]] = []
    current: list[FrameEvidence] = []
    previous_video: str | None = None
    previous_frame: int | None = None

    for item in ordered:
        contiguous = (
            current
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
