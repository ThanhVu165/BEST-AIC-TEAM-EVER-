from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Sequence


@dataclass(frozen=True)
class FrameEvidence:
    """A single retrieved frame with an optional temporal score."""

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
    """Select deterministic semantic-frame hypotheses from retrieved evidence.

    This is deliberately conservative: without a learned temporal model we do
    not invent temporal evidence. Retrieval score is used only as a baseline
    temporal proxy, while preserving the original source frame identity.
    """
    if max_candidates <= 0:
        return []

    ordered = sorted(
        frames,
        key=lambda x: (
            -_safe_score(x.retrieval_score),
            x.video_id,
            x.frame_id,
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


def align_event_sequence(
    events: Iterable[Sequence[FrameEvidence]],
    *,
    max_candidates_per_event: int = 100,
) -> list[list[TemporalCandidate]]:
    """Align each event independently while retaining top-k hypotheses."""
    return [
        select_semantic_keyframes(
            event,
            max_candidates=max_candidates_per_event,
        )
        for event in events
    ]
