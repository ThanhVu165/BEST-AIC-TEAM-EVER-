from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any, Protocol


@dataclass(frozen=True)
class FrameEvidence:
    """Retrieved source-frame evidence used by temporal stages."""

    video_id: str
    frame_id: int
    keyframe_n: int | None
    timestamp: float | None
    retrieval_score: float


@dataclass(frozen=True)
class TemporalCandidate:
    """Temporally grounded frame hypothesis."""

    video_id: str
    frame_id: int
    keyframe_n: int | None
    timestamp: float | None
    score: float
    rank: int


class SourceFrameReader(Protocol):
    def read_source_frame(self, video_id: str, frame_id: int) -> Any | None:
        ...


class ImageEncoder(Protocol):
    def encode(self, text: str):  # pragma: no cover - protocol
        ...

    def encode_images(self, images: Sequence[Any], *, batch_size: int = 16):  # pragma: no cover - protocol
        ...


def _safe_score(value: float) -> float:
    value = float(value)
    return value if isfinite(value) else float("-inf")


def select_semantic_keyframes(
    frames: Sequence[FrameEvidence],
    *,
    max_candidates: int = 100,
) -> list[TemporalCandidate]:
    """Select deterministic keyframe hypotheses from retrieved evidence.

    This stage is a candidate selector. When source-video access and an image
    encoder are available, ``fine_localize_source_frames`` performs the actual
    frame-level grounding before this output is used as the final hypothesis.
    """
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


def fine_localize_source_frames(
    frames: Sequence[FrameEvidence],
    *,
    query_text: str,
    reader: SourceFrameReader,
    image_encoder: ImageEncoder,
    radius: int = 16,
    stride: int = 1,
    max_candidates: int = 100,
    image_batch_size: int = 16,
) -> list[TemporalCandidate]:
    """Refine retrieved hypotheses against original-video frames.

    Each sparse retrieval hit is treated as an anchor, not as the final answer.
    A small source-video neighborhood is decoded and scored with CLIP image
    embeddings against the same natural-language query. The returned frame_id
    is therefore an original-video frame ID, including non-keyframes when the
    source video is available.

    This is still a local CLIP temporal proxy rather than a learned temporal
    grounding model, but it closes the architectural gap between sparse
    keyframe retrieval and source-frame output.
    """
    if not query_text.strip():
        raise ValueError("query_text must not be empty")
    if radius < 0 or stride <= 0 or max_candidates <= 0:
        raise ValueError("radius >= 0, stride > 0 and max_candidates > 0 are required")

    anchors = sorted(
        frames,
        key=lambda item: (-_safe_score(item.retrieval_score), item.video_id, item.frame_id),
    )[: max_candidates * 2]
    if not anchors:
        return []

    query_vector = image_encoder.encode(query_text)
    query_vector = query_vector / max(float((query_vector**2).sum()) ** 0.5, 1e-12)

    best_by_key: dict[tuple[str, int], TemporalCandidate] = {}
    for anchor in anchors:
        frame_ids = range(
            max(0, anchor.frame_id - radius),
            anchor.frame_id + radius + 1,
            stride,
        )
        images: list[Any] = []
        valid_ids: list[int] = []
        for frame_id in frame_ids:
            image = reader.read_source_frame(anchor.video_id, frame_id)
            if image is None:
                continue
            images.append(image)
            valid_ids.append(frame_id)
        if not images:
            # Source video unavailable: retain the sparse hypothesis rather than
            # inventing a frame or dropping the candidate entirely.
            fallback = TemporalCandidate(
                video_id=anchor.video_id,
                frame_id=anchor.frame_id,
                keyframe_n=anchor.keyframe_n,
                timestamp=anchor.timestamp,
                score=anchor.retrieval_score,
                rank=0,
            )
            best_by_key[(fallback.video_id, fallback.frame_id)] = fallback
            continue

        image_vectors = image_encoder.encode_images(images, batch_size=image_batch_size)
        scores = image_vectors @ query_vector
        best_idx = int(scores.argmax())
        best_frame_id = valid_ids[best_idx]
        best_score = float(scores[best_idx])
        candidate = TemporalCandidate(
            video_id=anchor.video_id,
            frame_id=best_frame_id,
            keyframe_n=anchor.keyframe_n if best_frame_id == anchor.frame_id else None,
            timestamp=None,
            score=best_score,
            rank=0,
        )
        key = (candidate.video_id, candidate.frame_id)
        previous = best_by_key.get(key)
        if previous is None or candidate.score > previous.score:
            best_by_key[key] = candidate

    ranked = sorted(
        best_by_key.values(),
        key=lambda item: (-_safe_score(item.score), item.video_id, item.frame_id),
    )[:max_candidates]
    return [
        TemporalCandidate(
            video_id=item.video_id,
            frame_id=item.frame_id,
            keyframe_n=item.keyframe_n,
            timestamp=item.timestamp,
            score=item.score,
            rank=rank,
        )
        for rank, item in enumerate(ranked, start=1)
    ]


def select_ordered_event_frames(
    events: Sequence[Sequence[FrameEvidence]],
    *,
    max_candidates_per_event: int = 100,
    allow_same_frame: bool = False,
) -> list[TemporalCandidate]:
    """Select one frame per event while respecting strict temporal event order.

    Every output frame must already be present in retrieval evidence. A valid
    path maximizes cumulative retrieval evidence under the event-order
    constraint. If no valid path exists, an empty result is returned so the
    caller does not silently emit a semantically invalid alignment.
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
        return []

    dp: list[list[float]] = [[_safe_score(item.retrieval_score) for item in candidates[0]]]
    parent: list[list[int | None]] = [[None] * len(candidates[0])]

    for event_idx in range(1, len(candidates)):
        current_scores: list[float] = []
        current_parent: list[int | None] = []
        for current in candidates[event_idx]:
            best_score = float("-inf")
            best_parent: int | None = None
            for previous_idx, previous in enumerate(candidates[event_idx - 1]):
                valid = (
                    current.frame_id >= previous.frame_id
                    if allow_same_frame
                    else current.frame_id > previous.frame_id
                )
                if not valid or not isfinite(dp[event_idx - 1][previous_idx]):
                    continue
                score = dp[event_idx - 1][previous_idx] + _safe_score(current.retrieval_score)
                if score > best_score:
                    best_score = score
                    best_parent = previous_idx
                elif score == best_score and best_parent is not None:
                    if previous.frame_id < candidates[event_idx - 1][best_parent].frame_id:
                        best_parent = previous_idx
            current_scores.append(best_score)
            current_parent.append(best_parent)
        dp.append(current_scores)
        parent.append(current_parent)

    final_idx = max(
        range(len(candidates[-1])),
        key=lambda idx: (dp[-1][idx], -candidates[-1][idx].frame_id, -idx),
    )
    if not isfinite(dp[-1][final_idx]):
        return []

    selected_indices = [final_idx]
    for event_idx in range(len(candidates) - 1, 0, -1):
        previous_idx = parent[event_idx][selected_indices[-1]]
        if previous_idx is None:
            return []
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
    """Group retrieved source frames into local frame-contiguous windows."""
    if max_gap_frames < 0:
        raise ValueError("max_gap_frames must be >= 0")
    ordered = sorted(frames, key=lambda item: (item.video_id, item.frame_id, item.keyframe_n or 0))
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
    """Retain ranked hypotheses per event without forcing alignment."""
    return [
        select_semantic_keyframes(event, max_candidates=max_candidates_per_event)
        for event in events
    ]
