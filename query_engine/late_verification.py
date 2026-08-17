"""Late-stage video verification over bounded temporal windows.

This module intentionally sits after cheap retrieval and semantic reranking.
It never scans the corpus: only a small set of source-video windows is passed
to an expensive video-language backend.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .video_windows import VideoWindow, make_video_window, materialize_window


class WindowVerifier(Protocol):
    def verify(self, video_path: str, query: str, *, fps: float = 2.0) -> float:
        ...


@dataclass(frozen=True)
class LateVerificationConfig:
    enabled: bool = False
    candidate_limit: int = 3
    window_radius: int = 16
    weight: float = 0.10
    materialize: bool = True
    fps: float = 2.0

    def __post_init__(self) -> None:
        if self.candidate_limit <= 0:
            raise ValueError("candidate_limit must be > 0")
        if self.window_radius < 0:
            raise ValueError("window_radius must be >= 0")
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError("weight must be in [0, 1]")
        if self.fps <= 0:
            raise ValueError("fps must be > 0")


def verify_candidate_windows(
    candidates: list[Any],
    *,
    datastore: Any,
    query: str,
    verifier: WindowVerifier | None,
    config: LateVerificationConfig,
) -> dict[tuple[str, int], float]:
    """Run an expensive verifier only on the highest-ranked temporal anchors.

    Candidates are expected to expose ``video_id`` and ``frame_id``. The
    datastore must expose ``get_video`` returning a record with ``path`` and
    optionally ``fps``. Missing source videos simply receive no verification
    score rather than causing the whole retrieval request to fail.
    """
    if verifier is None or not config.enabled or not candidates or not query.strip():
        return {}

    scores: dict[tuple[str, int], float] = {}
    seen: set[tuple[str, int]] = set()
    for candidate in candidates:
        key = (str(candidate.video_id), int(candidate.frame_id))
        if key in seen:
            continue
        seen.add(key)
        if len(seen) > config.candidate_limit:
            break

        video = datastore.get_video(key[0])
        if video is None:
            continue
        video_path = getattr(video, "path", None)
        fps = float(getattr(video, "fps", 0.0) or config.fps)
        if not video_path:
            continue

        try:
            window = make_video_window(
                video_id=key[0],
                video_path=video_path,
                center_frame=key[1],
                radius=config.window_radius,
                fps=fps,
            )
            score = _verify_window(verifier, window, query, fps, config.materialize)
        except (FileNotFoundError, OSError, RuntimeError, ValueError):
            continue
        scores[key] = max(0.0, min(1.0, float(score)))
    return scores


def _verify_window(
    verifier: WindowVerifier,
    window: VideoWindow,
    query: str,
    fps: float,
    materialize: bool,
) -> float:
    if not materialize:
        # A verifier may support direct source paths while using the window
        # boundaries through its own configuration. The default InternVideo3
        # adapter requires a bounded clip, so materialization is recommended.
        return float(verifier.verify(window.video_path, query, fps=fps))

    path = materialize_window(window)
    try:
        return float(verifier.verify(path, query, fps=fps))
    finally:
        try:
            import os
            os.unlink(path)
        except OSError:
            pass
