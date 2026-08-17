"""Temporal video-window utilities for late-stage video-model verification."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterator


@dataclass(frozen=True)
class VideoWindow:
    """A source-video interval selected for expensive model verification."""

    video_id: str
    video_path: str
    start_frame: int
    end_frame: int
    fps: float

    def __post_init__(self) -> None:
        if self.start_frame < 0 or self.end_frame < self.start_frame:
            raise ValueError("invalid frame interval")
        if self.fps <= 0:
            raise ValueError("fps must be > 0")

    @property
    def start_seconds(self) -> float:
        return self.start_frame / self.fps

    @property
    def end_seconds(self) -> float:
        return self.end_frame / self.fps


def make_video_window(
    *,
    video_id: str,
    video_path: str | Path,
    center_frame: int,
    radius: int,
    fps: float,
) -> VideoWindow:
    """Construct a bounded source-video interval around an event anchor."""
    if center_frame < 0:
        raise ValueError("center_frame must be >= 0")
    if radius < 0:
        raise ValueError("radius must be >= 0")
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"video not found: {path}")
    return VideoWindow(
        video_id=video_id,
        video_path=str(path),
        start_frame=max(0, center_frame - radius),
        end_frame=center_frame + radius,
        fps=fps,
    )


def iter_window_frames(window: VideoWindow) -> Iterator[tuple[int, object]]:
    """Yield decoded RGB frames in the requested source interval."""
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for video-window decoding") from exc

    capture = cv2.VideoCapture(window.video_path)
    try:
        if not capture.isOpened():
            return
        capture.set(cv2.CAP_PROP_POS_FRAMES, window.start_frame)
        frame_id = window.start_frame
        while frame_id <= window.end_frame:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            yield frame_id, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_id += 1
    finally:
        capture.release()


def materialize_window(window: VideoWindow, *, suffix: str = ".mp4") -> str:
    """Write a bounded interval to a temporary video for models requiring a path.

    The returned path is caller-owned and should be removed after inference.
    This function deliberately lives outside the verifier so expensive model
    backends can choose whether materialization is necessary.
    """
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required to materialize a video window") from exc

    frames = iter_window_frames(window)
    first = next(frames, None)
    if first is None:
        raise RuntimeError("video window contains no decodable frames")

    _, first_rgb = first
    height, width = first_rgb.shape[:2]
    with NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        output_path = handle.name

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        window.fps,
        (width, height),
    )
    try:
        writer.write(cv2.cvtColor(first_rgb, cv2.COLOR_RGB2BGR))
        for _, rgb in frames:
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return output_path
