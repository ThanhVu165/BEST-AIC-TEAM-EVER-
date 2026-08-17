from pathlib import Path

import pytest

from query_engine.video_windows import VideoWindow, make_video_window


def test_video_window_exposes_bounded_interval(tmp_path: Path):
    path = tmp_path / "video.mp4"
    path.write_bytes(b"placeholder")

    window = make_video_window(
        video_id="v1",
        video_path=path,
        center_frame=100,
        radius=8,
        fps=25.0,
    )

    assert window.start_frame == 92
    assert window.end_frame == 108
    assert window.start_seconds == pytest.approx(3.68)
    assert window.end_seconds == pytest.approx(4.32)


def test_video_window_rejects_invalid_interval():
    with pytest.raises(ValueError):
        VideoWindow("v1", "video.mp4", 10, 9, 25.0)
    with pytest.raises(ValueError):
        VideoWindow("v1", "video.mp4", 0, 1, 0.0)


def test_make_video_window_rejects_missing_source(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        make_video_window(
            video_id="v1",
            video_path=tmp_path / "missing.mp4",
            center_frame=10,
            radius=2,
            fps=25.0,
        )
