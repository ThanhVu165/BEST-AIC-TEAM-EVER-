from query_engine.temporal import FrameEvidence, group_into_temporal_windows


def test_temporal_windows_preserve_source_frames() -> None:
    frames = [
        FrameEvidence("V1", 20, 3, 2.0, 0.7),
        FrameEvidence("V1", 12, 1, 1.2, 0.9),
        FrameEvidence("V1", 13, 2, 1.3, 0.8),
        FrameEvidence("V2", 1, 1, 0.1, 0.6),
    ]
    windows = group_into_temporal_windows(frames, max_gap_frames=1)
    assert [[item.frame_id for item in window] for window in windows] == [
        [12, 13],
        [20],
        [1],
    ]
    assert windows[0][0].video_id == "V1"
    assert windows[0][0].keyframe_n == 1


def test_temporal_window_rejects_negative_gap() -> None:
    try:
        group_into_temporal_windows([], max_gap_frames=-1)
    except ValueError as exc:
        assert "max_gap_frames" in str(exc)
    else:
        raise AssertionError("negative max_gap_frames must fail")
