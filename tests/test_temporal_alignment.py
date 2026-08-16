from query_engine.temporal import FrameEvidence, group_into_temporal_windows, select_ordered_event_frames


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


def _frame(frame_id: int, score: float) -> FrameEvidence:
    return FrameEvidence("V1", frame_id, frame_id, frame_id / 30.0, score)


def test_ordered_alignment_prefers_high_score_temporal_path() -> None:
    events = [
        [_frame(20, 0.95), _frame(80, 0.70)],
        [_frame(10, 0.99), _frame(90, 0.94)],
    ]
    selected = select_ordered_event_frames(events)
    assert [item.frame_id for item in selected] == [20, 90]


def test_ordered_alignment_allows_same_frame_by_default() -> None:
    events = [[_frame(20, 0.8)], [_frame(20, 0.9)]]
    selected = select_ordered_event_frames(events)
    assert [item.frame_id for item in selected] == [20, 20]


def test_ordered_alignment_falls_back_when_no_strict_path_exists() -> None:
    events = [[_frame(20, 0.8)], [_frame(10, 0.9)]]
    selected = select_ordered_event_frames(events, allow_same_frame=False)
    assert [item.frame_id for item in selected] == [20, 10]
