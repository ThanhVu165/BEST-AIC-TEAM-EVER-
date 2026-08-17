from query_engine.semantic_temporal import select_semantic_ordered_event_frames
from query_engine.temporal import FrameEvidence


def frame(frame_id: int, retrieval: float) -> FrameEvidence:
    return FrameEvidence("v", frame_id, None, None, retrieval)


def test_semantic_score_changes_trake_path():
    events = [
        [frame(10, 0.95), frame(20, 0.60)],
        [frame(30, 0.60), frame(40, 0.95)],
    ]
    semantic = [
        {("v", 10): 0.10, ("v", 20): 1.0},
        {("v", 30): 1.0, ("v", 40): 0.10},
    ]
    selected = select_semantic_ordered_event_frames(events, semantic, semantic_weight=0.5)
    assert [item.frame_id for item in selected] == [20, 30]


def test_strict_order_is_preserved():
    events = [[frame(20, 1.0)], [frame(20, 1.0)]]
    semantic = [{("v", 20): 1.0}, {("v", 20): 1.0}]
    assert select_semantic_ordered_event_frames(events, semantic, semantic_weight=0.5) == []


def test_missing_semantic_score_falls_back_to_retrieval():
    events = [[frame(10, 0.8)], [frame(20, 0.7)]]
    selected = select_semantic_ordered_event_frames(events, [{}, {}], semantic_weight=0.9)
    assert [item.frame_id for item in selected] == [10, 20]
    assert [item.score for item in selected] == [0.8, 0.7]
