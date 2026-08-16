from query_engine.ranking import RankingEvidence, rerank_candidates
from query_engine.temporal import FrameEvidence, select_semantic_keyframes


def test_temporal_selection_preserves_source_frame_identity():
    frames = [
        FrameEvidence("v2", 17, 1.7, 0.4),
        FrameEvidence("v1", 99, 9.9, 0.9),
        FrameEvidence("v1", 12, 1.2, 0.9),
    ]
    result = select_semantic_keyframes(frames)

    assert [(x.video_id, x.frame_id) for x in result] == [
        ("v1", 12),
        ("v1", 99),
        ("v2", 17),
    ]
    assert result[0].rank == 1


def test_temporal_selection_has_explicit_candidate_limit():
    frames = [FrameEvidence("v", i, float(i), 1.0) for i in range(10)]
    result = select_semantic_keyframes(frames, max_candidates=3)
    assert len(result) == 3


def test_reranking_keeps_primary_retrieval_dominant():
    candidates = [
        RankingEvidence("v1", 2, retrieval_score=0.9, object_score=0.0),
        RankingEvidence("v2", 3, retrieval_score=0.8, object_score=1.0),
    ]
    result = rerank_candidates(candidates)
    assert result[0].video_id == "v1"


def test_reranking_is_deterministic_on_ties():
    candidates = [
        RankingEvidence("v2", 4, retrieval_score=0.5),
        RankingEvidence("v1", 9, retrieval_score=0.5),
    ]
    result = rerank_candidates(candidates)
    assert [(x.video_id, x.frame_id) for x in result] == [("v1", 9), ("v2", 4)]
