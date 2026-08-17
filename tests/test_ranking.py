import pytest

from query_engine.ranking import RankingEvidence


def test_semantic_weight_changes_fused_score():
    base = RankingEvidence(
        video_id="v1",
        frame_id=1,
        retrieval_score=0.5,
        semantic_score=1.0,
        semantic_weight=0.02,
    )
    strong = RankingEvidence(
        video_id="v1",
        frame_id=1,
        retrieval_score=0.5,
        semantic_score=1.0,
        semantic_weight=0.30,
    )
    assert strong.fused_score > base.fused_score


def test_semantic_weight_is_validated():
    with pytest.raises(ValueError):
        RankingEvidence("v1", 1, 0.5, semantic_weight=-0.1)
    with pytest.raises(ValueError):
        RankingEvidence("v1", 1, 0.5, semantic_weight=1.1)
