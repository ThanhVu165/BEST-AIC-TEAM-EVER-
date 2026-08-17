import pytest

from query_engine.ranking import RankingEvidence, diversify_candidates


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


def test_diversify_enforces_hard_per_video_cap():
    candidates = [
        RankingEvidence("v1", 1, 0.99),
        RankingEvidence("v1", 2, 0.98),
        RankingEvidence("v1", 3, 0.97),
        RankingEvidence("v2", 1, 0.96),
        RankingEvidence("v3", 1, 0.95),
    ]

    selected = diversify_candidates(candidates, limit=5, max_per_video=1)

    assert [(item.video_id, item.frame_id) for item in selected] == [
        ("v1", 1),
        ("v2", 1),
        ("v3", 1),
    ]
