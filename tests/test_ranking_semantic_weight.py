from query_engine.ranking import RankingEvidence


def test_semantic_weight_changes_fused_score():
    base = dict(video_id="v", frame_id=1, retrieval_score=0.8, semantic_score=1.0)
    low = RankingEvidence(**base, semantic_weight=0.02).fused_score
    high = RankingEvidence(**base, semantic_weight=0.30).fused_score
    assert high > low


def test_semantic_and_video_weights_are_normalized():
    evidence = RankingEvidence(
        video_id="v",
        frame_id=1,
        retrieval_score=0.5,
        semantic_score=1.0,
        video_verification_score=1.0,
        semantic_weight=0.2,
        video_verification_weight=0.3,
    )
    assert 0.0 <= evidence.fused_score <= 1.0
