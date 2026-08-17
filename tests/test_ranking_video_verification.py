from query_engine.ranking import RankingEvidence


def test_video_verification_weight_changes_score():
    base = RankingEvidence(
        video_id="v1",
        frame_id=1,
        retrieval_score=0.9,
        semantic_score=0.0,
        video_verification_score=1.0,
        semantic_weight=0.0,
        video_verification_weight=0.0,
    )
    verified = RankingEvidence(
        video_id="v1",
        frame_id=1,
        retrieval_score=0.9,
        semantic_score=0.0,
        video_verification_score=1.0,
        semantic_weight=0.0,
        video_verification_weight=0.20,
    )
    assert verified.fused_score > base.fused_score


def test_semantic_and_video_weights_cannot_exceed_one():
    try:
        RankingEvidence(
            video_id="v1",
            frame_id=1,
            retrieval_score=0.5,
            semantic_weight=0.7,
            video_verification_weight=0.4,
        )
    except ValueError as exc:
        assert "must be <= 1" in str(exc)
    else:
        raise AssertionError("expected invalid combined weights to fail")
