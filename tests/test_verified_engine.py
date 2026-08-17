from types import SimpleNamespace
from unittest.mock import patch

from query_engine.late_verification import LateVerificationConfig
from query_engine.verified_engine import VerifiedQueryEngine
from query_engine.video_verifier import VideoVerifierConfig


class FakeVerifier:
    pass


def _engine() -> VerifiedQueryEngine:
    engine = VerifiedQueryEngine.__new__(VerifiedQueryEngine)
    engine.final_limit = 10
    engine.late_verification_config = LateVerificationConfig(
        enabled=True,
        candidate_limit=2,
        weight=0.2,
        window_radius=2,
    )
    engine.video_verifier_config = VideoVerifierConfig(enabled=False)
    engine.video_verifier = FakeVerifier()
    engine.retriever = SimpleNamespace(datastore=object())
    return engine


def test_verified_kis_fuses_video_score_and_reranks():
    engine = _engine()
    base = [
        {
            "rank": 1,
            "video_id": "v1",
            "frame_id": 10,
            "score": 0.90,
            "rerank_score": 0.90,
            "evidence": {},
        },
        {
            "rank": 2,
            "video_id": "v2",
            "frame_id": 20,
            "score": 0.80,
            "rerank_score": 0.80,
            "evidence": {},
        },
    ]

    with patch("query_engine.verified_engine.BaselineQueryEngine._solve_kis", return_value=base), patch(
        "query_engine.verified_engine.verify_candidate_windows",
        return_value={("v1", 10): 0.10, ("v2", 20): 1.0},
    ):
        result = engine._solve_kis(SimpleNamespace(text="person riding motorcycle"))

    assert result[0]["video_id"] == "v2"
    assert result[0]["evidence"]["video_verification_score"] == 1.0
    assert result[0]["evidence"]["video_verification_weight"] == 0.2


def test_verified_kis_is_noop_when_disabled():
    engine = _engine()
    engine.late_verification_config = LateVerificationConfig(enabled=False)
    base = [{"rank": 1, "video_id": "v1", "frame_id": 1, "score": 0.5, "rerank_score": 0.5, "evidence": {}}]

    with patch("query_engine.verified_engine.BaselineQueryEngine._solve_kis", return_value=base) as parent:
        result = engine._solve_kis(SimpleNamespace(text="query"))

    parent.assert_called_once()
    assert result == base
