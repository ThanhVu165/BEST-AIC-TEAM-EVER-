import pytest

from query_engine.video_verifier import (
    InternVideo3Verifier,
    VideoVerifierConfig,
    build_video_verifier,
)


def test_video_verifier_disabled_by_default():
    config = VideoVerifierConfig()
    assert config.enabled is False
    assert build_video_verifier(config) is None


def test_video_verifier_config_validates():
    with pytest.raises(ValueError):
        VideoVerifierConfig(candidate_limit=0)
    with pytest.raises(ValueError):
        VideoVerifierConfig(weight=1.1)
    with pytest.raises(ValueError):
        VideoVerifierConfig(fps=0)


def test_internvideo3_verifier_is_lazy():
    verifier = InternVideo3Verifier()
    assert verifier._model is None
    assert verifier._processor is None


def test_score_parser():
    assert InternVideo3Verifier._parse_score("score: 0.83") == pytest.approx(0.83)
    assert InternVideo3Verifier._parse_score("confidence=0.27") == pytest.approx(0.27)
    assert InternVideo3Verifier._parse_score("no score") == 0.0
