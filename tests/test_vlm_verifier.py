import pytest

from query_engine.vlm_verifier import (
    Qwen25VLVerifier,
    VLMVerifierConfig,
    build_vlm_verifier,
)


def test_vlm_verifier_is_disabled_by_default():
    config = VLMVerifierConfig()
    assert config.enabled is False
    assert build_vlm_verifier(config) is None


def test_vlm_verifier_config_validates():
    with pytest.raises(ValueError):
        VLMVerifierConfig(candidate_limit=0)
    with pytest.raises(ValueError):
        VLMVerifierConfig(weight=1.1)


def test_qwen_verifier_is_lazy():
    verifier = Qwen25VLVerifier()
    assert verifier._model is None
    assert verifier._processor is None


def test_score_parser_handles_json_and_fallback():
    assert Qwen25VLVerifier._parse_score('{"score": 0.83, "reason": "riding"}') == pytest.approx(0.83)
    assert Qwen25VLVerifier._parse_score('score: 0.27') == pytest.approx(0.27)
    assert Qwen25VLVerifier._parse_score('{"score": 4}') == 1.0
