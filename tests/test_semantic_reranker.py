import numpy as np
import pytest

from query_engine.query_understanding import QuerySpec
from query_engine.semantic_reranker import (
    SemanticRerankConfig,
    SigLIP2ImageTextScorer,
    build_semantic_reranker,
    semantic_text,
)


def test_semantic_rerank_config_is_safe_by_default():
    config = SemanticRerankConfig()
    assert config.enabled is False
    assert config.model_id == "google/siglip2-base-patch16-256"
    assert build_semantic_reranker(config) is None


def test_semantic_rerank_config_validates_limits():
    with pytest.raises(ValueError):
        SemanticRerankConfig(candidate_limit=0)
    with pytest.raises(ValueError):
        SemanticRerankConfig(weight=1.1)


def test_semantic_text_preserves_relation_phrase():
    spec = QuerySpec(
        query_id="q1",
        text="person riding motorcycle",
        task="KIS",
        tokens=("person", "riding", "motorcycle"),
    )
    assert semantic_text(spec) == "person riding motorcycle"


def test_siglip_adapter_does_not_load_model_on_import():
    scorer = SigLIP2ImageTextScorer()
    assert scorer._model is None
    assert scorer._processor is None


def test_empty_image_batch_is_model_free():
    scorer = SigLIP2ImageTextScorer()
    result = scorer.score_images([], "person riding motorcycle")
    assert isinstance(result, np.ndarray)
    assert result.shape == (0,)


def test_empty_text_is_rejected_without_loading_model():
    scorer = SigLIP2ImageTextScorer()
    with pytest.raises(ValueError):
        scorer.score_images([object()], "   ")
    assert scorer._model is None
