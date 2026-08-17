from query_engine.query_understanding import QuerySpec
from query_engine.semantic import decompose_query, semantic_score


def test_decompose_action_relation_entities():
    spec = QuerySpec(
        query_id="q1",
        task="KIS",
        text="a man riding a bicycle near a car",
        tokens=("a", "man", "riding", "bicycle", "near", "car"),
    )
    semantic = decompose_query(spec)
    assert "riding" in semantic.actions
    assert "near" in semantic.relations
    assert "man" in semantic.entities
    assert "bicycle" in semantic.entities
    assert "car" in semantic.entities


def test_semantic_score_is_bounded_and_temporal_sensitive_for_actions():
    spec = QuerySpec(
        query_id="q2",
        task="KIS",
        text="a man riding a bicycle",
        tokens=("a", "man", "riding", "bicycle"),
        actions=("riding",),
        entities=("man", "bicycle"),
    )
    low = semantic_score(spec, object_score=0.5, temporal_score=0.0)
    high = semantic_score(spec, object_score=0.5, temporal_score=1.0)
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0
    assert high > low
