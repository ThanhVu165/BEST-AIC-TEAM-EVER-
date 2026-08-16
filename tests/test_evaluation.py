from query_engine.evaluation import recall_at_k, recall_at_ks


def test_recall_at_k():
    ranked = ["v3", "v2", "v1"]
    assert recall_at_k(ranked, {"v1"}, 2) == 0.0
    assert recall_at_k(ranked, {"v1"}, 3) == 1.0


def test_recall_at_ks_matches_competition_cutoffs():
    ranked = ["v1", "v2", "v3"]
    result = recall_at_ks(ranked, {"v2"})
    assert result[1] == 0.0
    assert result[5] == 1.0
    assert result[20] == 1.0
    assert result[50] == 1.0
    assert result[100] == 1.0
