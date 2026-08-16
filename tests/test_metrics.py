from __future__ import annotations

import pytest

from query_engine.metrics import recall_at_k, recall_curve


def test_recall_at_k() -> None:
    candidates = [
        {"video_id": "V1"},
        {"video_id": "V2"},
        {"video_id": "V3"},
    ]
    assert recall_at_k(candidates, {"V2"}, 1) == 0.0
    assert recall_at_k(candidates, {"V2"}, 2) == 1.0


def test_recall_curve() -> None:
    candidates = [{"video_id": "V1"}, {"video_id": "V2"}]
    assert recall_curve(candidates, {"V2"}, (1, 5)) == {1: 0.0, 5: 1.0}


def test_recall_requires_valid_inputs() -> None:
    with pytest.raises(ValueError):
        recall_at_k([], {"V1"}, 0)
    with pytest.raises(ValueError):
        recall_at_k([], set(), 1)
