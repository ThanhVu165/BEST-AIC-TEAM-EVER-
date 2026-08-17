import pytest

from query_engine.model_selection import ModelBenchmarkResult, ModelSelectionPolicy


def test_selection_prefers_primary_metric_within_budget() -> None:
    results = [
        ModelBenchmarkResult("small", 0.4, 0.7, 0.88, 20.0, 2.0),
        ModelBenchmarkResult("large", 0.5, 0.8, 0.94, 80.0, 8.0),
    ]
    selected = ModelSelectionPolicy(max_latency_ms=100, max_vram_gb=10).select(results)
    assert selected is not None
    assert selected.model == "large"


def test_selection_respects_latency_and_vram_budgets() -> None:
    results = [
        ModelBenchmarkResult("fast", 0.4, 0.7, 0.88, 20.0, 2.0),
        ModelBenchmarkResult("accurate", 0.5, 0.8, 0.99, 120.0, 10.0),
    ]
    selected = ModelSelectionPolicy(max_latency_ms=50, max_vram_gb=4).select(results)
    assert selected is not None
    assert selected.model == "fast"


def test_selection_returns_none_when_no_model_is_eligible() -> None:
    result = ModelBenchmarkResult("large", 0.5, 0.8, 0.99, 120.0, 10.0)
    assert ModelSelectionPolicy(max_latency_ms=50).select([result]) is None


def test_selection_rejects_unknown_primary_metric() -> None:
    result = ModelBenchmarkResult("m", 0.5, 0.8, 0.9, 10.0, 1.0)
    with pytest.raises(AttributeError):
        ModelSelectionPolicy(primary_metric="does_not_exist").select([result])
