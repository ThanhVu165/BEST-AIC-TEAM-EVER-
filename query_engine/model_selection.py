"""Model selection primitives for retrieval cascades.

Scores candidate backends on task quality and inference cost without coupling the
engine to a particular vendor or checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelBenchmarkResult:
    model: str
    recall_at_1: float
    recall_at_5: float
    recall_at_20: float
    latency_ms: float
    vram_gb: float


@dataclass(frozen=True)
class ModelSelectionPolicy:
    primary_metric: str = "recall_at_20"
    max_latency_ms: float | None = None
    max_vram_gb: float | None = None

    def eligible(self, result: ModelBenchmarkResult) -> bool:
        if self.max_latency_ms is not None and result.latency_ms > self.max_latency_ms:
            return False
        if self.max_vram_gb is not None and result.vram_gb > self.max_vram_gb:
            return False
        return True

    def select(self, results: list[ModelBenchmarkResult]) -> ModelBenchmarkResult | None:
        eligible = [r for r in results if self.eligible(r)]
        if not eligible:
            return None
        return max(eligible, key=lambda r: getattr(r, self.primary_metric))
