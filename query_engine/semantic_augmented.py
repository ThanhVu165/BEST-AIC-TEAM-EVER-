"""Shared late semantic stage for non-KIS tasks.

KIS already performs semantic scoring inside its RankingEvidence fusion. QA and
TRAKE need the same visual-language evidence without duplicating the baseline
retrieval implementation. This class therefore adds a bounded semantic pass
on the candidates produced by the canonical engine.
"""
from __future__ import annotations

from typing import Any

from .engine import BaselineQueryEngine
from .query_understanding import QuerySpec


class SemanticAugmentedQueryEngine(BaselineQueryEngine):
    """Baseline engine with visual semantic reranking for QA and TRAKE."""

    def _semantic_candidate_score(self, video_id: str, frame_id: int, text: str) -> float | None:
        if self.semantic_scorer is None:
            return None
        from .temporal import FrameEvidence

        scores = self._semantic_scores(
            [FrameEvidence(video_id=video_id, frame_id=frame_id, keyframe_n=None, timestamp=None, retrieval_score=0.0)],
            text,
        )
        return scores.get((video_id, frame_id))

    def _solve_qa(self, spec: QuerySpec) -> list[dict[str, Any]]:
        results = super()._solve_qa(spec)
        if self.semantic_scorer is None or not results:
            return results
        weight = self.semantic_config.weight
        for item in results:
            score = self._semantic_candidate_score(str(item["video_id"]), int(item["frame_id"]), spec.text)
            if score is None:
                continue
            base = max(0.0, min(1.0, float(item["score"])))
            final = (1.0 - weight) * base + weight * score
            item["score"] = final
            item["rerank_score"] = final
            evidence = item.setdefault("evidence", {})
            evidence["semantic_score"] = score
            evidence["semantic_model"] = self.semantic_config.model_id
            evidence["semantic_weight"] = weight
            evidence["rerank_score"] = final
        results.sort(key=lambda item: (-float(item["score"]), str(item["video_id"]), int(item["frame_id"])))
        for rank, item in enumerate(results, start=1):
            item["rank"] = rank
        return results[: self.final_limit]

    def _solve_trake(self, spec: QuerySpec) -> list[dict[str, Any]]:
        results = super()._solve_trake(spec)
        if self.semantic_scorer is None or not results:
            return results
        weight = self.semantic_config.weight
        for candidate in results:
            event_scores: list[float] = []
            for event in candidate.get("events", []):
                score = self._semantic_candidate_score(
                    str(candidate["video_id"]), int(event["frame_id"]),
                    next((e.description for e in spec.events if e.event_id == event["event_id"]), spec.text),
                )
                if score is None:
                    event_scores.append(float(event["score"]))
                    continue
                base = max(0.0, min(1.0, float(event["score"])))
                final = (1.0 - weight) * base + weight * score
                event["score"] = final
                event_scores.append(final)
            if event_scores:
                candidate["score"] = sum(event_scores) / len(event_scores)
        results.sort(key=lambda item: (-float(item["score"]), str(item["video_id"])))
        for rank, item in enumerate(results, start=1):
            item["rank"] = rank
        return results[: self.final_limit]


def build_semantic_augmented_query_engine(*args: Any, **kwargs: Any) -> SemanticAugmentedQueryEngine:
    return SemanticAugmentedQueryEngine(*args, **kwargs)
