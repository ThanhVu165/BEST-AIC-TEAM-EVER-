from __future__ import annotations

import numpy as np
import pytest

from query_engine import SemanticAugmentedQueryEngine
from query_engine.retrieval import ClipCandidateRetriever
from query_engine.semantic_reranker import SemanticRerankConfig
from query_engine.temporal import FrameEvidence


class Embedder:
    def encode(self, text: str):
        return np.array([1.0, 0.0], dtype=np.float32)


class Store:
    def read_source_frame(self, video_id: str, frame_id: int):
        return {"video_id": video_id, "frame_id": frame_id}

    def search_clip(self, vector, top_k):
        return []


class Scorer:
    def score_images(self, images, text):
        return np.asarray([1.0 if image["frame_id"] == 7 else 0.1 for image in images], dtype=np.float32)


class TinyScorer:
    def score_images(self, images, text):
        values = {7: 1e-6, 8: 3e-6, 9: 2e-6}
        return np.asarray([values[image["frame_id"]] for image in images], dtype=np.float32)


def test_semantic_candidate_score_uses_visual_scorer():
    retriever = ClipCandidateRetriever(Store(), Embedder(), frame_top_k=10, video_top_k=10)
    engine = SemanticAugmentedQueryEngine(
        retriever,
        semantic_scorer=Scorer(),
        semantic_config=SemanticRerankConfig(enabled=True, candidate_limit=10, weight=0.2),
    )
    assert engine._semantic_candidate_score("V1", 7, "person riding motorcycle") == pytest.approx(1.0)
    assert engine._semantic_candidate_score("V1", 8, "person riding motorcycle") == pytest.approx(0.1)


def test_semantic_scores_are_calibrated_across_candidates():
    retriever = ClipCandidateRetriever(Store(), Embedder(), frame_top_k=10, video_top_k=10)
    engine = SemanticAugmentedQueryEngine(
        retriever,
        semantic_scorer=TinyScorer(),
        semantic_config=SemanticRerankConfig(enabled=True, candidate_limit=10, weight=0.2),
    )
    items = [
        FrameEvidence(video_id="V1", frame_id=7, keyframe_n=None, timestamp=None, retrieval_score=0.0),
        FrameEvidence(video_id="V1", frame_id=8, keyframe_n=None, timestamp=None, retrieval_score=0.0),
        FrameEvidence(video_id="V1", frame_id=9, keyframe_n=None, timestamp=None, retrieval_score=0.0),
    ]
    scores = engine._semantic_scores(items, "person riding motorcycle")
    assert scores[("V1", 7)] == pytest.approx(0.0)
    assert scores[("V1", 8)] == pytest.approx(1.0)
    assert scores[("V1", 9)] == pytest.approx(0.5)
