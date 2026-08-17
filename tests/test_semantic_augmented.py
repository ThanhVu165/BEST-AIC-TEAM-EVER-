from __future__ import annotations

import numpy as np

from query_engine import SemanticAugmentedQueryEngine
from query_engine.retrieval import ClipCandidateRetriever
from query_engine.semantic_reranker import SemanticRerankConfig


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


def test_semantic_candidate_score_uses_visual_scorer():
    retriever = ClipCandidateRetriever(Store(), Embedder(), frame_top_k=10, video_top_k=10)
    engine = SemanticAugmentedQueryEngine(
        retriever,
        semantic_scorer=Scorer(),
        semantic_config=SemanticRerankConfig(enabled=True, candidate_limit=10, weight=0.2),
    )
    assert engine._semantic_candidate_score("V1", 7, "person riding motorcycle") == 1.0
    assert engine._semantic_candidate_score("V1", 8, "person riding motorcycle") == 0.1
