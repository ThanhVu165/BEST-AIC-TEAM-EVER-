from __future__ import annotations

import numpy as np

from query_engine import BaselineQueryEngine
from query_engine.retrieval import ClipCandidateRetriever
from schemas import QueryRequest


class FakeEmbedder:
    def encode(self, text: str) -> np.ndarray:
        assert text
        return np.array([1.0, 0.0], dtype=np.float32)


class FakeDataStore:
    def search_clip(self, vector: np.ndarray, top_k: int) -> list[dict]:
        assert vector.shape == (2,)
        assert top_k == 10
        return [
            {"video_id": "V2", "frame_id": 20, "score": 0.80, "faiss_id": 1},
            {"video_id": "V1", "frame_id": 10, "score": 0.95, "faiss_id": 0},
            {"video_id": "V1", "frame_id": 11, "score": 0.90, "faiss_id": 2},
        ]


def make_engine() -> BaselineQueryEngine:
    retriever = ClipCandidateRetriever(
        FakeDataStore(),
        FakeEmbedder(),
        frame_top_k=10,
        video_top_k=10,
        max_frames_per_video=3,
    )
    return BaselineQueryEngine(retriever)


def test_kis_preserves_multiple_frame_hypotheses() -> None:
    result = make_engine().search(
        QueryRequest(query_id="q1", task="KIS", description="a person speaks")
    )

    assert result.status == "completed"
    assert result.task == "KIS"
    assert [(item["video_id"], item["frame_id"]) for item in result.candidates] == [
        ("V1", 10),
        ("V1", 11),
        ("V2", 20),
    ]


def test_video_tasks_aggregate_to_one_hypothesis_per_video() -> None:
    result = make_engine().search(
        QueryRequest(query_id="q2", description="scene", question="what color?")
    )

    assert result.task == "QA"
    assert [(item["video_id"], item["frame_id"]) for item in result.candidates] == [
        ("V1", 10),
        ("V2", 20),
    ]
    assert result.candidates[0]["answer"] == ""


def test_task_inference() -> None:
    result = make_engine().search(
        QueryRequest(
            query_id="q3",
            events=[{"event_id": "E1", "description": "person starts running"}],
        )
    )

    assert result.task == "TRAKE"
    assert [item["video_id"] for item in result.candidates] == ["V1", "V2"]
