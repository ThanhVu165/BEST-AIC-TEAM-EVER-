"""Deterministic end-to-end engine used until real retrieval is implemented."""
from __future__ import annotations

from schemas import (
    Candidate,
    QACandidate,
    QueryRequest,
    SearchResponse,
    TRAKECandidate,
    TRAKEEventPrediction,
)
from .interfaces import QueryEngine


class MockQueryEngine(QueryEngine):
    """Provides contract-valid responses for API/UI integration tests."""

    def search(self, request: QueryRequest) -> SearchResponse:
        task = request.task or self._infer_task(request)
        if task == "KIS":
            candidates = [
                Candidate(
                    rank=1,
                    video_id="MOCK_VIDEO_001",
                    frame_id=100,
                    score=0.95,
                    retrieval_score=0.95,
                    evidence={"sources": ["mock"]},
                ).model_dump()
            ]
        elif task == "QA":
            candidates = [
                QACandidate(
                    rank=1,
                    video_id="MOCK_VIDEO_001",
                    frame_id=100,
                    score=0.92,
                    answer="mock answer",
                    evidence={"sources": ["mock"]},
                ).model_dump()
            ]
        else:
            candidates = [
                TRAKECandidate(
                    rank=1,
                    video_id="MOCK_VIDEO_001",
                    score=0.90,
                    events=[
                        TRAKEEventPrediction(
                            event_id=event.event_id,
                            frame_id=100 + i * 10,
                            score=0.90,
                        )
                        for i, event in enumerate(request.events)
                    ],
                ).model_dump()
            ]

        return SearchResponse(
            query_id=request.query_id,
            task=task,
            status="completed",
            candidates=candidates,
        )

    @staticmethod
    def _infer_task(request: QueryRequest) -> str:
        if request.events:
            return "TRAKE"
        if request.question:
            return "QA"
        return "KIS"
