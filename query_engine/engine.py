"""Query Engine orchestration for KIS, QA and TRAKE."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from schemas import Candidate, QACandidate, QueryRequest, SearchResponse, TRAKECandidate

from .interfaces import QueryEngine
from .retrieval import ClipCandidateRetriever, RetrievalHit
from .temporal import FrameEvidence, select_semantic_keyframes


class BaselineQueryEngine(QueryEngine):
    """Run CLIP retrieval and deterministic evidence-aware task solvers.

    The current temporal stage is explicitly a retrieval proxy. It preserves
    source frame IDs and never invents an event boundary. A learned temporal
    model can later replace ``select_semantic_keyframes`` without changing the
    API contract.
    """

    def __init__(self, retriever: ClipCandidateRetriever) -> None:
        self.retriever = retriever

    def search(self, request: QueryRequest) -> SearchResponse:
        task = request.task or self._infer_task(request)
        try:
            if task == "KIS":
                candidates = self._solve_kis(request)
            elif task == "QA":
                candidates = self._solve_qa(request)
            else:
                candidates = self._solve_trake(request)
        except Exception as exc:
            return SearchResponse(
                query_id=request.query_id,
                task=task,
                status="failed",
                candidates=[],
                error=str(exc),
            )

        return SearchResponse(
            query_id=request.query_id,
            task=task,
            status="completed",
            candidates=candidates[:100],
        )

    def _solve_kis(self, request: QueryRequest) -> list[dict[str, Any]]:
        query_text = self._build_query_text(request, "KIS")
        hits = self.retriever.retrieve(query_text)
        evidence = [self._frame_evidence(hit) for hit in hits]
        selected = select_semantic_keyframes(evidence, max_candidates=100)
        return [
            Candidate(
                rank=item.rank,
                video_id=item.video_id,
                frame_id=item.frame_id,
                score=item.score,
                retrieval_score=item.score,
                temporal_score=item.score,
                evidence={"sources": ["clip", "temporal_proxy"]},
            ).model_dump()
            for item in selected
        ]

    def _solve_qa(self, request: QueryRequest) -> list[dict[str, Any]]:
        query_text = self._build_query_text(request, "QA")
        hits = self.retriever.retrieve_videos(query_text)
        candidates: list[dict[str, Any]] = []
        for rank, hit in enumerate(hits, start=1):
            candidates.append(
                QACandidate(
                    rank=rank,
                    video_id=hit.video_id,
                    frame_id=hit.frame_id,
                    score=hit.score,
                    answer="",
                    retrieval_score=hit.score,
                    temporal_score=hit.score,
                    evidence={
                        "sources": ["clip", "temporal_proxy"],
                        "answer_status": "not_generated",
                    },
                ).model_dump()
            )
        return candidates[:100]

    def _solve_trake(self, request: QueryRequest) -> list[dict[str, Any]]:
        if not request.events:
            raise ValueError("TRAKE requires at least one event")

        per_event: dict[str, list[RetrievalHit]] = {}
        for event in request.events:
            description = event.description.strip()
            if not description:
                raise ValueError(f"event {event.event_id!r} has empty description")
            per_event[event.event_id] = self.retriever.retrieve(description)

        # Candidate videos are the union of per-event retrievals. A candidate is
        # emitted only when every event has evidence in that same video; this
        # avoids fabricating frames for missing events.
        video_scores: dict[str, list[float]] = defaultdict(list)
        for hits in per_event.values():
            best_by_video: dict[str, float] = {}
            for hit in hits:
                best_by_video[hit.video_id] = max(
                    best_by_video.get(hit.video_id, float("-inf")), hit.score
                )
            for video_id, score in best_by_video.items():
                video_scores[video_id].append(score)

        complete = [
            (video_id, sum(scores) / len(scores))
            for video_id, scores in video_scores.items()
            if len(scores) == len(request.events)
        ]
        complete.sort(key=lambda item: (-item[1], item[0]))

        results: list[dict[str, Any]] = []
        for rank, (video_id, video_score) in enumerate(complete[:100], start=1):
            event_predictions = []
            for event in request.events:
                hits = [hit for hit in per_event[event.event_id] if hit.video_id == video_id]
                selected = select_semantic_keyframes(
                    [self._frame_evidence(hit) for hit in hits],
                    max_candidates=1,
                )
                if not selected:
                    break
                frame = selected[0]
                event_predictions.append(
                    {
                        "event_id": event.event_id,
                        "frame_id": frame.frame_id,
                        "score": frame.score,
                    }
                )
            if len(event_predictions) != len(request.events):
                continue
            results.append(
                TRAKECandidate(
                    rank=rank,
                    video_id=video_id,
                    events=event_predictions,
                    score=video_score,
                ).model_dump()
            )
        return results

    def _frame_evidence(self, hit: RetrievalHit) -> FrameEvidence:
        timestamp = None
        try:
            frame = self.retriever.datastore.get_frame(hit.video_id, hit.frame_id)
            if frame is not None:
                timestamp = frame.timestamp
        except Exception:
            # Temporal metadata is auxiliary; retrieval must remain usable when
            # an older or partial SQLite package lacks the frame table.
            timestamp = None
        return FrameEvidence(
            video_id=hit.video_id,
            frame_id=hit.frame_id,
            timestamp=timestamp,
            retrieval_score=hit.score,
        )

    @staticmethod
    def _infer_task(request: QueryRequest) -> str:
        if request.events:
            return "TRAKE"
        if request.question:
            return "QA"
        return "KIS"

    @staticmethod
    def _build_query_text(request: QueryRequest, task: str) -> str:
        parts: list[str] = []
        if request.description:
            parts.append(request.description.strip())
        if request.raw_text:
            parts.append(request.raw_text.strip())
        if task == "QA" and request.question:
            parts.append(request.question.strip())
        text = " ".join(part for part in parts if part)
        if not text:
            raise ValueError("query request contains no searchable text")
        return text
