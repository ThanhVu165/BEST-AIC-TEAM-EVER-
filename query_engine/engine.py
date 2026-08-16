"""Query Engine orchestration for KIS, QA and TRAKE."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from schemas import Candidate, QACandidate, QueryRequest, SearchResponse, TRAKECandidate

from .answering import AnswerEvidence, AnswerExtractor, UnavailableAnswerExtractor
from .interfaces import QueryEngine
from .retrieval import ClipCandidateRetriever, RetrievalHit
from .temporal import FrameEvidence, select_semantic_keyframes


class BaselineQueryEngine(QueryEngine):
    """Run CLIP retrieval and deterministic evidence-aware task solvers."""

    def __init__(
        self,
        retriever: ClipCandidateRetriever,
        answer_extractor: AnswerExtractor | None = None,
    ) -> None:
        self.retriever = retriever
        self.answer_extractor = answer_extractor or UnavailableAnswerExtractor()

    def search(self, request: QueryRequest) -> SearchResponse:
        task = request.task or self._infer_task(request)
        try:
            if task == "KIS":
                candidates = self._solve_kis(request)
            elif task == "QA":
                candidates = self._solve_qa(request)
            else:
                candidates = self._solve_trake(request)
        except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
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
        selected = select_semantic_keyframes(
            [self._frame_evidence(hit) for hit in hits], max_candidates=100
        )
        by_frame = {(hit.video_id, hit.frame_id): hit for hit in hits}
        return [
            Candidate(
                rank=item.rank,
                video_id=item.video_id,
                frame_id=item.frame_id,
                score=item.score,
                retrieval_score=by_frame[(item.video_id, item.frame_id)].retrieval_score,
                temporal_score=item.score,
                evidence=self._ranking_evidence(
                    by_frame[(item.video_id, item.frame_id)]
                ),
            ).model_dump()
            for item in selected
        ]

    def _solve_qa(self, request: QueryRequest) -> list[dict[str, Any]]:
        query_text = self._build_query_text(request, "QA")
        hits = self.retriever.retrieve_videos(query_text)
        candidates: list[dict[str, Any]] = []
        for rank, hit in enumerate(hits, start=1):
            frame = self._frame_record(hit)
            answer = ""
            status = "evidence_unavailable"
            confidence = None
            if frame is not None:
                result = self.answer_extractor.answer(
                    AnswerEvidence(
                        video_id=hit.video_id,
                        frame_id=hit.frame_id,
                        frame_path=frame.path,
                        question=request.question or query_text,
                    )
                )
                answer = result.answer
                status = result.status
                confidence = result.confidence
            evidence = self._ranking_evidence(hit)
            evidence["answer_status"] = status
            if confidence is not None:
                evidence["answer_confidence"] = confidence
            candidates.append(
                QACandidate(
                    rank=rank,
                    video_id=hit.video_id,
                    frame_id=hit.frame_id,
                    score=hit.score,
                    answer=answer,
                    retrieval_score=hit.retrieval_score,
                    temporal_score=hit.score,
                    evidence=evidence,
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
                hits = [
                    hit
                    for hit in per_event[event.event_id]
                    if hit.video_id == video_id
                ]
                selected = select_semantic_keyframes(
                    [self._frame_evidence(hit) for hit in hits], max_candidates=1
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

    @staticmethod
    def _ranking_evidence(hit: RetrievalHit) -> dict[str, Any]:
        return {
            "sources": list(hit.sources),
            "clip_score": hit.retrieval_score,
            "object_score": hit.object_score,
            "fused_score": hit.score,
        }

    def _frame_record(self, hit: RetrievalHit):
        getter = getattr(self.retriever.datastore, "get_frame", None)
        if getter is None:
            return None
        return getter(hit.video_id, hit.frame_id)

    def _frame_evidence(self, hit: RetrievalHit) -> FrameEvidence:
        frame = self._frame_record(hit)
        return FrameEvidence(
            video_id=hit.video_id,
            frame_id=hit.frame_id,
            timestamp=frame.timestamp if frame is not None else None,
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
