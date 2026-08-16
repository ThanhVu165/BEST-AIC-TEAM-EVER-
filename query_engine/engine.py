"""Query Engine orchestration for KIS, QA and TRAKE."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from schemas import Candidate, QACandidate, QueryRequest, SearchResponse, TRAKECandidate

from .answering import AnswerEvidence, AnswerExtractor, UnavailableAnswerExtractor
from .interfaces import QueryEngine
from .retrieval import ClipCandidateRetriever, RetrievalHit
from .temporal import FrameEvidence, select_ordered_event_frames, select_semantic_keyframes


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
        by_keyframe = {
            (hit.video_id, hit.keyframe_n): hit
            for hit in hits
            if hit.keyframe_n is not None
        }
        results: list[dict[str, Any]] = []
        for item in selected:
            hit = by_keyframe.get((item.video_id, item.keyframe_n))
            if hit is None:
                continue
            results.append(
                Candidate(
                    rank=len(results) + 1,
                    video_id=item.video_id,
                    frame_id=item.frame_id,
                    score=item.score,
                    retrieval_score=hit.retrieval_score,
                    temporal_score=item.score,
                    evidence=self._ranking_evidence(hit),
                ).model_dump()
            )
        return results

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

        # Retrieve independently for each event, but keep the full frame pool.
        # The final video ranking is then based on all events jointly rather
        # than on the first event alone.
        per_event: dict[str, list[RetrievalHit]] = {}
        for event in request.events:
            description = event.description.strip()
            if not description:
                raise ValueError(f"event {event.event_id!r} has empty description")
            per_event[event.event_id] = self.retriever.retrieve(description)

        video_ids: set[str] = set()
        for hits in per_event.values():
            video_ids.update(hit.video_id for hit in hits)

        scored_videos: list[tuple[str, float, dict[str, list[RetrievalHit]]]] = []
        for video_id in video_ids:
            event_hits: dict[str, list[RetrievalHit]] = {}
            best_scores: list[float] = []
            complete = True
            for event in request.events:
                hits = [hit for hit in per_event[event.event_id] if hit.video_id == video_id]
                if not hits:
                    complete = False
                    break
                event_hits[event.event_id] = hits
                best_scores.append(max(hit.score for hit in hits))
            if complete:
                # Mean evidence rewards videos that explain every event while
                # avoiding domination by one exceptionally strong event.
                scored_videos.append((video_id, sum(best_scores) / len(best_scores), event_hits))

        scored_videos.sort(key=lambda item: (-item[1], item[0]))

        results: list[dict[str, Any]] = []
        for video_id, video_score, event_hits in scored_videos[:100]:
            ordered_inputs = [
                [self._frame_evidence(hit) for hit in event_hits[event.event_id]]
                for event in request.events
            ]
            selected = select_ordered_event_frames(
                ordered_inputs,
                max_candidates_per_event=100,
                allow_same_frame=True,
            )
            if len(selected) != len(request.events):
                continue

            event_predictions = [
                {
                    "event_id": event.event_id,
                    "frame_id": selected[idx].frame_id,
                    "score": selected[idx].score,
                }
                for idx, event in enumerate(request.events)
            ]
            results.append(
                TRAKECandidate(
                    rank=len(results) + 1,
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
            "keyframe_n": hit.keyframe_n,
        }

    def _frame_record(self, hit: RetrievalHit):
        getter = getattr(self.retriever.datastore, "get_frame", None)
        if getter is None or hit.keyframe_n is None:
            return None
        return getter(hit.video_id, hit.keyframe_n)

    def _frame_evidence(self, hit: RetrievalHit) -> FrameEvidence:
        frame = self._frame_record(hit)
        return FrameEvidence(
            video_id=hit.video_id,
            frame_id=hit.frame_id,
            keyframe_n=hit.keyframe_n,
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
