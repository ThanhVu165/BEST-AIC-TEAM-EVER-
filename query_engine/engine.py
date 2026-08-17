"""Query Engine orchestration for KIS, QA and TRAKE."""
from __future__ import annotations

from typing import Any

from schemas import Candidate, QACandidate, QueryRequest, SearchResponse, TRAKECandidate

from .answering import AnswerEvidence, AnswerExtractor, UnavailableAnswerExtractor
from .interfaces import QueryEngine
from .query_understanding import QuerySpec, understand_query
from .retrieval import ClipCandidateRetriever, RetrievalHit
from .temporal import FrameEvidence, select_ordered_event_frames, select_semantic_keyframes


class BaselineQueryEngine(QueryEngine):
    """Canonical pipeline orchestration with explicit intermediate stages."""

    def __init__(
        self,
        retriever: ClipCandidateRetriever,
        answer_extractor: AnswerExtractor | None = None,
    ) -> None:
        self.retriever = retriever
        self.answer_extractor = answer_extractor or UnavailableAnswerExtractor()

    def search(self, request: QueryRequest) -> SearchResponse:
        spec = understand_query(request)
        try:
            if spec.task == "KIS":
                candidates = self._solve_kis(spec)
            elif spec.task == "QA":
                candidates = self._solve_qa(spec)
            else:
                candidates = self._solve_trake(spec)
        except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
            return SearchResponse(
                query_id=request.query_id,
                task=spec.task,
                status="failed",
                candidates=[],
                error=str(exc),
            )

        return SearchResponse(
            query_id=request.query_id,
            task=spec.task,
            status="completed",
            candidates=candidates[:100],
        )

    def _solve_kis(self, spec: QuerySpec) -> list[dict[str, Any]]:
        hits = self.retriever.retrieve(spec)
        selected = select_semantic_keyframes(
            [self._frame_evidence(hit) for hit in hits],
            max_candidates=100,
        )
        by_keyframe = {
            (hit.video_id, hit.keyframe_n): hit
            for hit in hits
            if hit.keyframe_n is not None
        }
        by_frame = {(hit.video_id, hit.frame_id): hit for hit in hits}

        results: list[dict[str, Any]] = []
        for item in selected:
            hit = by_keyframe.get((item.video_id, item.keyframe_n)) if item.keyframe_n is not None else None
            if hit is None:
                hit = by_frame.get((item.video_id, item.frame_id))
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
                    rerank_score=hit.score,
                    evidence={
                        "sources": list(hit.sources),
                        "clip_score": hit.retrieval_score,
                        "object_score": hit.object_score,
                        "metadata_score": hit.metadata_score,
                        "ocr_score": hit.ocr_score,
                        "asr_score": hit.asr_score,
                        "fused_score": hit.score,
                        "keyframe_n": hit.keyframe_n,
                    },
                ).model_dump()
            )
        return results

    def _solve_qa(self, spec: QuerySpec) -> list[dict[str, Any]]:
        hits = self.retriever.retrieve_videos(spec)
        candidates: list[dict[str, Any]] = []
        for rank, hit in enumerate(hits, start=1):
            frame = self._keyframe_record(hit)
            answer = ""
            status = "evidence_unavailable"
            confidence = None
            if frame is not None and spec.question:
                result = self.answer_extractor.answer(
                    AnswerEvidence(
                        video_id=hit.video_id,
                        frame_id=hit.frame_id,
                        frame_path=frame.path,
                        question=spec.question,
                    )
                )
                answer = result.answer
                status = result.status
                confidence = result.confidence

            evidence = {
                "sources": list(hit.sources),
                "clip_score": hit.retrieval_score,
                "object_score": hit.object_score,
                "metadata_score": hit.metadata_score,
                "ocr_score": hit.ocr_score,
                "asr_score": hit.asr_score,
                "fused_score": hit.score,
                "answer_status": status,
            }
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
                    rerank_score=hit.score,
                    evidence=evidence,
                ).model_dump()
            )
        return candidates[:100]

    def _solve_trake(self, spec: QuerySpec) -> list[dict[str, Any]]:
        if not spec.events:
            raise ValueError("TRAKE requires at least one event")

        per_event: dict[str, list[RetrievalHit]] = {
            event.event_id: self.retriever.retrieve(event.description)
            for event in spec.events
        }

        video_ids: set[str] = set()
        for hits in per_event.values():
            video_ids.update(hit.video_id for hit in hits)

        scored_videos: list[tuple[str, float, dict[str, list[RetrievalHit]]]] = []
        for video_id in video_ids:
            event_hits: dict[str, list[RetrievalHit]] = {}
            best_scores: list[float] = []
            for event in spec.events:
                hits = [hit for hit in per_event[event.event_id] if hit.video_id == video_id]
                if not hits:
                    break
                event_hits[event.event_id] = hits
                best_scores.append(max(hit.score for hit in hits))
            else:
                scored_videos.append((video_id, sum(best_scores) / len(best_scores), event_hits))

        scored_videos.sort(key=lambda item: (-item[1], item[0]))
        results: list[dict[str, Any]] = []
        for video_id, _, event_hits in scored_videos[:100]:
            ordered_inputs = [
                [self._frame_evidence(hit) for hit in event_hits[event.event_id]]
                for event in spec.events
            ]
            selected = select_ordered_event_frames(
                ordered_inputs,
                max_candidates_per_event=100,
                allow_same_frame=True,
            )
            if len(selected) != len(spec.events):
                continue

            event_predictions = [
                {
                    "event_id": event.event_id,
                    "frame_id": selected[idx].frame_id,
                    "score": selected[idx].score,
                }
                for idx, event in enumerate(spec.events)
            ]
            aligned_score = sum(item["score"] for item in event_predictions) / len(event_predictions)
            results.append(
                TRAKECandidate(
                    rank=len(results) + 1,
                    video_id=video_id,
                    events=event_predictions,
                    score=aligned_score,
                ).model_dump()
            )
        return results

    def _keyframe_record(self, hit: RetrievalHit):
        getter = getattr(self.retriever.datastore, "get_frame", None)
        if getter is None or hit.keyframe_n is None:
            return None
        return getter(hit.video_id, hit.keyframe_n)

    def _frame_evidence(self, hit: RetrievalHit) -> FrameEvidence:
        frame = self._keyframe_record(hit)
        return FrameEvidence(
            video_id=hit.video_id,
            frame_id=hit.frame_id,
            keyframe_n=hit.keyframe_n,
            timestamp=frame.timestamp if frame is not None else None,
            retrieval_score=hit.score,
        )
