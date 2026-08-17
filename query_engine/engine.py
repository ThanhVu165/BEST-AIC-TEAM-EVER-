"""Query Engine orchestration for KIS, QA and TRAKE."""
from __future__ import annotations

from typing import Any

from schemas import Candidate, QACandidate, QueryRequest, SearchResponse, TRAKECandidate

from .answering import AnswerEvidence, AnswerExtractor, UnavailableAnswerExtractor
from .interfaces import QueryEngine
from .query_understanding import QuerySpec, understand_query
from .ranking import RankingEvidence, diversify_candidates, rerank_candidates
from .retrieval import ClipCandidateRetriever, RetrievalHit
from .temporal import (
    FrameEvidence,
    fine_localize_source_frames,
    select_ordered_event_frames,
    select_semantic_keyframes,
)


class BaselineQueryEngine(QueryEngine):
    """Canonical pipeline with retrieval, reranking, temporal and task stages."""

    def __init__(
        self,
        retriever: ClipCandidateRetriever,
        answer_extractor: AnswerExtractor | None = None,
        *,
        image_encoder: Any | None = None,
        final_limit: int = 100,
        max_kis_candidates_per_video: int = 10,
        fine_temporal_anchors: int = 20,
        fine_temporal_radius: int = 16,
        fine_temporal_video_limit: int = 10,
    ) -> None:
        if final_limit <= 0:
            raise ValueError("final_limit must be > 0")
        if max_kis_candidates_per_video <= 0:
            raise ValueError("max_kis_candidates_per_video must be > 0")
        if fine_temporal_anchors <= 0 or fine_temporal_radius < 0 or fine_temporal_video_limit <= 0:
            raise ValueError("invalid fine temporal configuration")
        self.retriever = retriever
        self.answer_extractor = answer_extractor or UnavailableAnswerExtractor()
        self.image_encoder = image_encoder
        self.final_limit = final_limit
        self.max_kis_candidates_per_video = max_kis_candidates_per_video
        self.fine_temporal_anchors = fine_temporal_anchors
        self.fine_temporal_radius = fine_temporal_radius
        self.fine_temporal_video_limit = fine_temporal_video_limit

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
            candidates=candidates[: self.final_limit],
        )

    def _solve_kis(self, spec: QuerySpec) -> list[dict[str, Any]]:
        hits = self.retriever.retrieve(spec)
        sparse = select_semantic_keyframes(
            [self._frame_evidence(hit) for hit in hits],
            max_candidates=max(self.final_limit * 5, 100),
        )
        temporal = self._fine_localize(sparse[: self.fine_temporal_anchors], spec.text)
        selected = temporal or sparse

        hit_by_exact_frame = {(hit.video_id, hit.frame_id): hit for hit in hits}
        best_hit_by_video: dict[str, RetrievalHit] = {}
        for hit in hits:
            previous = best_hit_by_video.get(hit.video_id)
            if previous is None or hit.score > previous.score:
                best_hit_by_video[hit.video_id] = hit

        ranking_inputs: list[RankingEvidence] = []
        for item in selected:
            hit = hit_by_exact_frame.get((item.video_id, item.frame_id)) or best_hit_by_video.get(item.video_id)
            if hit is None:
                continue
            ranking_inputs.append(
                RankingEvidence(
                    video_id=item.video_id,
                    frame_id=item.frame_id,
                    retrieval_score=hit.retrieval_score if hit.retrieval_score is not None else hit.score,
                    object_score=hit.object_score,
                    metadata_score=hit.metadata_score,
                    ocr_score=hit.ocr_score,
                    asr_score=hit.asr_score,
                    temporal_score=item.score,
                    semantic_score=0.0,
                    sources=hit.sources,
                )
            )

        ranked = rerank_candidates(ranking_inputs, limit=max(self.final_limit * 5, 100))
        ranked = diversify_candidates(
            ranked,
            limit=self.final_limit,
            max_per_video=self.max_kis_candidates_per_video,
        )
        temporal_lookup = {(item.video_id, item.frame_id): item for item in selected}

        results: list[dict[str, Any]] = []
        for ranking in ranked:
            hit = hit_by_exact_frame.get((ranking.video_id, ranking.frame_id)) or best_hit_by_video.get(ranking.video_id)
            if hit is None:
                continue
            temporal_item = temporal_lookup.get((ranking.video_id, ranking.frame_id))
            evidence = {
                "sources": list(hit.sources),
                "clip_score": hit.retrieval_score,
                "object_score": hit.object_score,
                "metadata_score": hit.metadata_score,
                "ocr_score": hit.ocr_score,
                "asr_score": hit.asr_score,
                "temporal_score": temporal_item.score if temporal_item is not None else 0.0,
                "rerank_score": ranking.fused_score,
                "keyframe_n": hit.keyframe_n if temporal_item is None else temporal_item.keyframe_n,
                "fine_temporal": temporal_item is not None and temporal_item.frame_id != hit.frame_id,
            }
            results.append(
                Candidate(
                    rank=len(results) + 1,
                    video_id=ranking.video_id,
                    frame_id=ranking.frame_id,
                    score=ranking.fused_score,
                    retrieval_score=hit.retrieval_score,
                    temporal_score=temporal_item.score if temporal_item is not None else 0.0,
                    rerank_score=ranking.fused_score,
                    evidence=evidence,
                ).model_dump()
            )
        return results

    def _solve_qa(self, spec: QuerySpec) -> list[dict[str, Any]]:
        hits = self.retriever.retrieve_videos(spec)
        sparse = select_semantic_keyframes(
            [self._frame_evidence(hit) for hit in hits],
            max_candidates=min(max(self.final_limit, 20), len(hits)),
        )
        temporal = self._fine_localize(sparse[: self.fine_temporal_anchors], spec.text)
        selected = temporal or sparse
        hit_by_exact_frame = {(hit.video_id, hit.frame_id): hit for hit in hits}
        best_hit_by_video: dict[str, RetrievalHit] = {}
        for hit in hits:
            previous = best_hit_by_video.get(hit.video_id)
            if previous is None or hit.score > previous.score:
                best_hit_by_video[hit.video_id] = hit

        candidates: list[dict[str, Any]] = []
        reader = getattr(self.retriever.datastore, "read_source_frame", None)
        for item in selected:
            hit = hit_by_exact_frame.get((item.video_id, item.frame_id)) or best_hit_by_video.get(item.video_id)
            if hit is None:
                continue
            frame = self._keyframe_record(hit) if item.frame_id == hit.frame_id else None
            image = None
            if frame is None and reader is not None:
                image = reader(item.video_id, item.frame_id)

            answer = ""
            status = "evidence_unavailable"
            confidence = None
            if spec.question and (frame is not None or image is not None):
                result = self.answer_extractor.answer(
                    AnswerEvidence(
                        video_id=item.video_id,
                        frame_id=item.frame_id,
                        frame_path=frame.path if frame is not None else None,
                        question=spec.question,
                        image=image,
                    )
                )
                answer = result.answer
                status = result.status
                confidence = result.confidence

            rerank_score = self._qa_score(hit, item.score, status)
            evidence = {
                "sources": list(hit.sources),
                "clip_score": hit.retrieval_score,
                "object_score": hit.object_score,
                "metadata_score": hit.metadata_score,
                "ocr_score": hit.ocr_score,
                "asr_score": hit.asr_score,
                "temporal_score": item.score,
                "answer_status": status,
                "rerank_score": rerank_score,
                "fine_temporal": item.frame_id != hit.frame_id,
            }
            if confidence is not None:
                evidence["answer_confidence"] = confidence
            candidates.append(
                QACandidate(
                    rank=0,
                    video_id=item.video_id,
                    frame_id=item.frame_id,
                    score=rerank_score,
                    answer=answer,
                    retrieval_score=hit.retrieval_score,
                    temporal_score=item.score,
                    rerank_score=rerank_score,
                    evidence=evidence,
                ).model_dump()
            )

        candidates.sort(key=lambda item: (-float(item["score"]), item["video_id"], int(item["frame_id"])))
        for rank, candidate in enumerate(candidates[: self.final_limit], start=1):
            candidate["rank"] = rank
        return candidates[: self.final_limit]

    @staticmethod
    def _qa_score(hit: RetrievalHit, temporal_score: float, answer_status: str) -> float:
        retrieval = hit.retrieval_score if hit.retrieval_score is not None else hit.score
        answer_bonus = 0.05 if answer_status == "completed" else 0.0
        return 0.82 * retrieval + 0.10 * temporal_score + 0.03 * hit.metadata_score + answer_bonus

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
        for video_id, _, event_hits in scored_videos[: self.final_limit * 2]:
            ordered_inputs: list[list[FrameEvidence]] = []
            for event in spec.events:
                event_evidence = [self._frame_evidence(hit) for hit in event_hits[event.event_id]]
                if self.image_encoder is not None and len(ordered_inputs) < len(spec.events):
                    fine = self._fine_localize(event_evidence[: self.fine_temporal_anchors], event.description)
                    event_evidence = [
                        FrameEvidence(
                            video_id=item.video_id,
                            frame_id=item.frame_id,
                            keyframe_n=item.keyframe_n,
                            timestamp=item.timestamp,
                            retrieval_score=item.score,
                        )
                        for item in (fine or select_semantic_keyframes(event_evidence, max_candidates=self.fine_temporal_anchors))
                    ]
                ordered_inputs.append(event_evidence)

            selected = select_ordered_event_frames(
                ordered_inputs,
                max_candidates_per_event=self.fine_temporal_anchors,
                allow_same_frame=False,
            )
            if len(selected) != len(spec.events):
                continue

            event_predictions = [
                {"event_id": event.event_id, "frame_id": selected[idx].frame_id, "score": selected[idx].score}
                for idx, event in enumerate(spec.events)
            ]
            aligned_score = sum(item["score"] for item in event_predictions) / len(event_predictions)
            results.append(
                TRAKECandidate(
                    rank=0,
                    video_id=video_id,
                    events=event_predictions,
                    score=aligned_score,
                ).model_dump()
            )

        results.sort(key=lambda item: (-float(item["score"]), item["video_id"]))
        for rank, candidate in enumerate(results[: self.final_limit], start=1):
            candidate["rank"] = rank
        return results[: self.final_limit]

    def _fine_localize(self, anchors: list[Any], query_text: str):
        if not anchors or self.image_encoder is None:
            return []
        reader = getattr(self.retriever.datastore, "read_source_frame", None)
        batch_reader = getattr(self.retriever.datastore, "read_source_frames", None)
        if reader is None and batch_reader is None:
            return []
        return fine_localize_source_frames(
            anchors,
            query_text=query_text,
            reader=self.retriever.datastore,
            image_encoder=self.image_encoder,
            radius=self.fine_temporal_radius,
            max_candidates=self.fine_temporal_anchors,
        )

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
