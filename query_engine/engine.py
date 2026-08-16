"""Baseline Query Engine orchestration.

This is intentionally a small, deterministic baseline. It establishes the
online inference boundary without pretending to solve temporal localization
or Q&A/TRAKE semantics before those components exist.
"""
from __future__ import annotations

from schemas import Candidate, QACandidate, QueryRequest, SearchResponse, TRAKECandidate

from .interfaces import QueryEngine
from .retrieval import ClipCandidateRetriever


class BaselineQueryEngine(QueryEngine):
    """Run CLIP retrieval and produce contract-valid ranked candidates.

    KIS currently uses the retrieved best frame directly. QA and TRAKE keep the
    same retrieved video/frame hypotheses but expose explicit baseline limits:
    answer extraction and fine temporal alignment are not silently fabricated.
    """

    def __init__(self, retriever: ClipCandidateRetriever) -> None:
        self.retriever = retriever

    def search(self, request: QueryRequest) -> SearchResponse:
        task = request.task or self._infer_task(request)
        query_text = self._build_query_text(request, task)

        try:
            hits = self.retriever.retrieve(query_text)
        except Exception as exc:  # API boundary converts this into failed status.
            return SearchResponse(
                query_id=request.query_id,
                task=task,
                status="failed",
                candidates=[],
                error=str(exc),
            )

        if task == "KIS":
            candidates = [
                Candidate(
                    rank=rank,
                    video_id=hit.video_id,
                    frame_id=hit.frame_id,
                    score=hit.score,
                    retrieval_score=hit.score,
                    evidence={"sources": list(hit.sources)},
                ).model_dump()
                for rank, hit in enumerate(hits, start=1)
            ]
        elif task == "QA":
            # Do not invent an answer. The candidate evidence is still useful
            # for integration and for a later dedicated answer extractor.
            answer = ""
            candidates = [
                QACandidate(
                    rank=rank,
                    video_id=hit.video_id,
                    frame_id=hit.frame_id,
                    score=hit.score,
                    answer=answer,
                    retrieval_score=hit.score,
                    evidence={
                        "sources": list(hit.sources),
                        "answer_status": "not_generated",
                    },
                ).model_dump()
                for rank, hit in enumerate(hits, start=1)
            ]
        else:
            # TRAKE alignment is not fabricated. The retrieved video hypotheses
            # are exposed with no event predictions until a temporal aligner is
            # plugged in.
            candidates = [
                TRAKECandidate(
                    rank=rank,
                    video_id=hit.video_id,
                    events=[],
                    score=hit.score,
                ).model_dump()
                for rank, hit in enumerate(hits, start=1)
            ]

        return SearchResponse(
            query_id=request.query_id,
            task=task,
            status="completed",
            candidates=candidates[:100],
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
        if task == "TRAKE":
            parts.extend(
                f"{event.event_id}: {event.description.strip()}"
                for event in request.events
                if event.description.strip()
            )
        text = " ".join(part for part in parts if part)
        if not text:
            raise ValueError("query request contains no searchable text")
        return text
