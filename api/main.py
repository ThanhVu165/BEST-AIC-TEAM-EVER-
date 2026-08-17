"""FastAPI boundary between Query Engine and UI.

The API can run with the deterministic mock engine for UI/contract tests or
with the real CLIP/FAISS runtime when local Batch 1 artifacts are configured
through environment variables. No dataset or model path is hard-coded.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException

from query_engine import BaselineQueryEngine, MockQueryEngine
from query_engine.runtime import build_clip_baseline_engine
from schemas import QueryRequest, SearchResponse, SubmissionRequest, SubmissionResponse

app = FastAPI(
    title="AIC 2026 Video Retrieval API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


def _build_engine() -> BaselineQueryEngine | MockQueryEngine:
    mode = os.getenv("AIC_ENGINE", "mock").strip().lower()
    if mode == "mock":
        return MockQueryEngine()
    if mode != "clip":
        raise ValueError("AIC_ENGINE must be 'mock' or 'clip'")

    db_path = os.getenv("AIC_DB_PATH")
    index_path = os.getenv("AIC_CLIP_INDEX")
    mapping_path = os.getenv("AIC_CLIP_MAPPING")
    if not all((db_path, index_path, mapping_path)):
        raise RuntimeError(
            "AIC_ENGINE=clip requires AIC_DB_PATH, AIC_CLIP_INDEX, and AIC_CLIP_MAPPING"
        )

    return build_clip_baseline_engine(
        db_path=Path(db_path),
        index_path=Path(index_path),
        mapping_path=Path(mapping_path),
        model_name=os.getenv("AIC_CLIP_MODEL", "openai/clip-vit-base-patch32"),
        device=os.getenv("AIC_DEVICE", "auto"),
        vlm_model_name=os.getenv("AIC_VLM_MODEL") or None,
        vlm_device=os.getenv("AIC_VLM_DEVICE") or None,
        fine_temporal_anchors=int(os.getenv("AIC_FINE_TEMPORAL_ANCHORS", "20")),
        fine_temporal_radius=int(os.getenv("AIC_FINE_TEMPORAL_RADIUS", "16")),
    )


engine = _build_engine()
_results: dict[str, SearchResponse] = {}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "engine": engine.__class__.__name__}


@app.post("/api/v1/search", response_model=SearchResponse)
def search(request: QueryRequest) -> SearchResponse:
    result = engine.search(request)
    _results[request.query_id] = result
    return result


@app.get("/api/v1/result/{query_id}", response_model=SearchResponse)
def get_result(query_id: str) -> SearchResponse:
    result = _results.get(query_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No result stored for {query_id}")
    return result


@app.get("/api/v1/video/{video_id}")
def get_video(video_id: str) -> dict[str, object]:
    datastore = getattr(getattr(engine, "retriever", None), "datastore", None)
    if datastore is None:
        return {"video_id": video_id, "status": "not_connected"}
    record = datastore.get_video(video_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Video not found: {video_id}")
    return record.model_dump()


@app.get("/api/v1/video/{video_id}/frame/{frame_id}")
def get_frame(video_id: str, frame_id: int) -> dict[str, object]:
    datastore = getattr(getattr(engine, "retriever", None), "datastore", None)
    if datastore is None:
        return {"video_id": video_id, "frame_id": frame_id, "status": "not_connected"}

    getter = getattr(datastore, "get_frame_by_id", None)
    record = getter(video_id, frame_id) if getter is not None else None
    if record is not None:
        return record.model_dump()

    # Source-frame IDs are not necessarily keyframes. Report source availability
    # without forcing the API to serialize binary image data.
    reader = getattr(datastore, "read_source_frame", None)
    if reader is not None and reader(video_id, frame_id) is not None:
        return {
            "video_id": video_id,
            "frame_id": frame_id,
            "is_keyframe": False,
            "status": "source_frame_available",
        }
    raise HTTPException(status_code=404, detail=f"Source frame not found: {video_id}/{frame_id}")


@app.post("/api/v1/submission", response_model=SubmissionResponse)
def create_submission(request: SubmissionRequest) -> SubmissionResponse:
    missing = [query_id for query_id in request.query_ids if query_id not in _results]
    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"No result stored for query_ids: {missing}",
        )
    return SubmissionResponse(status="completed", file_name="submission_mock.json")
