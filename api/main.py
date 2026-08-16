"""FastAPI boundary between Query Engine and UI.

The default runtime uses MockQueryEngine so all three team members can run an
end-to-end system before real retrieval is ready. Replace the engine factory
without changing API schemas or the Streamlit client.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException

from query_engine import MockQueryEngine
from schemas import QueryRequest, SearchResponse, SubmissionRequest, SubmissionResponse

app = FastAPI(
    title="AIC 2026 Video Retrieval API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

engine = MockQueryEngine()
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
def get_video(video_id: str) -> dict[str, str]:
    # Real DataStore integration will replace this endpoint implementation.
    return {"video_id": video_id, "status": "not_connected"}


@app.get("/api/v1/video/{video_id}/frame/{frame_id}")
def get_frame(video_id: str, frame_id: int) -> dict[str, str | int]:
    # Returning identifiers now keeps UI integration testable before the data
    # package is mounted on the machine running FastAPI.
    return {
        "video_id": video_id,
        "frame_id": frame_id,
        "status": "not_connected",
    }


@app.post("/api/v1/submission", response_model=SubmissionResponse)
def create_submission(request: SubmissionRequest) -> SubmissionResponse:
    missing = [query_id for query_id in request.query_ids if query_id not in _results]
    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"No result stored for query_ids: {missing}",
        )
    return SubmissionResponse(
        status="completed",
        file_name="submission_mock.json",
    )
