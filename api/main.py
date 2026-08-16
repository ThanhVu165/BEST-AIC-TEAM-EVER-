"""FastAPI integration skeleton.

The actual Query Engine is intentionally injected later. UI and Query Engine
can develop independently against this stable API surface.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException

from schemas import QueryRequest, SearchResponse, SubmissionRequest, SubmissionResponse

app = FastAPI(
    title="AIC 2026 Video Retrieval API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/search", response_model=SearchResponse)
def search(request: QueryRequest) -> SearchResponse:
    if request.task is None:
        raise HTTPException(status_code=422, detail="task must be provided in API v1")

    raise HTTPException(
        status_code=501,
        detail="Query Engine is not connected yet. Use mock mode in UI until integration is ready.",
    )


@app.get("/api/v1/result/{query_id}")
def get_result(query_id: str) -> dict[str, str]:
    raise HTTPException(status_code=404, detail=f"No result stored for {query_id}")


@app.get("/api/v1/video/{video_id}")
def get_video(video_id: str) -> dict[str, str]:
    raise HTTPException(status_code=501, detail="Video data service is not connected yet")


@app.get("/api/v1/video/{video_id}/frame/{frame_id}")
def get_frame(video_id: str, frame_id: int) -> dict[str, str | int]:
    raise HTTPException(status_code=501, detail="Video data service is not connected yet")


@app.post("/api/v1/submission", response_model=SubmissionResponse)
def create_submission(request: SubmissionRequest) -> SubmissionResponse:
    if not request.query_ids:
        raise HTTPException(status_code=422, detail="query_ids must not be empty")
    raise HTTPException(status_code=501, detail="Submission service is not connected yet")
