"""Shared integration contracts for the AIC 2026 system.

These Pydantic models are intentionally model-agnostic. They are the stable
boundary between Video Processing, Query Engine, and UI/API.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

TaskType = Literal["KIS", "QA", "TRAKE"]


class VideoRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video_id: str
    path: str
    fps: float
    width: int
    height: int
    duration: float
    total_frames: int
    batch_id: str | None = None
    metadata_available: bool = False


class FrameRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video_id: str
    keyframe_n: int
    frame_id: int
    timestamp: float
    path: str
    is_keyframe: bool = False


class ObjectDetection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    confidence: float
    bbox: list[float] = Field(min_length=4, max_length=4)


class ObjectRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video_id: str
    frame_id: int
    objects: list[ObjectDetection] = Field(default_factory=list)


class OCRRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video_id: str
    frame_id: int
    text: str
    confidence: float | None = None


class ASRSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video_id: str
    start_time: float
    end_time: float
    text: str


class QueryEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    description: str


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_id: str
    task: TaskType | None = None
    description: str | None = None
    question: str | None = None
    events: list[QueryEvent] = Field(default_factory=list)
    raw_text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CandidateEvidence(BaseModel):
    model_config = ConfigDict(extra="allow")

    sources: list[str] = Field(default_factory=list)


class Candidate(BaseModel):
    model_config = ConfigDict(extra="allow")

    rank: int
    video_id: str
    frame_id: int | None = None
    score: float
    retrieval_score: float | None = None
    temporal_score: float | None = None
    rerank_score: float | None = None
    evidence: dict[str, Any] | CandidateEvidence = Field(default_factory=dict)


class KISResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_id: str
    task: Literal["KIS"]
    candidates: list[Candidate] = Field(default_factory=list, max_length=100)


class QACandidate(Candidate):
    answer: str


class QAResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_id: str
    task: Literal["QA"]
    candidates: list[QACandidate] = Field(default_factory=list, max_length=100)


class TRAKEEventPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    frame_id: int
    score: float


class TRAKECandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int
    video_id: str
    events: list[TRAKEEventPrediction] = Field(default_factory=list)
    score: float


class TRAKEResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_id: str
    task: Literal["TRAKE"]
    candidates: list[TRAKECandidate] = Field(default_factory=list, max_length=100)


class SearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_id: str
    task: TaskType
    status: Literal["queued", "running", "completed", "failed"]
    candidates: list[dict[str, Any]] = Field(default_factory=list, max_length=100)
    error: str | None = None


class SubmissionRequest(BaseModel):
    query_ids: list[str] = Field(min_length=1)


class SubmissionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["completed", "failed"]
    file_name: str | None = None
    error: str | None = None
