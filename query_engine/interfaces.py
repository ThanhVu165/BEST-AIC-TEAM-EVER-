"""Stable interfaces owned by the Query Engine boundary.

Implementations can evolve without changing consumers in the UI or data layer.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np

from schemas import (
    FrameRecord,
    ObjectRecord,
    QueryRequest,
    SearchResponse,
    VideoRecord,
)


class DataStore(ABC):
    """Read-only access abstraction over the Video Processing data package."""

    @abstractmethod
    def get_video(self, video_id: str) -> VideoRecord:
        raise NotImplementedError

    @abstractmethod
    def get_frame(self, video_id: str, frame_id: int) -> FrameRecord:
        raise NotImplementedError

    @abstractmethod
    def get_frames(self, video_id: str) -> Sequence[FrameRecord]:
        raise NotImplementedError

    @abstractmethod
    def get_frames_in_range(
        self, video_id: str, start_frame: int, end_frame: int
    ) -> Sequence[FrameRecord]:
        raise NotImplementedError

    @abstractmethod
    def get_objects(self, video_id: str, frame_id: int) -> ObjectRecord:
        raise NotImplementedError

    @abstractmethod
    def get_metadata(self, video_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    def search_vector(
        self, index_name: str, vector: np.ndarray, top_k: int
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_embedding(
        self, index_name: str, video_id: str, frame_id: int
    ) -> np.ndarray:
        raise NotImplementedError


class QueryEngine(ABC):
    """Stable inference boundary consumed by the FastAPI application."""

    @abstractmethod
    def search(self, request: QueryRequest) -> SearchResponse:
        raise NotImplementedError
