"""Stable data-access boundary consumed by Query Engine."""
from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from schemas.contracts import FrameRecord, ObjectRecord, VideoRecord


class DataStore(ABC):
    """Storage contract. Query code never accesses SQLite/FAISS directly."""

    @abstractmethod
    def get_video(self, video_id: str) -> VideoRecord | None:
        raise NotImplementedError

    @abstractmethod
    def get_frame(self, video_id: str, frame_id: int) -> FrameRecord | None:
        raise NotImplementedError

    @abstractmethod
    def get_frames_in_range(
        self, video_id: str, start_frame: int, end_frame: int
    ) -> list[FrameRecord]:
        raise NotImplementedError

    @abstractmethod
    def get_objects(self, video_id: str, frame_id: int) -> ObjectRecord | None:
        raise NotImplementedError

    @abstractmethod
    def search_clip(self, vector: np.ndarray, top_k: int) -> list[dict[str, Any]]:
        raise NotImplementedError


class LocalDataStore(DataStore):
    """SQLite + optional FAISS implementation for a single-machine deployment."""

    def __init__(self, db_path: str | Path, clip_index: Any | None = None):
        self.db_path = Path(db_path)
        self.clip_index = clip_index

    def _connect(self) -> sqlite3.Connection:
        if not self.db_path.is_file():
            raise FileNotFoundError(f"SQLite database not found: {self.db_path}")
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_video(self, video_id: str) -> VideoRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM videos WHERE video_id = ?", (video_id,)
            ).fetchone()
        return VideoRecord(**dict(row)) if row else None

    def get_frame(self, video_id: str, frame_id: int) -> FrameRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM frames WHERE video_id = ? AND frame_id = ?",
                (video_id, frame_id),
            ).fetchone()
        return FrameRecord(**dict(row)) if row else None

    def get_frames_in_range(
        self, video_id: str, start_frame: int, end_frame: int
    ) -> list[FrameRecord]:
        if end_frame < start_frame:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM frames
                   WHERE video_id = ? AND frame_id BETWEEN ? AND ?
                   ORDER BY frame_id""",
                (video_id, start_frame, end_frame),
            ).fetchall()
        return [FrameRecord(**dict(row)) for row in rows]

    def get_objects(self, video_id: str, frame_id: int) -> ObjectRecord | None:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT label, confidence, x1, y1, x2, y2
                   FROM objects WHERE video_id = ? AND frame_id = ?""",
                (video_id, frame_id),
            ).fetchall()
        if not rows:
            return None
        objects = [
            {
                "label": row["label"],
                "confidence": float(row["confidence"]),
                "bbox": [
                    float(row["x1"]),
                    float(row["y1"]),
                    float(row["x2"]),
                    float(row["y2"]),
                ],
            }
            for row in rows
        ]
        return ObjectRecord(video_id=video_id, frame_id=frame_id, objects=objects)

    def search_clip(self, vector: np.ndarray, top_k: int) -> list[dict[str, Any]]:
        if self.clip_index is None:
            raise RuntimeError("CLIP index is not configured")
        if top_k <= 0:
            return []
        return self.clip_index.search(vector, top_k)
