"""Stable data-access boundary consumed by Query Engine."""
from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from schemas.contracts import ASRSegment, FrameRecord, OCRRecord, ObjectRecord, VideoRecord


class DataStore(ABC):
    """Storage contract. Query code never accesses SQLite/FAISS directly."""

    @abstractmethod
    def get_video(self, video_id: str) -> VideoRecord | None:
        raise NotImplementedError

    @abstractmethod
    def get_frame(self, video_id: str, keyframe_n: int) -> FrameRecord | None:
        raise NotImplementedError

    @abstractmethod
    def get_frame_by_id(self, video_id: str, frame_id: int) -> FrameRecord | None:
        raise NotImplementedError

    @abstractmethod
    def get_frames_in_range(
        self, video_id: str, start_frame: int, end_frame: int
    ) -> list[FrameRecord]:
        raise NotImplementedError

    @abstractmethod
    def get_objects(self, video_id: str, keyframe_n: int) -> ObjectRecord | None:
        raise NotImplementedError

    @abstractmethod
    def search_clip(self, vector: np.ndarray, top_k: int) -> list[dict[str, Any]]:
        raise NotImplementedError

    def get_ocr(self, video_id: str, keyframe_n: int | None = None) -> list[OCRRecord]:
        return []

    def get_asr(self, video_id: str) -> list[ASRSegment]:
        return []

    def get_metadata(self, video_id: str) -> dict[str, Any] | None:
        return None

    def read_source_frame(self, video_id: str, frame_id: int) -> Any | None:
        """Return one original-video RGB frame, or ``None`` when unavailable."""
        return None

    def read_source_frames(self, video_id: str, frame_ids: list[int]) -> dict[int, Any]:
        """Optional batch source-frame reader. Default falls back to single reads."""
        return {
            frame_id: frame
            for frame_id in frame_ids
            if (frame := self.read_source_frame(video_id, frame_id)) is not None
        }


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

    def get_frame(self, video_id: str, keyframe_n: int) -> FrameRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM frames WHERE video_id = ? AND keyframe_n = ?",
                (video_id, keyframe_n),
            ).fetchone()
        return FrameRecord(**dict(row)) if row else None

    def get_frame_by_id(self, video_id: str, frame_id: int) -> FrameRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT * FROM frames
                   WHERE video_id = ? AND frame_id = ?
                   ORDER BY keyframe_n LIMIT 1""",
                (video_id, frame_id),
            ).fetchone()
        return FrameRecord(**dict(row)) if row else None

    def get_frames_in_range(self, video_id: str, start_frame: int, end_frame: int) -> list[FrameRecord]:
        if end_frame < start_frame:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM frames
                   WHERE video_id = ? AND frame_id BETWEEN ? AND ?
                   ORDER BY frame_id, keyframe_n""",
                (video_id, start_frame, end_frame),
            ).fetchall()
        return [FrameRecord(**dict(row)) for row in rows]

    def get_objects(self, video_id: str, keyframe_n: int) -> ObjectRecord | None:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT frame_id, label, confidence, x1, y1, x2, y2
                   FROM objects WHERE video_id = ? AND keyframe_n = ?""",
                (video_id, keyframe_n),
            ).fetchall()
        if not rows:
            return None
        frame_id = int(rows[0]["frame_id"])
        objects = [
            {
                "label": row["label"],
                "confidence": float(row["confidence"]),
                "bbox": [float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])],
            }
            for row in rows
        ]
        return ObjectRecord(video_id=video_id, frame_id=frame_id, objects=objects)

    def get_ocr(self, video_id: str, keyframe_n: int | None = None) -> list[OCRRecord]:
        with self._connect() as conn:
            if keyframe_n is None:
                rows = conn.execute(
                    """SELECT video_id, frame_id, text, confidence
                       FROM ocr WHERE video_id = ? ORDER BY frame_id, keyframe_n""",
                    (video_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT video_id, frame_id, text, confidence
                       FROM ocr WHERE video_id = ? AND keyframe_n = ?
                       ORDER BY frame_id""",
                    (video_id, keyframe_n),
                ).fetchall()
        return [OCRRecord(**dict(row)) for row in rows]

    def get_asr(self, video_id: str) -> list[ASRSegment]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT video_id, start_time, end_time, text
                   FROM asr_segments WHERE video_id = ? ORDER BY start_time""",
                (video_id,),
            ).fetchall()
        return [ASRSegment(**dict(row)) for row in rows]

    def get_metadata(self, video_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT raw_json FROM metadata WHERE video_id = ?", (video_id,)).fetchone()
        if row is None:
            return None
        try:
            payload = json.loads(str(row["raw_json"]))
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def _video_path(self, video_id: str) -> Path | None:
        video = self.get_video(video_id)
        if video is None:
            return None
        path = Path(video.path)
        if not path.is_file() and not path.is_absolute():
            path = self.db_path.parent / path
        return path if path.is_file() else None

    def read_source_frame(self, video_id: str, frame_id: int) -> Any | None:
        frames = self.read_source_frames(video_id, [frame_id])
        return frames.get(frame_id)

    def read_source_frames(self, video_id: str, frame_ids: list[int]) -> dict[int, Any]:
        if not frame_ids:
            return {}
        if any(frame_id < 0 for frame_id in frame_ids):
            raise ValueError("frame_id must be >= 0")
        path = self._video_path(video_id)
        if path is None:
            return {}
        try:
            import cv2  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional local runtime
            raise RuntimeError("OpenCV is required for source-frame temporal localization") from exc

        wanted = sorted(set(int(frame_id) for frame_id in frame_ids))
        output: dict[int, Any] = {}
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                return {}
            for frame_id in wanted:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue
                output[frame_id] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            capture.release()
        return output

    def search_clip(self, vector: np.ndarray, top_k: int) -> list[dict[str, Any]]:
        if self.clip_index is None:
            raise RuntimeError("CLIP index is not configured")
        if top_k <= 0:
            return []
        return self.clip_index.search(vector, top_k)
