"""Validate a local Batch 1 SQLite + FAISS package without uploading data.

The validator checks the invariants required by the Query Engine runtime:
- SQLite contains the expected video/frame/object tables and columns.
- FAISS mapping length equals the index size.
- Every mapping entry has a source video_id, keyframe_n and original frame_id.
- Mapping keyframes are unique.
- Every mapping entry resolves to the corresponding SQLite frame row.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {str(row[1]) for row in rows}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate local AIC 2026 Batch 1 artifacts")
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    args = parser.parse_args()

    if not args.db.is_file():
        raise FileNotFoundError(f"SQLite database not found: {args.db}")
    if not args.index.is_file():
        raise FileNotFoundError(f"FAISS index not found: {args.index}")
    if not args.mapping.is_file():
        raise FileNotFoundError(f"FAISS mapping not found: {args.mapping}")

    import faiss

    index = faiss.read_index(str(args.index))
    payload = json.loads(args.mapping.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("FAISS mapping must be a JSON list")
    if len(payload) != index.ntotal:
        raise ValueError(f"mapping rows={len(payload)} != FAISS vectors={index.ntotal}")

    pairs: list[tuple[str, int]] = []
    for internal_id, item in enumerate(payload):
        if not isinstance(item, dict):
            raise TypeError(f"mapping[{internal_id}] is not an object")
        required_mapping = {"video_id", "keyframe_n", "frame_id"}
        if required_mapping - set(item):
            raise ValueError(
                f"mapping[{internal_id}] lacks {sorted(required_mapping)}"
            )
        pairs.append((str(item["video_id"]), int(item["keyframe_n"])))
        int(item["frame_id"])
    if len(set(pairs)) != len(pairs):
        raise ValueError("mapping contains duplicate (video_id, keyframe_n) pairs")

    with sqlite3.connect(args.db) as conn:
        tables = {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        required = {"videos", "frames"}
        missing = required - tables
        if missing:
            raise ValueError(f"SQLite missing required tables: {sorted(missing)}")

        video_columns = _table_columns(conn, "videos")
        frame_columns = _table_columns(conn, "frames")
        required_video = {"video_id", "path"}
        required_frame = {"video_id", "keyframe_n", "frame_id", "timestamp", "path"}
        if required_video - video_columns:
            raise ValueError(
                f"videos missing columns: {sorted(required_video - video_columns)}"
            )
        if required_frame - frame_columns:
            raise ValueError(
                f"frames missing columns: {sorted(required_frame - frame_columns)}"
            )

        video_count = int(conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0])
        frame_count = int(conn.execute("SELECT COUNT(*) FROM frames").fetchone()[0])
        object_count = 0
        if "objects" in tables:
            object_count = int(conn.execute("SELECT COUNT(*) FROM objects").fetchone()[0])

        missing_pairs = 0
        query = "SELECT 1 FROM frames WHERE video_id = ? AND keyframe_n = ? LIMIT 1"
        for video_id, keyframe_n in pairs:
            if conn.execute(query, (video_id, keyframe_n)).fetchone() is None:
                missing_pairs += 1

    report = {
        "status": "ok" if missing_pairs == 0 else "failed",
        "faiss_vectors": int(index.ntotal),
        "faiss_dimension": int(index.d),
        "mapping_rows": len(payload),
        "videos": video_count,
        "frames": frame_count,
        "objects": object_count,
        "mapping_pairs_missing_from_frames": missing_pairs,
        "database": str(args.db),
        "index": str(args.index),
        "mapping": str(args.mapping),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if missing_pairs == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
