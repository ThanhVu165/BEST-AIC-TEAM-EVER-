"""Build the local SQLite + FAISS package from the official AIC data tree.

Expected input tree (relative to --data-root):
    videos/
    keyframes/keyframes/<video_id>/*.jpg
    objects/objects/<video_id>/*.json
    clip/clip-features-32/<video_id>.npy
    mapping/map-keyframes/<video_id>.csv
    media_info/media-info/<video_id>.json

The repository stores this official data under ``data/raw``. Generated
artifacts are kept outside git in ``database/`` and ``indexes/``.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data" / "raw"
SCHEMA = ROOT / "data_layer" / "sqlite_schema.sql"


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _media_info(payload: Any) -> dict[str, Any]:
    streams = payload.get("streams", []) if isinstance(payload, dict) else []
    video_stream = next((s for s in streams if isinstance(s, dict) and s.get("codec_type") == "video"), {})
    fmt = payload.get("format", {}) if isinstance(payload, dict) else {}
    if not isinstance(fmt, dict):
        fmt = {}
    fps_raw = video_stream.get("avg_frame_rate", video_stream.get("r_frame_rate", 0))
    if isinstance(fps_raw, str) and "/" in fps_raw:
        a, b = fps_raw.split("/", 1)
        fps = _number(a) / _number(b) if _number(b) else 0.0
    else:
        fps = _number(fps_raw)
    return {
        "fps": fps,
        "width": _int(video_stream.get("width", payload.get("width", 0) if isinstance(payload, dict) else 0)),
        "height": _int(video_stream.get("height", payload.get("height", 0) if isinstance(payload, dict) else 0)),
        "duration": _number(video_stream.get("duration", fmt.get("duration", 0))),
        "total_frames": _int(video_stream.get("nb_frames", video_stream.get("nb_read_frames", 0))),
    }


def _find_artifact(directory: Path, n: str, frame_idx: str, suffix: str) -> Path | None:
    files = list(directory.glob(f"*.{suffix}")) if directory.is_dir() else []
    values = set()
    for value in (n, frame_idx):
        if value.lstrip("-").isdigit():
            values.add(str(int(value)))
    for path in files:
        stem = path.stem
        if stem.isdigit() and str(int(stem)) in values:
            return path
        tokens = {str(int(x)) for x in re.findall(r"\d+", stem)}
        if tokens & values:
            return path
    return None


def _object_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("objects", "detections", "predictions", "instances", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _object_fields(item: dict[str, Any]) -> tuple[str, float, list[float]] | None:
    label = item.get("label", item.get("class", item.get("name", item.get("category"))))
    bbox = item.get("bbox", item.get("box"))
    if isinstance(bbox, dict):
        bbox = [bbox.get("x1"), bbox.get("y1"), bbox.get("x2"), bbox.get("y2")]
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or label is None:
        return None
    confidence = item.get("confidence", item.get("score", item.get("probability", 0.0)))
    return str(label), _number(confidence), [_number(x) for x in bbox]


def _load_mapping(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"n", "pts_time", "fps", "frame_idx"}
    if not rows or required - set(rows[0]):
        raise ValueError(f"{path}: mapping must contain {sorted(required)}")
    return rows


def build(data_root: Path, db_path: Path, index_path: Path, mapping_out: Path) -> None:
    if not data_root.is_dir():
        raise FileNotFoundError(f"AIC data root does not exist: {data_root}")

    clip_root = data_root / "clip" / "clip-features-32"
    mapping_root = data_root / "mapping" / "map-keyframes"
    keyframe_root = data_root / "keyframes" / "keyframes"
    object_root = data_root / "objects" / "objects"
    media_root = data_root / "media_info" / "media-info"
    video_root = data_root / "videos"

    clip_files = {p.stem: p for p in clip_root.glob("*.npy")}
    mapping_files = {p.stem: p for p in mapping_root.glob("*.csv")}
    media_files = {p.stem: p for p in media_root.glob("*.json")}
    video_files = {p.stem: p for p in video_root.rglob("*") if p.is_file()}
    ids = sorted(set(clip_files) & set(mapping_files))
    if not ids:
        raise RuntimeError(f"No CLIP/mapping pairs found below {data_root}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_out.parent.mkdir(parents=True, exist_ok=True)

    import faiss  # type: ignore

    rows_for_index: list[dict[str, Any]] = []
    dimension: int | None = None

    # This is a generated artifact. Recreate it so schema changes are applied
    # deterministically instead of leaving an older SQLite table definition.
    if db_path.exists():
        db_path.unlink()

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA.read_text(encoding="utf-8"))

        for video_id in ids:
            clip = np.load(clip_files[video_id], mmap_mode="r")
            if clip.ndim != 2 or clip.shape[1] != 512:
                raise ValueError(f"{video_id}: expected CLIP shape (N, 512), got {clip.shape}")
            rows = _load_mapping(mapping_files[video_id])
            if len(rows) != clip.shape[0]:
                raise ValueError(f"{video_id}: CLIP rows != mapping rows")
            if [int(r["n"]) for r in rows] != list(range(1, len(rows) + 1)):
                raise ValueError(f"{video_id}: mapping n is not sequential 1..N")

            media_payload: Any = {}
            if video_id in media_files:
                media_payload = json.loads(media_files[video_id].read_text(encoding="utf-8"))
            media = _media_info(media_payload)
            video_path = video_files.get(video_id)
            if video_path is None:
                raise FileNotFoundError(f"{video_id}: video file not found")
            stored_video_path = video_path.relative_to(ROOT).as_posix() if video_path.is_relative_to(ROOT) else video_path.as_posix()

            conn.execute(
                """INSERT INTO videos(video_id,path,fps,width,height,duration,total_frames,batch_id,metadata_available)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (video_id, stored_video_path, media["fps"], media["width"], media["height"], media["duration"], media["total_frames"], None, int(bool(media_payload))),
            )
            if media_payload:
                conn.execute("INSERT INTO metadata(video_id,raw_json) VALUES(?,?)", (video_id, json.dumps(media_payload, ensure_ascii=False)))

            frame_dir = keyframe_root / video_id
            object_dir = object_root / video_id
            for row in rows:
                keyframe_n = int(row["n"])
                frame_id = int(row["frame_idx"])
                frame_path = _find_artifact(frame_dir, row["n"], row["frame_idx"], "jpg")
                if frame_path is None:
                    raise FileNotFoundError(f"{video_id}: keyframe missing for keyframe_n={keyframe_n}, frame_idx={frame_id}")
                stored_frame_path = frame_path.relative_to(ROOT).as_posix() if frame_path.is_relative_to(ROOT) else frame_path.as_posix()
                conn.execute(
                    "INSERT INTO frames(video_id,keyframe_n,frame_id,timestamp,path,is_keyframe) VALUES(?,?,?,?,?,1)",
                    (video_id, keyframe_n, frame_id, float(row["pts_time"]), stored_frame_path),
                )

                object_path = _find_artifact(object_dir, row["n"], row["frame_idx"], "json")
                if object_path is not None:
                    payload = json.loads(object_path.read_text(encoding="utf-8"))
                    for item in _object_items(payload):
                        parsed = _object_fields(item)
                        if parsed is None:
                            continue
                        label, confidence, bbox = parsed
                        conn.execute(
                            """INSERT INTO objects(video_id,keyframe_n,frame_id,label,confidence,x1,y1,x2,y2)
                               VALUES(?,?,?,?,?,?,?,?,?)""",
                            (video_id, keyframe_n, frame_id, label, confidence, *bbox),
                        )
                rows_for_index.append({"video_id": video_id, "keyframe_n": keyframe_n, "frame_id": frame_id})

            if dimension is None:
                dimension = int(clip.shape[1])
        conn.commit()

    matrix = np.empty((len(rows_for_index), dimension or 512), dtype=np.float32)
    cursor = 0
    for video_id in ids:
        clip = np.load(clip_files[video_id], mmap_mode="r")
        count = clip.shape[0]
        matrix[cursor : cursor + count] = np.asarray(clip, dtype=np.float32)
        cursor += count
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    faiss.write_index(index, str(index_path))
    mapping_out.write_text(json.dumps(rows_for_index, ensure_ascii=False, indent=2), encoding="utf-8")

    with sqlite3.connect(db_path) as check:
        object_count = check.execute("SELECT COUNT(*) FROM objects").fetchone()[0]
    print(f"videos={len(ids)}")
    print(f"frames={len(rows_for_index)}")
    print(f"objects={object_count}")
    print(f"faiss_vectors={index.ntotal}")
    print(f"db={db_path}")
    print(f"index={index_path}")
    print(f"mapping={mapping_out}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build local AIC2026 SQLite + FAISS artifacts")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--db", type=Path, default=ROOT / "database" / "aic2026.sqlite")
    parser.add_argument("--index", type=Path, default=ROOT / "indexes" / "clip_vit_b32.faiss")
    parser.add_argument("--mapping", type=Path, default=ROOT / "indexes" / "clip_vit_b32.mapping.json")
    args = parser.parse_args()
    build(args.data_root.resolve(), args.db.resolve(), args.index.resolve(), args.mapping.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())