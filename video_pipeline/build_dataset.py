"""Build the offline Batch-1 SQLite + CLIP FAISS package.

Usage::

    python -m video_pipeline.build_dataset --data-root ./data/raw \
        --database ./database/aic2026.sqlite \
        --index-root ./indexes

The command is intentionally fail-fast on CLIP/mapping misalignment. It may
write a manifest even when validation fails, but it never silently substitutes
keyframe ordinals for source ``frame_id`` values.
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from .manifest import ManifestBuilder, write_manifest

FRAME_ALIASES = ("frame_id", "frame", "source_frame", "source_frame_id", "frame_index")
PATH_ALIASES = ("path", "file", "filename", "file_name", "image", "image_path", "keyframe")
FPS_ALIASES = ("fps", "frame_rate", "framerate")
WIDTH_ALIASES = ("width", "video_width")
HEIGHT_ALIASES = ("height", "video_height")
DURATION_ALIASES = ("duration", "duration_sec", "duration_seconds")
TOTAL_FRAMES_ALIASES = ("total_frames", "frame_count", "frames")


def _norm(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())


def _value(row: dict[str, Any], aliases: tuple[str, ...]) -> str | None:
    values = {_norm(str(k)): v for k, v in row.items()}
    for alias in aliases:
        value = values.get(_norm(alias))
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return None


def _int(value: str | None, field: str) -> int:
    if value is None:
        raise ValueError(f"missing {field}")
    return int(float(value))


def _float(value: str | None, field: str) -> float:
    if value is None:
        raise ValueError(f"missing {field}")
    return float(value)


def _load_mapping(path: Path, video_root: Path, video_id: str) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty mapping CSV: {path}")

    image_candidates = sorted(
        p for p in video_root.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    records: list[dict[str, Any]] = []
    for ordinal, row in enumerate(rows):
        frame_text = _value(row, FRAME_ALIASES)
        if frame_text is None:
            raise ValueError(f"{path}: row {ordinal} has no source frame id")
        frame_id = _int(frame_text, "frame_id")
        raw_path = _value(row, PATH_ALIASES)
        if raw_path:
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                candidate = video_root / candidate
            if not candidate.exists():
                # Some mappings contain only a basename while the keyframes
                # are nested under the video directory.
                matches = [p for p in image_candidates if p.name == candidate.name]
                candidate = matches[0] if matches else candidate
            frame_path = candidate
        elif ordinal < len(image_candidates):
            frame_path = image_candidates[ordinal]
        else:
            raise ValueError(
                f"{path}: row {ordinal} has no keyframe path and no corresponding image"
            )
        records.append(
            {
                "video_id": video_id,
                "frame_id": frame_id,
                "path": str(frame_path),
                "is_keyframe": 1,
            }
        )
    return records


def _metadata_value(data: dict[str, Any], aliases: tuple[str, ...]) -> str | None:
    return _value(data, aliases)


def _video_info(video_path: Path, metadata_path: Path | None) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if metadata_path and metadata_path.exists():
        try:
            loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except Exception:
            data = {}

    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = total_frames / fps if fps > 0 else 0.0
        cap.release()
    except Exception as exc:
        raise RuntimeError(f"cannot read video metadata from {video_path}: {exc}") from exc

    fps = _float(_metadata_value(data, FPS_ALIASES), "fps") if _metadata_value(data, FPS_ALIASES) else fps
    width = _int(_metadata_value(data, WIDTH_ALIASES), "width") if _metadata_value(data, WIDTH_ALIASES) else width
    height = _int(_metadata_value(data, HEIGHT_ALIASES), "height") if _metadata_value(data, HEIGHT_ALIASES) else height
    duration = _float(_metadata_value(data, DURATION_ALIASES), "duration") if _metadata_value(data, DURATION_ALIASES) else duration
    total_frames = _int(_metadata_value(data, TOTAL_FRAMES_ALIASES), "total_frames") if _metadata_value(data, TOTAL_FRAMES_ALIASES) else total_frames

    if fps <= 0 or width <= 0 or height <= 0 or total_frames <= 0:
        raise ValueError(f"invalid video metadata for {video_path}: {fps=}, {width=}, {height=}, {total_frames=}")
    return {
        "fps": fps,
        "width": width,
        "height": height,
        "duration": duration,
        "total_frames": total_frames,
        "raw_metadata": data,
    }


def _object_records(paths: tuple[str, ...], video_id: str) -> list[tuple[Any, ...]]:
    result: list[tuple[Any, ...]] = []
    for raw_path in paths:
        path = Path(raw_path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        frame_id: int | None = None
        if isinstance(payload, dict):
            for key in ("frame_id", "frame", "source_frame_id", "frame_index"):
                if key in payload:
                    try:
                        frame_id = int(payload[key])
                    except (TypeError, ValueError):
                        pass
                    break
            objects = payload.get("objects", payload.get("detections", payload.get("annotations", [])))
        else:
            objects = payload
        if frame_id is None and path.stem.isdigit():
            frame_id = int(path.stem)
        if frame_id is None or not isinstance(objects, list):
            continue
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            label = obj.get("label", obj.get("class", obj.get("name")))
            confidence = obj.get("confidence", obj.get("score", obj.get("probability")))
            bbox = obj.get("bbox", obj.get("box", obj.get("bounding_box")))
            if label is None or confidence is None or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            try:
                x1, y1, x2, y2 = (float(v) for v in bbox)
                result.append((video_id, frame_id, str(label), float(confidence), x1, y1, x2, y2))
            except (TypeError, ValueError):
                continue
    return result


def build(data_root: Path, database: Path, index_root: Path, batch_id: str = "batch1") -> dict[str, Any]:
    manifests = ManifestBuilder(data_root).build()
    manifest_path = index_root / "manifest.json"
    write_manifest(manifests, manifest_path)

    invalid = [item for item in manifests if item.errors]
    if invalid:
        sample = "\n".join(f"{m.video_id}: {', '.join(m.errors)}" for m in invalid[:10])
        raise RuntimeError(f"Batch validation failed for {len(invalid)} videos.\n{sample}")

    schema_path = Path(__file__).resolve().parents[1] / "data_layer" / "sqlite_schema.sql"
    database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as conn:
        conn.executescript(schema_path.read_text(encoding="utf-8"))
        conn.execute("DELETE FROM objects")
        conn.execute("DELETE FROM metadata")
        conn.execute("DELETE FROM frames")
        conn.execute("DELETE FROM videos")

        for manifest in manifests:
            video_path = Path(manifest.video_path or "")
            info = _video_info(video_path, Path(manifest.metadata_path) if manifest.metadata_path else None)
            conn.execute(
                "INSERT INTO videos VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    manifest.video_id,
                    str(video_path),
                    info["fps"],
                    info["width"],
                    info["height"],
                    info["duration"],
                    info["total_frames"],
                    batch_id,
                    1 if manifest.metadata_path else 0,
                ),
            )
            if manifest.metadata_path:
                raw = Path(manifest.metadata_path).read_text(encoding="utf-8")
                conn.execute("INSERT INTO metadata VALUES (?, ?)", (manifest.video_id, raw))

            video_root = video_path.parent
            mappings = _load_mapping(Path(manifest.keyframe_mapping_path), video_root, manifest.video_id)
            conn.executemany(
                "INSERT INTO frames(video_id, frame_id, timestamp, path, is_keyframe) VALUES (?, ?, ?, ?, ?)",
                [
                    (m["video_id"], m["frame_id"], m["frame_id"] / info["fps"], m["path"], m["is_keyframe"])
                    for m in mappings
                ],
            )
            objects = _object_records(manifest.object_files, manifest.video_id)
            conn.executemany(
                "INSERT INTO objects(video_id, frame_id, label, confidence, x1, y1, x2, y2) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                objects,
            )
        conn.commit()

    build_faiss(manifests, index_root / "clip")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def build_faiss(manifests: list[Any], output_dir: Path) -> None:
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("FAISS is required to build the CLIP index; install the [ml] extra") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    index = None
    mapping: list[dict[str, Any]] = []
    for manifest in sorted(manifests, key=lambda item: item.video_id):
        if not manifest.clip_feature_path or not manifest.keyframe_mapping_path:
            raise RuntimeError(f"missing CLIP/mapping for {manifest.video_id}")
        vectors = np.load(manifest.clip_feature_path, mmap_mode="r", allow_pickle=False)
        rows = _load_mapping(Path(manifest.keyframe_mapping_path), Path(manifest.video_path).parent, manifest.video_id)
        if vectors.ndim != 2 or vectors.shape[0] != len(rows):
            raise RuntimeError(f"CLIP/mapping mismatch for {manifest.video_id}: {vectors.shape} vs {len(rows)}")
        if index is None:
            index = faiss.IndexFlatIP(int(vectors.shape[1]))
        elif index.d != int(vectors.shape[1]):
            raise RuntimeError(f"inconsistent CLIP dimensions: {index.d} vs {vectors.shape[1]}")
        for start in range(0, len(rows), 4096):
            batch = np.asarray(vectors[start:start + 4096], dtype=np.float32)
            norms = np.linalg.norm(batch, axis=1, keepdims=True)
            batch = batch / np.maximum(norms, 1e-12)
            index.add(batch)
            mapping.extend(rows[start:start + len(batch)])

    if index is None:
        raise RuntimeError("no CLIP vectors found")
    if index.ntotal != len(mapping):
        raise RuntimeError(f"index/mapping size mismatch: {index.ntotal} != {len(mapping)}")
    faiss.write_index(index, str(output_dir / "frame.faiss"))
    (output_dir / "frame_mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    (output_dir / "frame_index_meta.json").write_text(
        json.dumps({"index_type": "IndexFlatIP", "metric": "inner_product_on_l2_normalized_vectors", "ntotal": index.ntotal, "dimension": index.d}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--database", default="./database/aic2026.sqlite")
    parser.add_argument("--index-root", default="./indexes")
    parser.add_argument("--batch-id", default="batch1")
    args = parser.parse_args()
    summary = build(Path(args.data_root), Path(args.database), Path(args.index_root), args.batch_id)
    print(json.dumps(summary["summary"], indent=2))


if __name__ == "__main__":
    main()
