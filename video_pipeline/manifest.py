"""Batch manifest discovery and validation.

The BTC auxiliary files are kept outside Git. This module deliberately does not
assume one exact directory layout: it discovers files by artifact type and
matches per-video artifacts by their filename stem. Source-frame identifiers
must come from the supplied mapping files; the keyframe row number is never
used as a replacement for ``frame_id``.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _stem(path: Path) -> str:
    return path.stem


def _find_by_stem(files: Iterable[Path]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in files:
        result.setdefault(_norm(_stem(path)), path)
    return result


def _column(row: dict[str, str], aliases: tuple[str, ...]) -> str | None:
    normalized = {_norm(k): v for k, v in row.items()}
    for alias in aliases:
        value = normalized.get(_norm(alias))
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return None


@dataclass(frozen=True)
class ArtifactManifest:
    video_id: str
    video_path: str | None
    keyframe_mapping_path: str | None
    clip_feature_path: str | None
    metadata_path: str | None
    object_files: tuple[str, ...]
    keyframe_count: int = 0
    clip_rows: int = 0
    mapping_rows: int = 0
    object_count: int = 0
    mapping_valid: bool = False
    clip_mapping_aligned: bool = False
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ManifestBuilder:
    """Discover Batch-1 artifacts and validate per-video alignment."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()

    def build(self) -> list[ArtifactManifest]:
        files = [p for p in self.root.rglob("*") if p.is_file()]
        videos = [p for p in files if p.suffix.lower() in VIDEO_EXTENSIONS]
        if not videos:
            raise FileNotFoundError(f"No video files found below {self.root}")

        mappings = [p for p in files if p.suffix.lower() == ".csv"]
        clips = [p for p in files if p.suffix.lower() == ".npy"]
        json_files = [p for p in files if p.suffix.lower() == ".json"]
        images = [p for p in files if p.suffix.lower() in IMAGE_EXTENSIONS]

        mapping_by_stem = _find_by_stem(mappings)
        clip_by_stem = _find_by_stem(clips)
        json_by_stem = _find_by_stem(json_files)
        video_by_stem = _find_by_stem(videos)

        # Object JSONs are frequently stored as <video>/<frame>.json. Treat a
        # numeric JSON filename as an object file and group it by its parent.
        object_groups: dict[str, list[Path]] = {}
        for path in json_files:
            if path.stem.isdigit():
                object_groups.setdefault(_norm(path.parent.name), []).append(path)

        manifests: list[ArtifactManifest] = []
        for video_path in sorted(videos):
            video_id = video_path.stem
            key = _norm(video_id)
            mapping = mapping_by_stem.get(key)
            clip = clip_by_stem.get(key)
            metadata = json_by_stem.get(key)
            objects = tuple(str(p) for p in sorted(object_groups.get(key, [])))
            keyframe_count = self._count_keyframes(images, key)
            mapping_rows, mapping_error = self._validate_mapping(mapping)
            clip_rows, clip_error = self._count_clip_rows(clip)
            aligned = mapping_rows > 0 and clip_rows == mapping_rows
            errors = [e for e in (mapping_error, clip_error) if e]
            if mapping is None:
                errors.append("missing mapping CSV")
            if clip is None:
                errors.append("missing CLIP feature file")
            if mapping is not None and clip is not None and not aligned:
                errors.append(f"CLIP rows ({clip_rows}) != mapping rows ({mapping_rows})")

            manifests.append(
                ArtifactManifest(
                    video_id=video_id,
                    video_path=str(video_path),
                    keyframe_mapping_path=str(mapping) if mapping else None,
                    clip_feature_path=str(clip) if clip else None,
                    metadata_path=str(metadata) if metadata else None,
                    object_files=objects,
                    keyframe_count=keyframe_count,
                    clip_rows=clip_rows,
                    mapping_rows=mapping_rows,
                    object_count=len(objects),
                    mapping_valid=not mapping_error and mapping is not None,
                    clip_mapping_aligned=aligned,
                    errors=tuple(errors),
                )
            )
        return manifests

    @staticmethod
    def _count_clip_rows(path: Path | None) -> tuple[int, str | None]:
        if path is None:
            return 0, None
        try:
            import numpy as np

            arr = np.load(path, mmap_mode="r", allow_pickle=False)
            if arr.ndim != 2:
                return 0, f"CLIP array must be 2D, got shape {arr.shape}"
            if arr.shape[0] == 0 or arr.shape[1] == 0:
                return 0, f"CLIP array is empty: {arr.shape}"
            return int(arr.shape[0]), None
        except Exception as exc:
            return 0, f"failed to read CLIP file: {exc}"

    @staticmethod
    def _validate_mapping(path: Path | None) -> tuple[int, str | None]:
        if path is None:
            return 0, None
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if not rows:
                return 0, "mapping CSV is empty"
            required = ("frame_id", "frame", "source_frame", "source_frame_id", "frame_index")
            missing = [i for i, row in enumerate(rows) if _column(row, required) is None]
            if missing:
                return len(rows), (
                    "mapping CSV has no source frame id in rows "
                    + ",".join(map(str, missing[:5]))
                )
            return len(rows), None
        except Exception as exc:
            return 0, f"failed to read mapping CSV: {exc}"

    @staticmethod
    def _count_keyframes(images: list[Path], video_key: str) -> int:
        count = 0
        for path in images:
            parts = {_norm(part) for part in path.parts}
            if video_key in parts or _norm(path.parent.name) == video_key:
                count += 1
        return count


def write_manifest(manifests: list[ArtifactManifest], output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest_version": 1,
        "records": [item.to_dict() for item in manifests],
        "summary": {
            "videos": len(manifests),
            "mapping_valid": sum(item.mapping_valid for item in manifests),
            "clip_mapping_aligned": sum(item.clip_mapping_aligned for item in manifests),
            "keyframes": sum(item.keyframe_count for item in manifests),
            "objects": sum(item.object_count for item in manifests),
            "errors": sum(bool(item.errors) for item in manifests),
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
