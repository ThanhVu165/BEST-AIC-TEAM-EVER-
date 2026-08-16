from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def video_id_from_file(path: Path) -> str:
    return path.stem


def audit(data_root: Path) -> int:
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

    ids = sorted(set(clip_files) | set(mapping_files) | set(media_files))
    errors: list[str] = []
    warnings: list[str] = []
    totals = {"clip_rows": 0, "mapping_rows": 0, "keyframes": 0, "objects": 0}

    for video_id in ids:
        clip_path = clip_files.get(video_id)
        mapping_path = mapping_files.get(video_id)
        frame_dir = keyframe_root / video_id
        object_dir = object_root / video_id

        if clip_path is None:
            errors.append(f"{video_id}: missing CLIP file")
            continue
        if mapping_path is None:
            errors.append(f"{video_id}: missing mapping CSV")
            continue

        try:
            clip = np.load(clip_path, mmap_mode="r")
        except Exception as exc:
            errors.append(f"{video_id}: cannot load CLIP: {exc}")
            continue

        if clip.ndim != 2 or clip.shape[1] != 512:
            errors.append(f"{video_id}: unexpected CLIP shape {clip.shape}")

        with mapping_path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))

        totals["clip_rows"] += int(clip.shape[0])
        totals["mapping_rows"] += len(rows)

        if len(rows) != clip.shape[0]:
            errors.append(f"{video_id}: CLIP rows={clip.shape[0]} mapping rows={len(rows)}")

        expected_n = list(range(1, len(rows) + 1))
        actual_n = []
        for row in rows:
            try:
                actual_n.append(int(row["n"]))
                float(row["pts_time"])
                float(row["fps"])
                int(row["frame_idx"])
            except Exception as exc:
                errors.append(f"{video_id}: malformed mapping row {row}: {exc}")
        if actual_n != expected_n:
            errors.append(f"{video_id}: mapping n is not sequential 1..N")

        keyframes = {p.stem for p in frame_dir.glob("*.jpg")} if frame_dir.exists() else set()
        objects = {p.stem for p in object_dir.glob("*.json")} if object_dir.exists() else set()
        totals["keyframes"] += len(keyframes)
        totals["objects"] += len(objects)

        expected = {str(i) for i in range(1, len(rows) + 1)}
        missing_k = expected - keyframes
        extra_k = keyframes - expected
        missing_o = expected - objects
        extra_o = objects - expected
        if missing_k or extra_k:
            errors.append(f"{video_id}: keyframes missing={len(missing_k)} extra={len(extra_k)}")
        if missing_o or extra_o:
            errors.append(f"{video_id}: objects missing={len(missing_o)} extra={len(extra_o)}")

        if video_id not in media_files:
            errors.append(f"{video_id}: missing media_info")

    clip_ids = set(clip_files)
    mapping_ids = set(mapping_files)
    media_ids = set(media_files)
    if clip_ids != mapping_ids:
        errors.append(f"video ID mismatch CLIP↔mapping: {len(clip_ids ^ mapping_ids)} IDs")
    if clip_ids != media_ids:
        errors.append(f"video ID mismatch CLIP↔media: {len(clip_ids ^ media_ids)} IDs")

    print("AIC2026 DATASET INTEGRITY AUDIT (READ-ONLY)")
    print(f"data_root       : {data_root}")
    print(f"videos          : {len(video_files)}")
    print(f"CLIP files      : {len(clip_files)}")
    print(f"mapping files   : {len(mapping_files)}")
    print(f"media_info      : {len(media_files)}")
    print(f"CLIP vectors    : {totals['clip_rows']}")
    print(f"mapping rows    : {totals['mapping_rows']}")
    print(f"keyframes       : {totals['keyframes']}")
    print(f"object files    : {totals['objects']}")
    print(f"errors          : {len(errors)}")
    print(f"warnings        : {len(warnings)}")

    if errors:
        print("\nERRORS:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("\nRESULT: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only AIC2026 dataset integrity audit")
    parser.add_argument("data_root", type=Path)
    args = parser.parse_args()
    return audit(args.data_root.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
