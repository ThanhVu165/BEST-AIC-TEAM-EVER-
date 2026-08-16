from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


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

        actual_n = []
        for row in rows:
            try:
                actual_n.append(int(row["n"]))
                float(row["pts_time"])
                float(row["fps"])
                int(row["frame_idx"])
            except Exception as exc:
                errors.append(f"{video_id}: malformed mapping row: {exc}")
        if actual_n != list(range(1, len(rows) + 1)):
            errors.append(f"{video_id}: mapping n is not sequential 1..N")

        keyframes = {p.stem for p in frame_dir.glob("*.jpg")} if frame_dir.exists() else set()
        objects = {p.stem for p in object_dir.glob("*.json")} if object_dir.exists() else set()
        totals["keyframes"] += len(keyframes)
        totals["objects"] += len(objects)

        expected = {str(i) for i in range(1, len(rows) + 1)}
        if keyframes != expected:
            errors.append(f"{video_id}: keyframes missing={len(expected-keyframes)} extra={len(keyframes-expected)}")
        if objects != expected:
            errors.append(f"{video_id}: objects missing={len(expected-objects)} extra={len(objects-expected)}")
        if video_id not in media_files:
            errors.append(f"{video_id}: missing media_info")

    if set(clip_files) != set(mapping_files):
        errors.append(f"video ID mismatch CLIP↔mapping: {len(set(clip_files) ^ set(mapping_files))} IDs")
    if set(clip_files) != set(media_files):
        errors.append(f"video ID mismatch CLIP↔media: {len(set(clip_files) ^ set(media_files))} IDs")

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
