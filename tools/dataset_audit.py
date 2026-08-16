from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import numpy as np


_INTEGER_RE = re.compile(r"\d+")


def _normalized_stem(stem: str) -> str:
    """Normalize a filename stem for comparison without assuming zero padding."""
    stem = stem.strip()
    if stem.isdigit():
        return str(int(stem))
    return stem.lower()


def _numeric_tokens(stem: str) -> set[str]:
    return {str(int(token)) for token in _INTEGER_RE.findall(stem)}


def _match_frame_files(rows: list[dict[str, str]], files: set[str]) -> tuple[bool, str]:
    """Check frame/object filenames against mapping rows without assuming one naming scheme.

    BTC-derived artifacts may use the sequential mapping number (``n``), the source
    frame number (``frame_idx``), or a filename containing either value. The old
    audit assumed ``n`` and therefore reported ``missing=N extra=N`` even when the
    artifact count was exactly correct.
    """
    if len(files) != len(rows):
        return False, "count-mismatch"

    normalized_files = {_normalized_stem(stem) for stem in files}

    # Direct numeric stems: 1.jpg, 001.jpg, 000001.jpg, ...
    n_values = {_normalized_stem(row["n"]) for row in rows if row.get("n", "").isdigit()}
    if n_values == normalized_files:
        return True, "n"

    # Source frame stems: 1532.jpg, 0001532.jpg, ...
    frame_values = {
        _normalized_stem(row["frame_idx"])
        for row in rows
        if row.get("frame_idx", "").lstrip("-").isdigit()
    }
    if frame_values == normalized_files:
        return True, "frame_idx"

    # Prefixed filenames such as frame_1532.jpg or L26_V001_1532.jpg.
    for field, values in (("n", n_values), ("frame_idx", frame_values)):
        matched = sum(bool(_numeric_tokens(stem) & values) for stem in files)
        if matched == len(files):
            return True, f"{field}-suffix"

    return False, "unmatched"


def audit(data_root: Path, verbose: bool = False, max_errors: int = 20) -> int:
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
    keyframe_mismatches: list[tuple[str, str]] = []
    object_mismatches: list[tuple[str, str]] = []
    keyframe_modes: Counter[str] = Counter()
    object_modes: Counter[str] = Counter()
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

        actual_n: list[int] = []
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

        keyframe_ok, keyframe_mode = _match_frame_files(rows, keyframes)
        object_ok, object_mode = _match_frame_files(rows, objects)
        keyframe_modes[keyframe_mode] += 1
        object_modes[object_mode] += 1

        if not keyframe_ok:
            keyframe_mismatches.append((video_id, keyframe_mode))
        if not object_ok:
            object_mismatches.append((video_id, object_mode))

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
    print(f"keyframe match  : {dict(keyframe_modes)}")
    print(f"object match    : {dict(object_modes)}")
    print(f"errors          : {len(errors)}")
    print(f"keyframe mismatches: {len(keyframe_mismatches)}")
    print(f"object mismatches  : {len(object_mismatches)}")

    if keyframe_mismatches:
        print("\nKEYFRAME MISMATCH EXAMPLES:")
        for video_id, mode in keyframe_mismatches[:max_errors]:
            print(f"- {video_id}: mode={mode}")
        if len(keyframe_mismatches) > max_errors:
            print(f"  ... {len(keyframe_mismatches) - max_errors} more (use --verbose)")

    if object_mismatches:
        print("\nOBJECT MISMATCH EXAMPLES:")
        for video_id, mode in object_mismatches[:max_errors]:
            print(f"- {video_id}: mode={mode}")
        if len(object_mismatches) > max_errors:
            print(f"  ... {len(object_mismatches) - max_errors} more (use --verbose)")

    if errors:
        print("\nERRORS:")
        shown = errors if verbose else errors[:max_errors]
        for error in shown:
            print(f"- {error}")
        if not verbose and len(errors) > max_errors:
            print(f"  ... {len(errors) - max_errors} more (use --verbose)")

    if errors or keyframe_mismatches or object_mismatches:
        print("\nRESULT: FAIL")
        return 1

    print("\nRESULT: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only AIC2026 dataset integrity audit")
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--verbose", action="store_true", help="print every diagnostic entry")
    parser.add_argument("--max-errors", type=int, default=20, help="number of examples shown by default")
    args = parser.parse_args()
    return audit(args.data_root.resolve(), verbose=args.verbose, max_errors=max(1, args.max_errors))


if __name__ == "__main__":
    raise SystemExit(main())
