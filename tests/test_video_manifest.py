from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from video_pipeline.manifest import ManifestBuilder


def _write_mapping(path: Path, rows: list[tuple[int, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame_id", "path"])
        writer.writeheader()
        for frame_id, image in rows:
            writer.writerow({"frame_id": frame_id, "path": image})


def test_manifest_requires_source_frame_id(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    (root / "videos").mkdir(parents=True)
    (root / "keyframes" / "V1").mkdir(parents=True)
    (root / "videos" / "V1.mp4").write_bytes(b"not-a-real-video")
    (root / "keyframes" / "V1" / "0001.jpg").write_bytes(b"x")
    (root / "maps").mkdir()
    with (root / "maps" / "V1.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["keyframe_index", "path"])
        writer.writeheader()
        writer.writerow({"keyframe_index": 0, "path": "0001.jpg"})
    np.save(root / "V1.npy", np.ones((1, 4), dtype=np.float32))

    manifest = ManifestBuilder(root).build()[0]
    assert manifest.mapping_valid is False
    assert any("no source frame id" in error for error in manifest.errors)


def test_manifest_accepts_aligned_clip_and_mapping(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    (root / "videos").mkdir(parents=True)
    (root / "keyframes" / "V1").mkdir(parents=True)
    (root / "objects" / "V1").mkdir(parents=True)
    (root / "videos" / "V1.mp4").write_bytes(b"placeholder")
    for name in ("0001.jpg", "0002.jpg"):
        (root / "keyframes" / "V1" / name).write_bytes(b"x")
    _write_mapping(root / "V1.csv", [(10, "0001.jpg"), (20, "0002.jpg")])
    np.save(root / "V1.npy", np.ones((2, 4), dtype=np.float32))
    (root / "objects" / "V1" / "10.json").write_text(json.dumps({"frame_id": 10, "objects": []}))

    manifest = ManifestBuilder(root).build()[0]
    assert manifest.mapping_valid is True
    assert manifest.clip_mapping_aligned is True
    assert manifest.mapping_rows == 2
    assert manifest.clip_rows == 2
    assert manifest.object_count == 1
