"""Benchmark source-video frame decoding strategies.

Compares the batch-window reader used by fine localization with repeated
single-frame reads. The latter is retained only as a diagnostic because
repeated H.264 random seeks can be slow and decoder-fragile.
"""
from __future__ import annotations

import argparse
import random
import sqlite3
import statistics
import time
from pathlib import Path

from data_layer.datastore import LocalDataStore


def _sample_anchors(db: Path, count: int, seed: int) -> list[tuple[str, int]]:
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            "SELECT video_id, total_frames FROM videos WHERE total_frames > 32"
        ).fetchall()
    finally:
        conn.close()
    rng = random.Random(seed)
    anchors: list[tuple[str, int]] = []
    for _ in range(count):
        video_id, total_frames = rng.choice(rows)
        anchors.append((str(video_id), rng.randint(16, int(total_frames) - 17)))
    return anchors


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, max(0, int(round(0.95 * len(ordered))) - 1))
    return {
        "n": len(values),
        "mean_ms": round(statistics.mean(values), 2),
        "p50_ms": round(statistics.median(values), 2),
        "p95_ms": round(ordered[p95_index], 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--radius", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.samples <= 0 or args.radius < 0:
        raise ValueError("samples must be > 0 and radius must be >= 0")

    store = LocalDataStore(db_path=args.db)
    anchors = _sample_anchors(args.db, args.samples, args.seed)
    batch_times: list[float] = []
    single_times: list[float] = []
    batch_missing = 0
    single_missing = 0
    requested_per_window = 2 * args.radius + 1

    for video_id, anchor in anchors:
        frame_ids = list(range(anchor - args.radius, anchor + args.radius + 1))

        started = time.perf_counter()
        batch = store.read_source_frames(video_id, frame_ids)
        batch_times.append((time.perf_counter() - started) * 1000)
        if len(batch) != len(frame_ids):
            batch_missing += 1

        started = time.perf_counter()
        singles = [store.read_source_frame(video_id, frame_id) for frame_id in frame_ids]
        single_times.append((time.perf_counter() - started) * 1000)
        if sum(image is not None for image in singles) != len(frame_ids):
            single_missing += 1

    batch_stats = _stats(batch_times)
    single_stats = _stats(single_times)
    speedup = (single_stats["mean_ms"] / batch_stats["mean_ms"]) if batch_stats["mean_ms"] else 0.0

    print("SOURCE FRAME DECODER BENCHMARK")
    print(f"samples={args.samples} radius={args.radius} requested_per_window={requested_per_window}")
    print(f"batch:  {batch_stats} missing_windows={batch_missing}")
    print(f"single: {single_stats} missing_windows={single_missing}")
    print(f"mean_speedup_single_over_batch={speedup:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
