"""Run semantic hard-negative evaluation against real video frames/images.

Example:
    python tools/semantic_visual_benchmark.py manifest.json --backend siglip2

The manifest contains query/positive_frame/negative_frame paths. Model weights
are loaded only when this tool is executed, never during normal CI imports.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def _load_image(path: Path) -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for the visual benchmark") from exc
    return Image.open(path).convert("RGB")


def _build_scorer(backend: str):
    if backend == "siglip2":
        from query_engine.semantic_reranker import SigLIP2ImageTextScorer
        return SigLIP2ImageTextScorer()
    raise ValueError(f"unsupported backend: {backend}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--backend", default="siglip2", choices=["siglip2"])
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()

    rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    scorer = _build_scorer(args.backend)
    wins = 0
    margins: list[float] = []
    latencies: list[float] = []

    for row in rows:
        root = args.root or args.manifest.parent
        positive_path = (root / row["positive_frame"]).resolve()
        negative_path = (root / row["negative_frame"]).resolve()
        if not positive_path.is_file() or not negative_path.is_file():
            raise FileNotFoundError(f"missing benchmark frame: {positive_path} or {negative_path}")

        images = [_load_image(positive_path), _load_image(negative_path)]
        start = time.perf_counter()
        scores = scorer.score_images(images, row["query"])
        latencies.append((time.perf_counter() - start) * 1000.0)
        positive, negative = map(float, scores)
        margins.append(positive - negative)
        wins += int(positive > negative)

    count = len(rows)
    print(json.dumps({
        "backend": args.backend,
        "pairs": count,
        "pairwise_accuracy": wins / count if count else 0.0,
        "mean_margin": sum(margins) / count if count else 0.0,
        "mean_inference_ms_per_pair": sum(latencies) / count if count else 0.0,
    }, indent=2))


if __name__ == "__main__":
    main()
