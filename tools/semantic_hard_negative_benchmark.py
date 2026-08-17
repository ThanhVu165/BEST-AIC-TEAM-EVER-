"""Evaluate image-text semantic models on positive/hard-negative pairs.

Input JSONL format:
{"image": "path/to/frame.jpg", "positive": "person riding motorcycle", "negative": "person standing beside motorcycle"}

The tool reports pairwise accuracy and mean positive-minus-negative margin. It
is intentionally model-agnostic so SigLIP2 and future encoders can be compared
without changing the benchmark protocol.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_scorer(model_id: str):
    from query_engine.semantic_reranker import SigLIP2ImageTextScorer

    # The benchmark currently supports the SigLIP2 adapter; future adapters
    # should expose the same score_images contract.
    return SigLIP2ImageTextScorer(model_id=model_id)


def run(path: Path, model_id: str) -> dict[str, Any]:
    scorer = _load_scorer(model_id)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("benchmark file is empty")

    correct = 0
    margins: list[float] = []
    for row in rows:
        image_path = Path(row["image"])
        positive = str(row["positive"])
        negative = str(row["negative"])
        from PIL import Image

        with Image.open(image_path) as image:
            pos = float(scorer.score_images([image], positive)[0])
            neg = float(scorer.score_images([image], negative)[0])
        margin = pos - neg
        margins.append(margin)
        correct += int(pos > neg)

    return {
        "model": model_id,
        "examples": len(rows),
        "pairwise_accuracy": correct / len(rows),
        "mean_positive_minus_negative_margin": sum(margins) / len(margins),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--model", default="google/siglip2-base-patch16-256")
    args = parser.parse_args()
    print(json.dumps(run(args.jsonl, args.model), indent=2))


if __name__ == "__main__":
    main()
