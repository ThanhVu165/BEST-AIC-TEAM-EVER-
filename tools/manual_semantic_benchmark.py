"""Prepare and summarize manual semantic hard-negative evaluations.

There is no official public ground truth for the local AIC material available to
this repository, so this tool uses explicitly curated human-reviewed pairs.

Manifest JSON format:
[
  {
    "query": "person riding a motorcycle",
    "positive_frame": "keyframes/Lxx/Vyy/001.jpg",
    "negative_frame": "keyframes/Lxx/Vyy/020.jpg",
    "notes": "positive shows riding; negative shows standing beside bike"
  }
]

Use `--score` to run SigLIP2 on the manifest and report pairwise accuracy.
Without `--score`, the tool validates the manifest and prints the rows for
manual inspection/annotation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest must be a non-empty JSON list")
    for index, row in enumerate(rows, start=1):
        for key in ("query", "positive_frame", "negative_frame"):
            if key not in row or not str(row[key]).strip():
                raise ValueError(f"row {index} missing required field: {key}")
    return rows


def validate_paths(rows: list[dict[str, Any]], root: Path) -> None:
    missing: list[str] = []
    for index, row in enumerate(rows, start=1):
        for key in ("positive_frame", "negative_frame"):
            path = (root / str(row[key])).resolve()
            if not path.is_file():
                missing.append(f"row {index}: {path}")
    if missing:
        raise FileNotFoundError("missing benchmark frames:\n" + "\n".join(missing))


def score(rows: list[dict[str, Any]], root: Path, model_id: str) -> dict[str, float | int | str]:
    from PIL import Image
    from query_engine.semantic_reranker import SigLIP2ImageTextScorer

    scorer = SigLIP2ImageTextScorer(model_id=model_id)
    wins = 0
    margins: list[float] = []
    for row in rows:
        with Image.open((root / str(row["positive_frame"])).resolve()).convert("RGB") as positive_image:
            with Image.open((root / str(row["negative_frame"])).resolve()).convert("RGB") as negative_image:
                positive, negative = map(float, scorer.score_images([positive_image, negative_image], str(row["query"])))
        margins.append(positive - negative)
        wins += int(positive > negative)
    return {
        "model": model_id,
        "pairs": len(rows),
        "pairwise_accuracy": wins / len(rows),
        "mean_margin": sum(margins) / len(margins),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual AIC semantic benchmark")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--model", default="google/siglip2-base-patch16-256")
    args = parser.parse_args()

    rows = load_manifest(args.manifest)
    validate_paths(rows, args.root)
    if args.score:
        print(json.dumps(score(rows, args.root, args.model), indent=2))
        return

    print(json.dumps({
        "pairs": len(rows),
        "instructions": "Review each positive/negative frame pair manually and keep only unambiguous semantic contrasts.",
        "queries": sorted({str(row["query"]) for row in rows}),
    }, indent=2))
    for index, row in enumerate(rows, start=1):
        print(f"[{index}] {row['query']}")
        print(f"  positive: {row['positive_frame']}")
        print(f"  negative: {row['negative_frame']}")
        if row.get("notes"):
            print(f"  notes: {row['notes']}")


if __name__ == "__main__":
    main()
