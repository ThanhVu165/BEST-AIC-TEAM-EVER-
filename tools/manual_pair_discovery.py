"""Discover hard-negative frame pairs for human validation.

This tool intentionally does NOT assign ground truth. It uses SigLIP2 only to
find visually/semantically confusing candidate pairs from local keyframes. A
human must validate which frame, if any, actually satisfies the query.

Example (PowerShell):
    python tools/manual_pair_discovery.py \
      --images "data/raw/keyframes/**/*.jpg" \
      --query "a person riding a motorcycle" \
      --top 20 \
      --out benchmarks/manual_review_riding.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def _paths(patterns: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        base = Path(pattern)
        if base.is_file():
            candidates = [base]
        else:
            candidates = list(Path.cwd().glob(pattern))
        for path in candidates:
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    out.append(resolved)
    return out


def _score_images(paths: list[Path], query: str) -> list[float]:
    from PIL import Image
    from query_engine.semantic_reranker import SigLIP2ImageTextScorer

    scorer = SigLIP2ImageTextScorer()
    scores: list[float] = []
    batch = 16
    for start in range(0, len(paths), batch):
        current = paths[start : start + batch]
        with __import__("contextlib").ExitStack() as stack:
            images = [stack.enter_context(Image.open(path).convert("RGB")) for path in current]
            scores.extend(float(x) for x in scorer.score_images(images, query))
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    paths = _paths(args.images)
    if not paths:
        raise SystemExit("No local image files matched --images")

    scores = _score_images(paths, args.query)
    ranked = sorted(zip(paths, scores), key=lambda item: item[1], reverse=True)
    top = ranked[: max(1, args.top)]

    # Pair high-scoring frames with lower-scoring frames from the same sample.
    # The pair is deliberately unlabeled: automatic scoring is not ground truth.
    rows = []
    for idx, (positive_candidate, positive_score) in enumerate(top):
        negative_candidate, negative_score = ranked[min(len(ranked) - 1, len(top) + idx)]
        rows.append({
            "query": args.query,
            "frame_a": str(positive_candidate),
            "score_a": positive_score,
            "frame_b": str(negative_candidate),
            "score_b": negative_score,
            "human_label": None,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"images={len(paths)}")
    print(f"query={args.query}")
    print(f"pairs={len(rows)}")
    print(f"output={args.out}")
    print("human_label is intentionally null: validate the pair manually before using it as benchmark data.")


if __name__ == "__main__":
    main()
