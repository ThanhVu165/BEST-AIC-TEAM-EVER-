"""Evaluate Query Engine candidates against a local JSONL ground-truth file.

Expected JSONL rows contain a QueryRequest-compatible payload plus either
``relevant_video_ids`` or ``relevant_video_id``. Dataset and index files remain
local and are never committed to the repository.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from query_engine.evaluation import DEFAULT_KS, recall_at_ks
from query_engine.runtime import build_clip_baseline_engine
from schemas import QueryRequest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    engine = build_clip_baseline_engine(
        db_path=args.db,
        index_path=args.index,
        mapping_path=args.mapping,
        model_name=args.model,
        device=args.device,
    )

    totals = {int(k): 0.0 for k in DEFAULT_KS}
    count = 0
    for line_number, line in enumerate(
        args.queries.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        relevant = row.pop("relevant_video_ids", None)
        if relevant is None:
            single = row.pop("relevant_video_id", None)
            relevant = [single] if single is not None else []
        if not relevant:
            raise ValueError(f"line {line_number}: missing relevant_video_ids")

        request = QueryRequest.model_validate(row)
        response = engine.search(request)
        if response.status != "completed":
            raise RuntimeError(f"query {request.query_id} failed: {response.error}")
        curve = recall_at_ks(response.candidates, relevant)
        for k, value in curve.items():
            totals[int(k)] += value
        count += 1

    if count == 0:
        raise ValueError("no queries found")

    recalls = {f"R@{k}": round(totals[k] / count, 6) for k in sorted(totals)}
    final_score = sum(recalls.values()) / len(recalls)
    report = {
        "num_queries": count,
        **recalls,
        "FinalScore": round(final_score, 6),
        "note": "video-level retrieval score; this is not the official task evaluator",
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
