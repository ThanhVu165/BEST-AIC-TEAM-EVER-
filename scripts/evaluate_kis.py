"""Evaluate KIS stage-by-stage on a local JSONL benchmark.

This is an internal/stress-test evaluator. It must not be interpreted as the
official AIC 2026 evaluator until BTC supplies official queries and GT.

JSONL row format::

    {
      "query_id": "q001",
      "task": "KIS",
      "description": "...",
      "ground_truth": [
        {"video_id": "L21_V001", "start_frame": 1200, "end_frame": 1210}
      ]
    }

The report measures four checkpoints:

1. frame-level retrieval
2. video candidate generation
3. fine temporal localization
4. final Top-100

It also records source-frame decoder calls, partial reads/failures and latency.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from query_engine.ranking import RankingEvidence, diversify_candidates, rerank_candidates
from query_engine.runtime import build_clip_baseline_engine
from query_engine.semantic_reranker import semantic_text
from query_engine.temporal import select_semantic_keyframes
from query_engine.query_understanding import understand_query
from schemas import QueryRequest

KS = (1, 5, 20, 50, 100)


def _gt_frame_hit(video_id: str, frame_id: int | None, gt: list[dict[str, Any]]) -> bool:
    if frame_id is None:
        return False
    return any(
        video_id == str(item["video_id"])
        and int(item["start_frame"]) <= int(frame_id) <= int(item["end_frame"])
        for item in gt
    )


def _gt_video_hit(video_id: str, gt: list[dict[str, Any]]) -> bool:
    return any(video_id == str(item["video_id"]) for item in gt)


def _curve(rows: list[dict[str, Any]], gt: list[dict[str, Any]], *, frame: bool) -> dict[str, float]:
    result: dict[str, float] = {}
    for k in KS:
        subset = rows[:k]
        hit = any(
            (_gt_frame_hit(str(row["video_id"]), row.get("frame_id"), gt) if frame else _gt_video_hit(str(row["video_id"]), gt))
            for row in subset
        )
        result[f"R@{k}"] = float(hit)
    return result


def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {f"R@{k}": 0.0 for k in KS}
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def _final_score(curve: dict[str, float]) -> float:
    return sum(curve.values()) / len(curve) if curve else 0.0


def _rank_canonical(engine, spec, hits):
    sparse = select_semantic_keyframes(
        [engine._frame_evidence(hit) for hit in hits],
        max_candidates=max(engine.final_limit * 5, 100),
    )
    temporal = (
        engine._fine_localize(sparse[: engine.fine_temporal_anchors], spec.text)
        if engine.image_encoder is not None
        else []
    )
    selected = engine._merge_kis_candidates(
        temporal,
        sparse,
        limit=max(engine.final_limit * 2, engine.fine_temporal_anchors),
    )

    hit_by_exact_frame = {(hit.video_id, hit.frame_id): hit for hit in hits}
    hit_by_anchor = {(hit.video_id, hit.frame_id): hit for hit in hits}
    semantic_scores = engine._semantic_scores(selected, semantic_text(spec))

    ranking_inputs: list[RankingEvidence] = []
    for item in selected:
        exact_hit = hit_by_exact_frame.get((item.video_id, item.frame_id))
        anchor_frame_id = item.anchor_frame_id if item.anchor_frame_id is not None else item.frame_id
        anchor_hit = hit_by_anchor.get((item.video_id, anchor_frame_id))
        hit = exact_hit or anchor_hit
        if hit is None:
            continue
        retrieval_score = hit.retrieval_score if hit.retrieval_score is not None else hit.score
        ranking_inputs.append(
            RankingEvidence(
                video_id=item.video_id,
                frame_id=item.frame_id,
                retrieval_score=retrieval_score,
                object_score=hit.object_score if exact_hit is not None else 0.0,
                metadata_score=hit.metadata_score if exact_hit is not None else 0.0,
                ocr_score=hit.ocr_score if exact_hit is not None else 0.0,
                asr_score=hit.asr_score,
                temporal_score=item.score,
                semantic_score=semantic_scores.get((item.video_id, item.frame_id), 0.0),
                semantic_weight=engine.semantic_config.weight,
                sources=tuple(
                    dict.fromkeys(
                        (*hit.sources, "temporal_anchor" if exact_hit is None else "exact_frame")
                    )
                ),
            )
        )

    ranked = rerank_candidates(ranking_inputs, limit=max(engine.final_limit * 5, 100))
    ranked = diversify_candidates(
        ranked,
        limit=engine.final_limit,
        max_per_video=engine.max_kis_candidates_per_video,
    )

    retrieval_rows = [
        {"rank": i + 1, "video_id": hit.video_id, "frame_id": hit.frame_id}
        for i, hit in enumerate(hits[:100])
    ]

    seen_videos: OrderedDict[str, None] = OrderedDict()
    for hit in hits:
        seen_videos.setdefault(hit.video_id, None)
    video_rows = [
        {"rank": i + 1, "video_id": video_id}
        for i, video_id in enumerate(seen_videos, 1)
    ]

    temporal_rows = [
        {"rank": i + 1, "video_id": item.video_id, "frame_id": item.frame_id}
        for i, item in enumerate(temporal)
    ]
    final_rows = [
        {"rank": i + 1, "video_id": item.video_id, "frame_id": item.frame_id}
        for i, item in enumerate(ranked)
    ]
    return retrieval_rows, video_rows, temporal_rows, final_rows, len(sparse), len(temporal)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--frame-top-k", type=int, default=5000)
    parser.add_argument("--temporal-anchors", type=int, default=64)
    parser.add_argument("--temporal-radius", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    engine = build_clip_baseline_engine(
        db_path=args.db,
        index_path=args.index,
        mapping_path=args.mapping,
        model_name=args.model,
        device=args.device,
        frame_top_k=args.frame_top_k,
        fine_temporal_anchors=args.temporal_anchors,
        fine_temporal_radius=args.temporal_radius,
    )

    stage_curves: dict[str, list[dict[str, float]]] = {
        "retrieval_frame": [],
        "video_candidates": [],
        "fine_localization": [],
        "final_top100": [],
    }
    query_reports: list[dict[str, Any]] = []
    decode_calls = 0
    decode_partial_or_failed = 0
    decode_ms = 0.0
    query_count = 0

    original_reader = engine.retriever.datastore.read_source_frames

    def audited_reader(video_id: str, frame_ids: list[int]):
        nonlocal decode_calls, decode_partial_or_failed, decode_ms
        started = time.perf_counter()
        try:
            result = original_reader(video_id, frame_ids)
            if len(result) < len(set(frame_ids)):
                decode_partial_or_failed += 1
            return result
        except Exception:
            decode_partial_or_failed += 1
            raise
        finally:
            decode_calls += 1
            decode_ms += (time.perf_counter() - started) * 1000

    engine.retriever.datastore.read_source_frames = audited_reader

    for line_number, line in enumerate(
        args.queries.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        gt = row.pop("ground_truth", None)
        if not isinstance(gt, list) or not gt:
            raise ValueError(f"line {line_number}: ground_truth must be a non-empty list")

        request = QueryRequest.model_validate(row)
        if request.task not in (None, "KIS"):
            raise ValueError(f"line {line_number}: only KIS is supported")

        spec = understand_query(request)
        query_count += 1
        started = time.perf_counter()
        hits = engine.retriever.retrieve(spec)
        retrieval_s = time.perf_counter() - started

        retrieval_rows, video_rows, temporal_rows, final_rows, sparse_n, temporal_n = _rank_canonical(
            engine, spec, hits
        )
        curves = {
            "retrieval_frame": _curve(retrieval_rows, gt, frame=True),
            "video_candidates": _curve(video_rows, gt, frame=False),
            "fine_localization": _curve(temporal_rows, gt, frame=True),
            "final_top100": _curve(final_rows, gt, frame=True),
        }
        for stage, curve in curves.items():
            stage_curves[stage].append(curve)

        query_reports.append(
            {
                "query_id": request.query_id,
                "retrieval_hits": len(hits),
                "sparse_candidates": sparse_n,
                "temporal_candidates": temporal_n,
                "final_candidates": len(final_rows),
                "retrieval_s": round(retrieval_s, 4),
                "stage_curves": curves,
            }
        )

    aggregate: dict[str, dict[str, float]] = {}
    for stage, curves in stage_curves.items():
        curve = _mean(curves)
        aggregate[stage] = {**curve, "FinalScore": _final_score(curve)}

    report = {
        "status": "ok",
        "num_queries": query_count,
        "official": False,
        "warning": (
            "Internal/stress-test evaluator. Do not report as official AIC 2026 score "
            "unless BTC supplies the official query/GT format."
        ),
        "stage_metrics": aggregate,
        "decoder": {
            "calls": decode_calls,
            "partial_or_failed_calls": decode_partial_or_failed,
            "total_ms": round(decode_ms, 2),
            "mean_ms_per_call": round(decode_ms / decode_calls, 2) if decode_calls else 0.0,
        },
        "queries": query_reports,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
