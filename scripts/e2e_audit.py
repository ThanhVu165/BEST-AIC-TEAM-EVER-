"""Run a real KIS query and generate an HTML stage-by-stage audit report.

The report inspects:
  retrieval candidates -> source-frame decoding -> fine localization -> Top-100
without changing the query-engine implementation.

Usage:
  python scripts/e2e_audit.py "A man is riding a motorcycle on the street"
  python scripts/e2e_audit.py "..." --output reports/e2e_audit.html
"""
from __future__ import annotations

import argparse
import html
import json
import time
from pathlib import Path
from types import MethodType
from typing import Any

from query_engine.ranking import RankingEvidence, diversify_candidates, rerank_candidates
from query_engine.runtime import build_clip_baseline_engine
from query_engine.semantic_reranker import semantic_text
from query_engine.temporal import FrameEvidence, select_semantic_keyframes
from query_engine.query_understanding import understand_query
from schemas import QueryRequest


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports" / "e2e_audit.html"


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return '<div class="empty">No rows.</div>'
    head = "".join(f"<th>{_esc(label)}</th>" for _, label in columns)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{_esc(row.get(key, ''))}</td>" for key, _ in columns) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frame-top-k", type=int, default=5000)
    parser.add_argument("--temporal-anchors", type=int, default=20)
    parser.add_argument("--temporal-radius", type=int, default=16)
    args = parser.parse_args()

    db = ROOT / "database" / "aic2026.sqlite"
    index = ROOT / "indexes" / "clip_vit_b32.faiss"
    mapping = ROOT / "indexes" / "clip_vit_b32.mapping.json"

    timings: dict[str, float] = {}
    errors: list[str] = []
    decode_calls: list[dict[str, Any]] = []

    t0 = time.perf_counter()
    engine = build_clip_baseline_engine(
        db_path=db,
        index_path=index,
        mapping_path=mapping,
        frame_top_k=args.frame_top_k,
        fine_temporal_anchors=args.temporal_anchors,
        fine_temporal_radius=args.temporal_radius,
    )
    timings["engine_load_s"] = time.perf_counter() - t0

    original_reader = engine.retriever.datastore.read_source_frames

    def audited_reader(self, video_id: str, frame_ids: list[int]):
        started = time.perf_counter()
        error = None
        try:
            result = original_reader(video_id, frame_ids)
            return result
        except Exception as exc:  # audit must preserve the real failure
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            decode_calls.append({
                "video_id": video_id,
                "requested": len(frame_ids),
                "start_frame": min(frame_ids) if frame_ids else None,
                "end_frame": max(frame_ids) if frame_ids else None,
                "returned": len(locals().get("result", {})),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                "error": error,
            })

    engine.retriever.datastore.read_source_frames = MethodType(audited_reader, engine.retriever.datastore)

    request = QueryRequest(query_id="e2e-audit", task="KIS", description=args.query)
    spec = understand_query(request)

    t = time.perf_counter()
    try:
        hits = engine.retriever.retrieve(spec)
    except Exception as exc:
        errors.append(f"retrieval: {type(exc).__name__}: {exc}")
        hits = []
    timings["retrieval_s"] = time.perf_counter() - t

    sparse = select_semantic_keyframes(
        [engine._frame_evidence(hit) for hit in hits],
        max_candidates=max(engine.final_limit * 5, 100),
    )

    temporal: list[Any] = []
    t = time.perf_counter()
    try:
        temporal = engine._fine_localize(sparse[: engine.fine_temporal_anchors], spec.text)
    except Exception as exc:
        errors.append(f"fine_localization: {type(exc).__name__}: {exc}")
    timings["fine_localization_s"] = time.perf_counter() - t

    selected = engine._merge_kis_candidates(
        temporal,
        sparse,
        limit=max(engine.final_limit * 2, engine.fine_temporal_anchors),
    )

    # Reproduce the canonical KIS ranking path so the displayed Top-100 is
    # the same ranking family as BaselineQueryEngine._solve_kis().
    hit_by_exact_frame = {(hit.video_id, hit.frame_id): hit for hit in hits}
    hit_by_anchor = {(hit.video_id, hit.frame_id): hit for hit in hits}
    semantic_scores: dict[tuple[str, int], float] = {}
    t = time.perf_counter()
    try:
        semantic_scores = engine._semantic_scores(selected, semantic_text(spec))
    except Exception as exc:
        errors.append(f"semantic_rerank: {type(exc).__name__}: {exc}")
    timings["semantic_rerank_s"] = time.perf_counter() - t

    ranking_inputs: list[RankingEvidence] = []
    for item in selected:
        exact_hit = hit_by_exact_frame.get((item.video_id, item.frame_id))
        anchor_frame_id = item.anchor_frame_id if item.anchor_frame_id is not None else item.frame_id
        anchor_hit = hit_by_anchor.get((item.video_id, anchor_frame_id))
        hit = exact_hit or anchor_hit
        if hit is None:
            continue
        retrieval_score = hit.retrieval_score if hit.retrieval_score is not None else hit.score
        ranking_inputs.append(RankingEvidence(
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
            sources=tuple(dict.fromkeys((*hit.sources, "temporal_anchor" if exact_hit is None else "exact_frame"))),
        ))

    ranked = rerank_candidates(ranking_inputs, limit=max(engine.final_limit * 5, 100))
    ranked = diversify_candidates(ranked, limit=engine.final_limit, max_per_video=engine.max_kis_candidates_per_video)
    timings["ranking_s"] = 0.0

    temporal_lookup = {(item.video_id, item.frame_id): item for item in selected}
    top100: list[dict[str, Any]] = []
    for rank in ranked:
        item = temporal_lookup.get((rank.video_id, rank.frame_id))
        exact = hit_by_exact_frame.get((rank.video_id, rank.frame_id))
        anchor_id = item.anchor_frame_id if item and item.anchor_frame_id is not None else rank.frame_id
        anchor = hit_by_anchor.get((rank.video_id, anchor_id))
        evidence = exact or anchor
        top100.append({
            "rank": len(top100) + 1,
            "video_id": rank.video_id,
            "frame_id": rank.frame_id,
            "fused": round(float(rank.fused_score), 6),
            "retrieval": round(float(rank.retrieval_score), 6),
            "temporal": round(float(item.score), 6) if item else 0.0,
            "semantic": round(float(rank.semantic_score), 6),
            "anchor_frame": anchor_id,
            "localized": bool(item and item.frame_id != anchor_id),
            "sources": ", ".join(rank.sources),
            "clip": round(float(evidence.retrieval_score), 6) if evidence and evidence.retrieval_score is not None else "",
        })

    status = "PASS" if top100 and not errors else ("PARTIAL" if top100 else "FAIL")
    decode_total_ms = sum(float(item["elapsed_ms"]) for item in decode_calls)
    decode_failures = [item for item in decode_calls if item["error"]]
    report = {
        "status": status,
        "query": args.query,
        "config": {
            "frame_top_k": args.frame_top_k,
            "temporal_anchors": engine.fine_temporal_anchors,
            "temporal_radius": engine.fine_temporal_radius,
            "final_limit": engine.final_limit,
            "max_kis_candidates_per_video": engine.max_kis_candidates_per_video,
        },
        "counts": {
            "retrieval_hits": len(hits),
            "sparse_candidates": len(sparse),
            "temporal_candidates": len(temporal),
            "selected_before_ranking": len(selected),
            "ranking_inputs": len(ranking_inputs),
            "top100": len(top100),
            "decode_calls": len(decode_calls),
            "decode_failures": len(decode_failures),
        },
        "timings": timings,
        "decode_total_ms": round(decode_total_ms, 2),
        "errors": errors,
    }

    retrieval_rows = [
        {
            "rank": i + 1,
            "video_id": h.video_id,
            "frame_id": h.frame_id,
            "score": round(float(h.score), 6),
            "retrieval": round(float(h.retrieval_score), 6) if h.retrieval_score is not None else "",
            "sources": ", ".join(h.sources),
        }
        for i, h in enumerate(hits[:100])
    ]
    decode_rows = decode_calls[:200]
    top_rows = top100

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AIC2026 E2E Retrieval Audit</title>
<style>
body{{font-family:Inter,Segoe UI,Arial,sans-serif;margin:0;background:#0f1115;color:#e8eaed}}
main{{max-width:1500px;margin:auto;padding:28px}} h1{{margin:0 0 8px}} h2{{margin-top:32px}}
.sub{{color:#9aa0a6}} .grid{{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin:20px 0}}
.card{{background:#191c22;border:1px solid #2a2f38;border-radius:10px;padding:16px}} .value{{font-size:25px;font-weight:700;margin-top:6px}}
.pass{{color:#61d095}} .fail{{color:#ff6b6b}} .partial{{color:#f3c969}}
table{{width:100%;border-collapse:collapse;background:#15181d;font-size:13px}} th,td{{padding:8px 10px;border-bottom:1px solid #282d35;text-align:left}} th{{position:sticky;top:0;background:#20242b}}
.wrap{{overflow:auto;max-height:650px;border:1px solid #292e37;border-radius:8px}} pre{{background:#15181d;padding:16px;overflow:auto;border-radius:8px}}
.badge{{padding:4px 8px;border-radius:999px;background:#292e37}} code{{color:#b9c7ff}}
</style></head><body><main>
<h1>AIC2026 End-to-End Retrieval Audit</h1><div class="sub">Query: <code>{_esc(args.query)}</code></div>
<div class="grid">
<div class="card">Status<div class="value {_esc(status.lower())}">{_esc(status)}</div></div>
<div class="card">Retrieval hits<div class="value">{len(hits)}</div></div>
<div class="card">Temporal candidates<div class="value">{len(temporal)}</div></div>
<div class="card">Ranking inputs<div class="value">{len(ranking_inputs)}</div></div>
<div class="card">Top-100<div class="value">{len(top100)}</div></div>
<div class="card">Decode failures<div class="value">{len(decode_failures)}</div></div>
</div>
<h2>Pipeline checkpoints</h2>
<table><thead><tr><th>Stage</th><th>Status</th><th>Count / detail</th><th>Time</th></tr></thead><tbody>
<tr><td>Retrieval candidates</td><td>{'PASS' if hits else 'FAIL'}</td><td>{len(hits)} hits</td><td>{timings['retrieval_s']:.3f}s</td></tr>
<tr><td>Source-frame decoding</td><td>{'FAIL' if decode_failures else ('PASS' if decode_calls else 'N/A')}</td><td>{len(decode_calls)} batch reads / {decode_total_ms:.1f}ms</td><td>{decode_total_ms/1000:.3f}s</td></tr>
<tr><td>Fine localization</td><td>{'PASS' if temporal else 'FAIL'}</td><td>{len(temporal)} candidates</td><td>{timings['fine_localization_s']:.3f}s</td></tr>
<tr><td>Top-100 ranking</td><td>{'PASS' if len(top100) == engine.final_limit else 'PARTIAL'}</td><td>{len(top100)} results</td><td>{timings['ranking_s']:.3f}s</td></tr>
</tbody></table>
<h2>Retrieval candidates</h2><div class="wrap">{_table(retrieval_rows,[('rank','Rank'),('video_id','Video'),('frame_id','Frame'),('score','Score'),('retrieval','Retrieval'),('sources','Sources')])}</div>
<h2>Source-frame decoding</h2><div class="wrap">{_table(decode_rows,[('video_id','Video'),('requested','Requested'),('start_frame','Start'),('end_frame','End'),('returned','Returned'),('elapsed_ms','ms'),('error','Error')])}</div>
<h2>Final Top-100</h2><div class="wrap">{_table(top_rows,[('rank','Rank'),('video_id','Video'),('frame_id','Frame'),('fused','Fused'),('retrieval','Retrieval'),('temporal','Temporal'),('semantic','Semantic'),('anchor_frame','Anchor'),('localized','Localized'),('sources','Sources')])}</div>
<h2>Raw audit JSON</h2><pre>{_esc(payload)}</pre>
</main></body></html>"""

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"HTML report: {args.output}")
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
