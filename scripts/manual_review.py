"""Run a local query and produce a human-reviewable ranked result report.

This is an internal validation tool. It does not claim official AIC ground truth
and does not alter model predictions.

Example (PowerShell):
    python scripts/manual_review.py --query "a man riding a bicycle" --top-k 20
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from data_layer.datastore import LocalDataStore
from data_layer.faiss_store import FAISSFrameStore
from query_engine import BaselineQueryEngine, CLIPTextEncoder
from query_engine.retrieval import ClipCandidateRetriever
from schemas import QueryRequest


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "database" / "aic2026.sqlite"
DEFAULT_INDEX = ROOT / "indexes" / "clip_vit_b32.faiss"
DEFAULT_MAPPING = ROOT / "indexes" / "clip_vit_b32.mapping.json"
DEFAULT_OUT = ROOT / "manual_reviews"


def _build_engine(db: Path, index: Path, mapping: Path) -> BaselineQueryEngine:
    store = FAISSFrameStore(index, mapping)
    store.load()
    datastore = LocalDataStore(db, clip_index=store)
    retriever = ClipCandidateRetriever(
        datastore,
        CLIPTextEncoder(),
        frame_top_k=5000,
        video_top_k=100,
    )
    return BaselineQueryEngine(retriever)


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return ROOT / path


def _render_html(
    query: str,
    task: str,
    candidates: list[dict[str, Any]],
    datastore: LocalDataStore,
    output: Path,
) -> None:
    cards: list[str] = []
    for item in candidates:
        evidence = item.get("evidence", {})
        keyframe_n = evidence.get("keyframe_n")
        video_id = str(item.get("video_id", ""))
        frame_id = item.get("frame_id", "")
        score = item.get("score", "")
        frame_path = ""
        timestamp = ""
        if keyframe_n is not None:
            frame = datastore.get_frame(video_id, int(keyframe_n))
            if frame is not None:
                frame_path = str(_resolve_path(frame.path))
                timestamp = f"{frame.timestamp:.3f}s"

        image_html = "<p><i>keyframe image unavailable</i></p>"
        if frame_path and Path(frame_path).is_file():
            image_uri = Path(frame_path).resolve().as_uri()
            image_html = (
                f'<img src="{html.escape(image_uri, quote=True)}" '
                'style="max-width:420px;max-height:300px;object-fit:contain">'
            )

        cards.append(
            "<article class='card'>"
            f"<h3>#{html.escape(str(item.get('rank', '')))} — {html.escape(video_id)}</h3>"
            f"{image_html}"
            f"<p>frame={html.escape(str(frame_id))} &nbsp; "
            f"keyframe={html.escape(str(keyframe_n))} &nbsp; "
            f"time={html.escape(timestamp)} &nbsp; "
            f"score={html.escape(str(score))}</p>"
            f"<p>retrieval={html.escape(str(item.get('retrieval_score')))} &nbsp; "
            f"temporal={html.escape(str(item.get('temporal_score')))}</p>"
            f"<p>sources={html.escape(str(evidence.get('sources', [])))}</p>"
            "</article>"
        )

    document = """<!doctype html>
<html><head><meta charset="utf-8"><title>AIC manual review</title>
<style>
body{font-family:Arial,sans-serif;margin:24px;line-height:1.4}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(450px,1fr));gap:14px}
.card{border:1px solid #bbb;border-radius:8px;padding:12px}
code{background:#eee;padding:2px 4px}
img{display:block;margin:8px 0}
</style></head><body>
"""
    document += f"<h1>AIC manual retrieval review</h1><p>Task: <b>{html.escape(task)}</b></p>"
    document += f"<p>Query: <code>{html.escape(query)}</code></p><div class='grid'>"
    document += "".join(cards)
    document += "</div></body></html>"
    output.write_text(document, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a query and create a manual review report")
    parser.add_argument("--query", required=True, help="Natural-language query")
    parser.add_argument("--task", choices=["KIS", "QA"], default="KIS")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0")

    engine = _build_engine(args.db, args.index, args.mapping)
    result = engine.search(
        QueryRequest(
            query_id="manual-review",
            task=args.task,
            description=args.query,
        )
    )
    if result.status != "completed":
        raise RuntimeError(result.error or "query failed")

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "query": args.query,
        "task": result.task,
        "status": result.status,
        "candidates": result.candidates[: args.top_k],
    }
    json_path = args.out / "latest.json"
    html_path = args.out / "latest.html"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _render_html(args.query, result.task, payload["candidates"], engine.retriever.datastore, html_path)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"JSON report: {json_path}")
    print(f"HTML report: {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
