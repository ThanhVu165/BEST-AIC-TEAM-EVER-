"""Run the real KIS query engine against a local AIC data package."""
from __future__ import annotations

import argparse
import base64
import html
import json
import time
from io import BytesIO
from pathlib import Path

from data_layer.datastore import LocalDataStore
from data_layer.faiss_store import FAISSFrameStore
from query_engine.clip_encoder import CLIPTextEncoder
from query_engine.engine import BaselineQueryEngine
from query_engine.retrieval import ClipCandidateRetriever
from query_engine.semantic_reranker import SemanticRerankConfig, SigLIP2ImageTextScorer
from schemas import QueryRequest

ROOT = Path(__file__).resolve().parents[1]


def _preflight(datastore: LocalDataStore, index: FAISSFrameStore) -> None:
    """Validate DB/index/source-video access before loading ML models."""
    if index.index is None or index.index.ntotal <= 0:
        raise RuntimeError("FAISS index is missing or empty")
    first = index.mapping[0]
    video_id = str(first["video_id"])
    frame_id = int(first["frame_id"])
    video = datastore.get_video(video_id)
    if video is None:
        raise RuntimeError(f"SQLite has no video record for {video_id}")
    if datastore.get_frame_by_id(video_id, frame_id) is None:
        raise RuntimeError(f"SQLite has no frame record for {video_id}:{frame_id}")
    if datastore.read_source_frame(video_id, frame_id) is None:
        raise RuntimeError(
            f"Cannot decode source frame {video_id}:{frame_id} from {video.path!r}. "
            "Check --project-root and the local videos directory."
        )
    print("preflight=ok")
    print(f"videos_record_sample={video_id}")
    print(f"source_frame_sample={video_id}:{frame_id}")
    print(f"faiss_vectors={index.index.ntotal}")


def _make_html(query: str, candidates: list[dict], datastore: LocalDataStore, output: Path, max_cards: int) -> None:
    cards: list[str] = []
    for candidate in candidates[:max_cards]:
        video_id = str(candidate["video_id"])
        frame_id = int(candidate["frame_id"])
        image = datastore.read_source_frame(video_id, frame_id)
        image_html = "<div class='missing'>frame unavailable</div>"
        if image is not None:
            try:
                from PIL import Image
                pil = Image.fromarray(image)
                buf = BytesIO()
                pil.save(buf, format="JPEG", quality=78, optimize=True)
                encoded = base64.b64encode(buf.getvalue()).decode("ascii")
                image_html = f"<img src='data:image/jpeg;base64,{encoded}' loading='lazy'>"
            except Exception as exc:  # pragma: no cover
                image_html = f"<div class='missing'>image error: {html.escape(str(exc))}</div>"
        evidence = candidate.get("evidence") or {}
        cards.append(
            "<article class='card'>"
            f"<div class='rank'>#{candidate.get('rank', '?')}</div>{image_html}"
            "<div class='meta'>"
            f"<b>{html.escape(video_id)}</b> · frame {frame_id}<br>"
            f"final={float(candidate.get('score', 0.0)):.6f} · retrieval={float(candidate.get('retrieval_score') or 0.0):.6f} · "
            f"temporal={float(candidate.get('temporal_score') or 0.0):.6f}<br>"
            f"semantic={float(evidence.get('semantic_score') or 0.0):.6f} · "
            f"sources={html.escape(', '.join(evidence.get('sources', [])))}"
            "</div></article>"
        )
    document = """<!doctype html><html><head><meta charset="utf-8"><title>AIC KIS local inspection</title>
<style>body{font-family:system-ui,sans-serif;margin:24px;background:#f5f5f5;color:#222}h1{font-size:22px}.query{padding:12px;background:white;border-radius:8px;margin-bottom:18px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px}.card{background:white;border-radius:10px;overflow:hidden;box-shadow:0 1px 4px #bbb;position:relative}.card img{display:block;width:100%;aspect-ratio:16/9;object-fit:cover;background:#ddd}.rank{position:absolute;z-index:2;top:8px;left:8px;background:#111;color:white;padding:4px 8px;border-radius:6px;font-weight:700}.meta{padding:10px;font-size:13px;line-height:1.55}.missing{height:158px;display:grid;place-items:center;background:#ddd;color:#666}</style></head><body>
<h1>AIC 2026 — local KIS inspection</h1><div class="query"><b>Query:</b> __QUERY__<br><b>Candidates:</b> __COUNT__</div><div class="grid">__CARDS__</div></body></html>"""
    document = document.replace("__QUERY__", html.escape(query)).replace("__COUNT__", str(len(candidates))).replace("__CARDS__", "".join(cards))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run KIS against the local AIC2026 data package")
    parser.add_argument("query", nargs="?", help="Natural-language KIS query")
    parser.add_argument("--db", type=Path, default=ROOT / "database" / "aic2026.sqlite")
    parser.add_argument("--index", type=Path, default=ROOT / "indexes" / "clip_vit_b32.faiss")
    parser.add_argument("--mapping", type=Path, default=ROOT / "indexes" / "clip_vit_b32.mapping.json")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frame-top-k", type=int, default=5000)
    parser.add_argument("--final-limit", type=int, default=100)
    parser.add_argument("--max-per-video", type=int, default=10)
    parser.add_argument("--fine-anchors", type=int, default=32)
    parser.add_argument("--fine-radius", type=int, default=8)
    parser.add_argument("--semantic-candidates", type=int, default=50)
    parser.add_argument("--semantic-weight", type=float, default=0.15)
    parser.add_argument("--semantic-model", default="google/siglip2-base-patch16-256")
    parser.add_argument("--no-semantic", action="store_true", help="Run CLIP/temporal KIS only")
    parser.add_argument("--no-temporal", action="store_true", help="Disable source-video frame refinement")
    parser.add_argument("--preflight-only", action="store_true", help="Validate DB/index/source-frame access without loading ML models")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--html-out", type=Path)
    parser.add_argument("--html-limit", type=int, default=30)
    args = parser.parse_args()

    index = FAISSFrameStore(args.index, args.mapping)
    index.load()
    datastore = LocalDataStore(args.db, index, project_root=args.project_root)
    _preflight(datastore, index)
    if args.preflight_only:
        return 0
    if not args.query or not args.query.strip():
        raise SystemExit("query must not be empty")
    if args.device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is unavailable; use --device cpu or install CUDA PyTorch")

    embedder = CLIPTextEncoder(device=args.device)
    semantic_config = SemanticRerankConfig(enabled=not args.no_semantic, model_id=args.semantic_model, candidate_limit=args.semantic_candidates, weight=args.semantic_weight)
    semantic_scorer = None if args.no_semantic else SigLIP2ImageTextScorer(model_id=args.semantic_model, device=args.device)
    engine = BaselineQueryEngine(
        retriever=ClipCandidateRetriever(datastore, embedder, frame_top_k=args.frame_top_k),
        image_encoder=None if args.no_temporal else embedder,
        semantic_scorer=semantic_scorer,
        semantic_config=semantic_config,
        final_limit=args.final_limit,
        max_kis_candidates_per_video=args.max_per_video,
        fine_temporal_anchors=args.fine_anchors,
        fine_temporal_radius=args.fine_radius,
    )
    request = QueryRequest(query_id="local-kis", task="KIS", description=args.query, raw_text=args.query)
    start = time.perf_counter()
    response = engine.search(request)
    elapsed = time.perf_counter() - start
    print(f"status={response.status}")
    print(f"query={args.query}")
    print(f"candidates={len(response.candidates)}")
    print(f"elapsed_sec={elapsed:.3f}")
    if response.error:
        print(f"error={response.error}")
        return 1
    for candidate in response.candidates:
        evidence = candidate.get("evidence") or {}
        print(f"rank={int(candidate['rank']):03d} score={float(candidate['score']):.6f} video={candidate['video_id']} frame={candidate['frame_id']} retrieval={float(candidate.get('retrieval_score') or 0.0):.6f} temporal={float(candidate.get('temporal_score') or 0.0):.6f} semantic={float(evidence.get('semantic_score') or 0.0):.6f}")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(response.candidates, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"json={args.json_out.resolve()}")
    if args.html_out:
        _make_html(args.query, response.candidates, datastore, args.html_out, args.html_limit)
        print(f"html={args.html_out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
