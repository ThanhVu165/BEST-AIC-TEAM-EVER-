"""Run one real Query Engine retrieval against the local AIC package."""
from __future__ import annotations

import argparse
from pathlib import Path

from query_engine.engine import BaselineQueryEngine
from query_engine.runtime import build_clip_baseline_engine
from schemas import QueryRequest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "database" / "aic2026.sqlite"
DEFAULT_INDEX_PATH = PROJECT_ROOT / "indexes" / "clip_vit_b32.faiss"
DEFAULT_MAPPING_PATH = PROJECT_ROOT / "indexes" / "clip_vit_b32.mapping.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="natural-language retrieval query")
    parser.add_argument("--task", choices=("KIS", "QA"), default="KIS")
    parser.add_argument("--question", default=None)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--mapping-path", type=Path, default=DEFAULT_MAPPING_PATH)
    args = parser.parse_args()

    engine: BaselineQueryEngine = build_clip_baseline_engine(
        db_path=args.db_path,
        index_path=args.index_path,
        mapping_path=args.mapping_path,
    )
    request = QueryRequest(
        query_id="smoke-001",
        task=args.task,
        description=args.query,
        question=args.question,
    )
    result = engine.search(request)
    print(result.model_dump_json(indent=2, exclude_none=True))
    return 0 if result.status == "completed" and result.candidates else 2


if __name__ == "__main__":
    raise SystemExit(main())
