"""Run one real Query Engine retrieval against the local AIC package."""
from __future__ import annotations

import argparse

from query_engine.engine import BaselineQueryEngine
from query_engine.runtime import build_clip_baseline_engine
from schemas import QueryRequest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="natural-language retrieval query")
    parser.add_argument("--task", choices=("KIS", "QA"), default="KIS")
    parser.add_argument("--question", default=None)
    args = parser.parse_args()

    engine: BaselineQueryEngine = build_clip_baseline_engine()
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
