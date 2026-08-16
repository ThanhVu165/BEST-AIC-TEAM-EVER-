"""Evaluate ranked KIS/QA/TRAKE predictions with AIC-style R-Score."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from query_engine.task_evaluation import DEFAULT_KS, evaluate_ranked


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate AIC task predictions")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("input contains no query records")

    reports: list[dict[str, Any]] = []
    for row in rows:
        report = evaluate_ranked(
            row["task"],
            row.get("predictions", [])[:100],
            row["ground_truth"],
            DEFAULT_KS,
        )
        reports.append(
            {
                "query_id": row.get("query_id"),
                "task": row["task"],
                **report,
            }
        )

    aggregate = {
        f"R@{k}": sum(row[f"R@{k}"] for row in reports) / len(reports)
        for k in DEFAULT_KS
    }
    aggregate["FinalScore"] = sum(
        row["FinalScore"] for row in reports
    ) / len(reports)
    payload = {"queries": len(reports), "aggregate": aggregate, "per_query": reports}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
