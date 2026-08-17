"""Model-agnostic semantic benchmark for action/relation hard negatives.

This tool deliberately keeps model loading behind adapters so benchmark code does
not prescribe a production model. It is intended for local/GPU evaluation.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


@dataclass(frozen=True)
class Pair:
    query: str
    positive: str
    negative: str


class Scorer(Protocol):
    name: str

    def score(self, query: str, candidate: str) -> float: ...


def pairwise_accuracy(scorer: Scorer, pairs: Sequence[Pair]) -> float:
    if not pairs:
        return 0.0
    wins = sum(scorer.score(p.query, p.positive) > scorer.score(p.query, p.negative) for p in pairs)
    return wins / len(pairs)


def mean_margin(scorer: Scorer, pairs: Sequence[Pair]) -> float:
    if not pairs:
        return 0.0
    return sum(scorer.score(p.query, p.positive) - scorer.score(p.query, p.negative) for p in pairs) / len(pairs)


def load_pairs(path: Path) -> list[Pair]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [Pair(str(r["query"]), str(r["positive"]), str(r["negative"])) for r in rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    pairs = load_pairs(args.dataset)
    # Model adapters are intentionally injected by future benchmark runners.
    print(f"loaded {len(pairs)} hard-negative pairs; no model backend selected")


if __name__ == "__main__":
    main()
