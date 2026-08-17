"""Lightweight semantic decomposition and evidence scoring.

This is an inspectable baseline, not a learned action/relation model. It makes
query semantics explicit so later CLIP/VLM rerankers can replace individual
signals without changing the pipeline contract.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

from .query_understanding import QuerySpec

# Conservative lexical groups. They are deliberately small: false semantic
# matches are worse than leaving a concept unresolved for the visual model.
_ACTIONS = frozenset({
    "ride", "riding", "rides", "rided", "repair", "repairing", "fix", "fixing",
    "sit", "sitting", "stand", "standing", "walk", "walking", "run", "running",
    "hold", "holding", "carry", "carrying", "open", "opening", "close", "closing",
    "eat", "eating", "drink", "drinking", "talk", "talking", "speak", "speaking",
    "write", "writing", "read", "reading", "drive", "driving", "enter", "entering",
    "leave", "leaving", "throw", "throwing", "catch", "catching", "play", "playing",
})
_RELATIONS = frozenset({
    "at", "on", "in", "near", "behind", "front", "beside", "next", "under", "over",
    "with", "inside", "outside", "between", "against", "toward", "towards",
})
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "to", "is", "are", "was", "were", "be",
    "being", "this", "that", "there", "here", "what", "who", "where", "when", "how",
    "which", "with", "from", "for", "into", "than", "then", "someone", "something",
})


@dataclass(frozen=True)
class SemanticQuery:
    entities: tuple[str, ...]
    actions: tuple[str, ...]
    relations: tuple[str, ...]
    attributes: tuple[str, ...]


def decompose_query(spec: QuerySpec) -> SemanticQuery:
    tokens = [token.casefold() for token in spec.tokens]
    actions = tuple(token for token in tokens if token in _ACTIONS)
    relations = tuple(token for token in tokens if token in _RELATIONS)
    entities = tuple(token for token in tokens if token not in _STOPWORDS and token not in _ACTIONS and token not in _RELATIONS)
    # Adjectives/attributes are kept as residual semantic tokens. This is a
    # baseline heuristic; a learned parser can later populate these explicitly.
    attributes = tuple(token for token in entities if token.endswith(("ing", "ed", "ive", "al", "ous")))
    return SemanticQuery(
        entities=entities,
        actions=actions,
        relations=relations,
        attributes=attributes,
    )


def semantic_score(
    spec: QuerySpec,
    *,
    object_score: float = 0.0,
    metadata_score: float = 0.0,
    ocr_score: float = 0.0,
    asr_score: float = 0.0,
    temporal_score: float = 0.0,
) -> float:
    """Return a bounded semantic-evidence score for reranking.

    The score intentionally does not re-use the raw CLIP retrieval score. It
    combines auxiliary semantic evidence and temporal verification so the
    retrieval score can remain a separate, single-counted signal.
    """
    semantic = decompose_query(spec)
    auxiliary = (
        0.45 * float(object_score)
        + 0.20 * float(metadata_score)
        + 0.15 * float(ocr_score)
        + 0.20 * float(asr_score)
    )
    # Queries containing an explicit action/relation need stronger temporal
    # verification. For simple entity queries, auxiliary evidence is enough.
    if semantic.actions or semantic.relations:
        auxiliary = 0.65 * auxiliary + 0.35 * float(temporal_score)
    else:
        auxiliary = 0.85 * auxiliary + 0.15 * float(temporal_score)
    return max(0.0, min(1.0, auxiliary))
