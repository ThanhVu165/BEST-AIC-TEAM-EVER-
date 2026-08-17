"""Structured query understanding for the AIC 2026 Query Engine."""
from __future__ import annotations

from dataclasses import dataclass, field
import re

from schemas import QueryRequest


@dataclass(frozen=True)
class QueryEventSpec:
    event_id: str
    description: str


@dataclass(frozen=True)
class QuerySpec:
    query_id: str
    task: str
    text: str
    description: str | None = None
    question: str | None = None
    events: tuple[QueryEventSpec, ...] = field(default_factory=tuple)
    tokens: tuple[str, ...] = field(default_factory=tuple)
    entities: tuple[str, ...] = field(default_factory=tuple)
    actions: tuple[str, ...] = field(default_factory=tuple)
    relations: tuple[str, ...] = field(default_factory=tuple)
    attributes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def has_multiple_events(self) -> bool:
        return len(self.events) > 1


def understand_query(request: QueryRequest, *, task: str | None = None) -> QuerySpec:
    resolved_task = (task or request.task or _infer_task(request)).upper()
    if resolved_task not in {"KIS", "QA", "TRAKE"}:
        raise ValueError(f"Unsupported task: {resolved_task}")

    description = (request.description or "").strip() or None
    question = (request.question or "").strip() or None
    raw_text = (request.raw_text or "").strip()

    parts: list[str] = []
    for value in (description, raw_text):
        if value and value not in parts:
            parts.append(value)
    if resolved_task == "QA" and question and question not in parts:
        parts.append(question)

    text = " ".join(parts).strip()
    if resolved_task == "TRAKE":
        events = tuple(
            QueryEventSpec(event_id=event.event_id, description=event.description.strip())
            for event in request.events
        )
        if not events:
            raise ValueError("TRAKE requires at least one event")
        if any(not event.description for event in events):
            raise ValueError("TRAKE event descriptions must not be empty")
        if not text:
            text = " ".join(event.description for event in events)
    else:
        events = tuple()

    if not text:
        raise ValueError("query request contains no searchable text")

    tokens = tuple(_tokens(text))
    entities, actions, relations, attributes = _decompose(tokens)
    return QuerySpec(
        query_id=request.query_id,
        task=resolved_task,
        text=text,
        description=description,
        question=question,
        events=events,
        tokens=tokens,
        entities=entities,
        actions=actions,
        relations=relations,
        attributes=attributes,
    )


def _infer_task(request: QueryRequest) -> str:
    if request.events:
        return "TRAKE"
    if request.question:
        return "QA"
    return "KIS"


def _tokens(text: str) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for token in re.findall(r"[\w-]+", text.casefold()):
        if len(token) <= 1 or token in seen:
            continue
        seen.add(token)
        output.append(token)
    return output


def _decompose(tokens: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    # Keep this parser deterministic and model-free. The semantic.py module owns
    # the reusable vocabulary/scoring baseline; these fields make the contract
    # explicit for downstream learned rerankers.
    actions = frozenset({
        "ride", "riding", "rides", "repair", "repairing", "fix", "fixing", "sit", "sitting",
        "stand", "standing", "walk", "walking", "run", "running", "hold", "holding", "carry",
        "carrying", "open", "opening", "close", "closing", "eat", "eating", "drink", "drinking",
        "talk", "talking", "speak", "speaking", "write", "writing", "read", "reading", "drive",
        "driving", "enter", "entering", "leave", "leaving", "throw", "throwing", "catch", "catching",
        "play", "playing",
    })
    relations = frozenset({
        "at", "on", "in", "near", "behind", "front", "beside", "next", "under", "over", "with",
        "inside", "outside", "between", "against", "toward", "towards",
    })
    stopwords = frozenset({
        "a", "an", "the", "and", "or", "of", "to", "is", "are", "was", "were", "be", "being",
        "this", "that", "there", "here", "what", "who", "where", "when", "how", "which", "with",
        "from", "for", "into", "than", "then", "someone", "something",
    })
    action_tokens = tuple(token for token in tokens if token in actions)
    relation_tokens = tuple(token for token in tokens if token in relations)
    entity_tokens = tuple(token for token in tokens if token not in stopwords and token not in actions and token not in relations)
    attribute_tokens = tuple(token for token in entity_tokens if token.endswith(("ing", "ed", "ive", "al", "ous")))
    return entity_tokens, action_tokens, relation_tokens, attribute_tokens
