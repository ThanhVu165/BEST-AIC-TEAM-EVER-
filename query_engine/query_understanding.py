"""Structured query understanding for the AIC 2026 Query Engine.

This module stays model-agnostic. It converts the shared QueryRequest into a
stable semantic representation consumed by retrieval/reranking stages.
"""
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

    return QuerySpec(
        query_id=request.query_id,
        task=resolved_task,
        text=text,
        description=description,
        question=question,
        events=events,
        tokens=tuple(_tokens(text)),
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
