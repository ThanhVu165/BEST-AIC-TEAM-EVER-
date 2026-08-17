"""Optional answer extraction behind a stable interface.

The Query Engine must localize evidence before asking a VLM for an answer. This
module intentionally does not ship a hard-coded model choice; model weights
and hardware are deployment concerns.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class AnswerEvidence:
    video_id: str
    frame_id: int
    frame_path: str | None
    question: str
    image: Any | None = None


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    confidence: float | None
    status: str


class AnswerExtractor(Protocol):
    """Extract an answer from one already-localized frame."""

    def answer(self, evidence: AnswerEvidence) -> AnswerResult:
        ...


class UnavailableAnswerExtractor:
    """Explicit fallback used when no VLM answer model is configured."""

    def answer(self, evidence: AnswerEvidence) -> AnswerResult:
        return AnswerResult(
            answer="",
            confidence=None,
            status="model_unavailable",
        )
