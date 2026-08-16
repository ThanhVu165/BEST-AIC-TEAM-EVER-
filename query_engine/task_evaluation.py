"""Competition-aligned task scoring for local validation.

The formulas mirror the AIC 2026 preliminary-round description:
- KIS: correct video + frame inside the accepted interval.
- QA: KIS conditions + answer match.
- TRAKE: correct video + fraction of event frames inside their intervals.
- Final Score: mean of the best R-Score at ranks 1, 5, 20, 50, 100.

Answer matching is intentionally explicit and conservative. The BTC document
requires semantic answer matching but does not specify an automatic text
metric, so this module supports an exact normalized match or an accepted-answer
list rather than pretending to reproduce the hidden semantic judge.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any

DEFAULT_KS = (1, 5, 20, 50, 100)


def kis_r_score(candidate: Any, ground_truth: dict[str, Any]) -> float:
    """Score one Textual KIS candidate in [0, 1]."""
    video_id = _get(candidate, "video_id")
    frame_id = _get_int(candidate, "frame_id")
    start, end = _frame_interval(ground_truth)
    return float(video_id == str(ground_truth["video_id"]) and start <= frame_id <= end)


def qa_r_score(candidate: Any, ground_truth: dict[str, Any]) -> float:
    """Score one QA candidate using explicit accepted answers."""
    if kis_r_score(candidate, ground_truth) == 0.0:
        return 0.0
    answer = _get(candidate, "answer")
    accepted = ground_truth.get("accepted_answers")
    if accepted is None:
        accepted = [ground_truth.get("answer", "")]
    return float(_normalize(answer) in {_normalize(item) for item in accepted})


def trake_r_score(candidate: Any, ground_truth: dict[str, Any]) -> float:
    """Score one TRAKE candidate as the fraction of correctly localized events."""
    if _get(candidate, "video_id") != str(ground_truth["video_id"]):
        return 0.0

    predicted = _event_predictions(candidate)
    gt_events = ground_truth.get("events", [])
    if not gt_events:
        raise ValueError("TRAKE ground truth must contain events")

    correct = 0
    for event in gt_events:
        event_id = str(event["event_id"])
        frame_id = predicted.get(event_id)
        if frame_id is None:
            continue
        start, end = _frame_interval(event)
        correct += int(start <= frame_id <= end)
    return correct / len(gt_events)


def final_score(r_scores: Sequence[float], ks: Sequence[int] = DEFAULT_KS) -> dict[str, float]:
    """Compute R@K as max R-Score in the first K predictions and their mean."""
    if not r_scores:
        return {f"R@{k}": 0.0 for k in ks} | {"FinalScore": 0.0}
    values = [float(score) for score in r_scores]
    top_scores = {
        f"R@{k}": max(values[:k], default=0.0)
        for k in ks
    }
    top_scores["FinalScore"] = sum(top_scores.values()) / len(top_scores)
    return top_scores


def evaluate_ranked(
    task: str,
    predictions: Iterable[Any],
    ground_truth: dict[str, Any],
    ks: Sequence[int] = DEFAULT_KS,
) -> dict[str, float]:
    """Evaluate a ranked candidate list for one query."""
    task_name = task.upper()
    if task_name in {"KIS", "TKIS"}:
        scores = [kis_r_score(item, ground_truth) for item in predictions]
    elif task_name == "QA":
        scores = [qa_r_score(item, ground_truth) for item in predictions]
    elif task_name == "TRAKE":
        scores = [trake_r_score(item, ground_truth) for item in predictions]
    else:
        raise ValueError(f"Unsupported task: {task}")
    return final_score(scores, ks)


def _get(candidate: Any, key: str) -> str:
    if isinstance(candidate, dict):
        return str(candidate[key])
    return str(getattr(candidate, key))


def _get_int(candidate: Any, key: str) -> int:
    return int(_get(candidate, key))


def _frame_interval(record: dict[str, Any]) -> tuple[int, int]:
    start = record.get("frame_start", record.get("start_frame", record.get("start")))
    end = record.get("frame_end", record.get("end_frame", record.get("end")))
    if start is None or end is None:
        raise ValueError("ground truth event requires frame_start/frame_end")
    start_int = int(start)
    end_int = int(end)
    if end_int < start_int:
        raise ValueError("ground truth frame interval is reversed")
    return start_int, end_int


def _event_predictions(candidate: Any) -> dict[str, int]:
    events = candidate.get("events", []) if isinstance(candidate, dict) else getattr(candidate, "events", [])
    output: dict[str, int] = {}
    for event in events:
        if isinstance(event, dict):
            output[str(event["event_id"])] = int(event["frame_id"])
        else:
            output[str(event.event_id)] = int(event.frame_id)
    return output


def _normalize(value: Any) -> str:
    text = str(value or "").casefold().strip()
    text = re.sub(r"\s+", " ", text)
    return text
