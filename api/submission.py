"""Submission (exam-answer) formatter.

Isolated from retrieval/inference on purpose (see docs/API_CONTRACT.md).

The exact BTC column schema for AIC 2026 has not been published inside this
repository's docs (see docs/AI_CONTEXT.md: "invent official BTC query
formats" is explicitly listed as something to avoid). This module therefore
implements the common, publicly documented AIC/HCMC-AI-Challenge answer-file
convention as a **team default** that is easy to swap out in one place if
BTC publishes a different schema:

- KIS   -> one CSV per query, each row: ``video_id,frame_id``
- QA    -> one CSV per query, each row: ``video_id,frame_id,answer``
- TRAKE -> one CSV per query, each row:
           ``video_id,frame_id_event_1,frame_id_event_2,...``
           (columns follow the order events were supplied in the request)

Up to 100 ranked rows per query are written, matching the competition's
"up to 100 answers per query" rule. All CSVs for the requested query_ids are
zipped together into a single downloadable submission package.
"""
from __future__ import annotations

import csv
import io
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from schemas import SearchResponse


def _kis_rows(candidates: list[dict[str, Any]]) -> list[list[Any]]:
    rows = []
    for cand in candidates[:100]:
        frame_id = cand.get("frame_id")
        if frame_id is None:
            continue
        rows.append([cand["video_id"], frame_id])
    return rows


def _qa_rows(candidates: list[dict[str, Any]]) -> list[list[Any]]:
    rows = []
    for cand in candidates[:100]:
        frame_id = cand.get("frame_id")
        if frame_id is None:
            continue
        rows.append([cand["video_id"], frame_id, cand.get("answer", "")])
    return rows


def _trake_rows(candidates: list[dict[str, Any]]) -> list[list[Any]]:
    rows = []
    for cand in candidates[:100]:
        events = cand.get("events") or []
        ordered = sorted(events, key=lambda e: e.get("event_id", ""))
        row: list[Any] = [cand["video_id"]] + [e.get("frame_id") for e in ordered]
        rows.append(row)
    return rows


_ROW_BUILDERS = {
    "KIS": _kis_rows,
    "QA": _qa_rows,
    "TRAKE": _trake_rows,
}


def build_query_csv(result: SearchResponse) -> str:
    """Render one query's ranked candidates as CSV text (no header row,
    matching the common BTC answer-file convention)."""
    builder = _ROW_BUILDERS.get(result.task)
    if builder is None:
        raise ValueError(f"Unsupported task for submission formatting: {result.task}")
    rows = builder(result.candidates)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerows(rows)
    return buffer.getvalue()


def build_submission_zip(results: list[SearchResponse], out_dir: Path) -> Path:
    """Write one CSV per query into a timestamped zip under ``out_dir``.

    Returns the path to the created zip file. Queries with ``status`` other
    than ``completed`` are skipped with a clear placeholder note so a failed
    query never silently becomes an empty (misleadingly "correct") answer.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    zip_path = out_dir / f"submission_{stamp}.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for result in results:
            if result.status != "completed":
                zf.writestr(
                    f"{result.query_id}.SKIPPED.txt",
                    f"query status={result.status} error={result.error or ''}\n",
                )
                continue
            zf.writestr(f"{result.query_id}.csv", build_query_csv(result))

    return zip_path