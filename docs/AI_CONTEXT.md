# AI Context / Source of Truth

This file is intentionally concise enough to paste into an AI coding assistant, while pointing to the detailed source-of-truth documents.

## Project

AIC 2026 Video Retrieval system. Current official preliminary-round data is Batch 1. BTC has not supplied the official preliminary query file yet.

## Official task types

1. Textual KIS: retrieve correct video and a frame inside the GT event interval.
2. Q&A: retrieve correct video, a frame inside the GT interval, and the semantically correct answer.
3. TRAKE: retrieve correct video and one semantic keyframe for each queried event. Wrong video gives zero R-Score. Correct video is scored by fraction of events whose submitted frames fall in the corresponding GT intervals.

Final Score uses the best R-Score seen at ranks 1, 5, 20, 50 and 100. Therefore outputs must be ranked candidate lists, up to 100 entries.

## Dataset facts currently known

Team audit of official Batch 1:

- 873 videos represented by keyframes
- 177,321 keyframes
- 873 CLIP feature files
- 873 mapping CSV files
- 873 media-info JSON files
- 177,321 object JSON files

BTC says Video is the official competition data. Keyframes, Objects, CLIP features and Metadata are auxiliary/supporting data.

## Current machine

Primary machine: Windows 11, Intel i5-12450H, 32 GB RAM, NVIDIA RTX 4050 Laptop GPU with ~6 GB dedicated VRAM.

Prefer local/offline execution and staged retrieval. Avoid requiring a huge model across the whole corpus.

## Team modules

- `video_pipeline/`: Person 1. Offline ingestion, frame mapping, BTC features/objects/metadata, SQLite, FAISS, exact frame access, optional OCR/ASR.
- `query_engine/`: Person 2. Query understanding, multimodal retrieval, temporal localization, reranking, KIS/Q&A/TRAKE, top-100 ranking, submission formatting.
- `ui/`: Person 3. Streamlit UI only; uses FastAPI.

## Contract rules

- Shared schemas are in `schemas/` and are authoritative.
- Data access is behind interfaces; Query Engine must not hard-code SQLite or FAISS implementation details.
- UI must not import Query Engine internals.
- Submission formatting is separate from retrieval/inference.
- OCR and ASR are optional and must not block baseline execution.
- Do not assume any unofficial query dataset is the official AIC 2026 query specification.

## Coding behavior for AI assistants

Before editing code:

1. Read `docs/PROJECT_CONTEXT.md`.
2. Read `docs/ARCHITECTURE.md`.
3. Read `docs/DATA_CONTRACT.md`.
4. Read `docs/API_CONTRACT.md`.
5. Read `docs/TEAM_WORKFLOW.md`.
6. State the module and task being changed.
7. Preserve public contracts unless the task explicitly asks for a contract change.

Do not:

- invent unsupported BTC requirements
- silently change `frame_id` semantics
- commit dataset/model artifacts or secrets
- rewrite another person's module as a shortcut
- choose a new model/storage technology without documenting the reason

When uncertain, inspect the repository and contracts first; ask rather than inventing a convention that affects integration.
