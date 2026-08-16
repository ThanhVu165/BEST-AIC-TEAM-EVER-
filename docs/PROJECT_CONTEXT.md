# Project Context — AIC 2026 Video Retrieval

## Purpose

Build a local-first Video Retrieval and Video Understanding system for the AIC 2026 preliminary round. The system must support three official task types:

- Textual Known Item Search (Textual KIS)
- Visual Question Answering (Q&A)
- Temporal Retrieval and Alignment of Key Events (TRAKE)

The competition accepts up to 100 answers per query and computes Final Score from R@1, R@5, R@20, R@50 and R@100. Therefore the system is a ranked-candidate system, not a single-prediction classifier.

## Official source constraints

The source document supplied by the team states:

- Textual KIS: correct when video matches GT and frame_id is inside the ground-truth interval.
- Q&A: correct when video, frame interval and semantic answer all match.
- TRAKE: wrong video gives R-Score 0; with correct video, score is the fraction of correctly aligned events. Semantic-event intervals are usually very short, commonly under 10 frames.
- Semantic keyframe is a content/semantic moment, not a codec I-frame.
- Official competition data is Video. Keyframes, Objects, CLIP features and Metadata are auxiliary/supporting data.
- Current provided data is Batch 1; Batch 2 is expected later and must be ingested without changing public contracts.

## Current data audit

Team audit reports Batch 1 contains:

- 873 videos represented by keyframes
- 177,321 keyframes
- 873 CLIP feature files
- 873 mapping CSV files
- 873 media-info JSON files
- 177,321 object JSON files

## Query availability

BTC has not yet supplied an official preliminary-round query file. Any query datasets found from other sources are reference/stress-test material only and must not be treated as official AIC 2026 specification or GT.

## Compute

Primary development/inference machine:

- Windows 11 64-bit
- Intel Core i5-12450H
- 32 GB RAM
- NVIDIA GeForce RTX 4050 Laptop GPU
- ~6 GB dedicated VRAM

Design for local execution and constrained VRAM. Expensive models should be placed after candidate reduction rather than applied to the full corpus.

## Deployment assumption

Initial target is one-machine, local/offline execution. The architecture must remain separable enough to move components to servers later.

## Team split

### Person 1 — Video Processing / Data Layer

Owns offline ingestion, exact frame mapping, BTC data validation, SQLite, FAISS and optional OCR/ASR preprocessing.

### Person 2 — Query Engine

Owns natural-language query parsing, task detection, multimodal retrieval, temporal localization, reranking, KIS/Q&A/TRAKE solvers, candidate generation and submission formatting.

### Person 3 — UI

Owns Streamlit UI and UI-facing integration through FastAPI only.

## Non-negotiable architecture rules

1. Shared schemas are the integration contract.
2. Query Engine must not depend on direct FAISS/SQLite implementation details.
3. UI must not import Query Engine internals.
4. Submission formatting must be isolated from inference.
5. Dataset and model artifacts are never committed to Git.
6. OCR/ASR are optional extensions and must not block the baseline.
7. Do not hard-code third-party query formats as official competition inputs.
8. Every model choice must be replaceable without breaking the shared interfaces.

## AI assistant rule

Any AI assistant working on this repository must read this file and the other documents in `docs/` before changing code. It must state which module it is modifying, honor ownership boundaries, and avoid changing shared contracts unless explicitly tasked to do so.
