# AIC 2026 Video Retrieval

Team repository for the AIC 2026 Video Retrieval system.

## System scope

The system is split into three independently owned parts:

1. `video_pipeline/` — offline video/data processing and indexing (Person 1)
2. `query_engine/` — natural-language query understanding, retrieval, temporal localization, reranking, KIS/Q&A/TRAKE, candidate ranking (Person 2)
3. `ui/` — Streamlit development/competition UI (Person 3)

The integration boundary is defined by the shared contracts in `schemas/` and the API in `api/`.

## Competition requirements

The official AIC 2026 preliminary-round document defines three tasks:

- Textual KIS: output video + frame
- Q&A: output video + frame + answer
- TRAKE: output one video + one semantic keyframe per event

Up to 100 ranked answers may be submitted per query. Final Score is the mean of R@1, R@5, R@20, R@50 and R@100. TRAKE gives zero R-Score when the submitted video is wrong; when the video is correct, the score is the fraction of events whose submitted frame falls in the corresponding ground-truth frame interval.

## Data assumptions

Current official data is Batch 1. The team audit reports:

- 873 videos represented by keyframes
- 177,321 keyframes
- 873 CLIP feature files
- 873 mapping CSV files
- 873 media-info JSON files
- 177,321 object JSON files

Batch 2 is not currently available. The pipeline must therefore support adding future batches without changing the query contracts.

BTC states that Video is the official competition data; Keyframes, Objects, CLIP features and Metadata are auxiliary/supporting data.

## Architecture

```text
                  OFFLINE

Videos ──> Video Processing ──> SQLite + FAISS + feature files
                                      │
                                      │ shared data contract
                                      ▼
Natural Language Query ──> Query Engine ──> ranked top-100 candidates
                                      │
                                      │ REST API
                                      ▼
                                  Streamlit UI
```

### Data layer

- SQLite for structured metadata
- FAISS for vector search
- Raw/large features remain in files
- OCR and ASR are optional extensions and must not block the baseline

### Model policy

The architecture is model-agnostic. Do not hard-code a specific VLM, temporal model, reranker, OCR engine, or ASR engine into shared contracts. Choose and benchmark models inside the owning module.

## Team boundaries

### Person 1 — Video Processing

Owns:

- video manifest
- video/keyframe/frame mapping
- BTC CLIP loading/validation/indexing
- BTC object loading/normalization/indexing
- metadata loading
- exact frame access
- SQLite database
- vector indexes
- optional OCR/ASR preprocessing

Must not own query semantics or task-specific ranking.

### Person 2 — Query Engine

Owns:

```text
Natural Language Query
 -> Query Understanding
 -> Candidate Retrieval
 -> Temporal Localization
 -> Reranking
 -> KIS / Q&A / TRAKE
 -> Top-100 candidate generation
```

Must consume the data layer through stable interfaces rather than depending on FAISS/SQLite implementation details.

### Person 3 — UI

Owns the Streamlit application.

Modes:

- Development/debug mode
- Competition/operator mode

UI is read-only with respect to model results: no human editing/reordering/correction of predictions.

UI must use FastAPI endpoints rather than importing query-engine internals directly.

## Shared contracts

Do not change `schemas/` without team agreement. The important shared concepts are:

- `VideoRecord`
- `FrameRecord`
- `ObjectRecord`
- `QueryRequest`
- `Candidate`
- `KISResult`
- `QAResult`
- `TRAKEEvent`
- `TRAKECandidate`
- `SubmissionRecord`

Internal model scores may be decomposed into retrieval/temporal/reranking components for debugging. Official submission formatting is isolated from inference logic.

## Development principles

1. Keep module boundaries stable.
2. Prefer adapters/interfaces over direct coupling.
3. Do not commit dataset files, video files, generated FAISS indexes, SQLite databases, model weights, or secrets.
4. Use mock implementations so each team member can develop independently.
5. Every change to a shared schema/API requires a documented version update.
6. Baseline first, research optimization second.

## Branching

Recommended branches:

- `main` — stable integration
- `develop` — integration branch
- `feature/video-pipeline`
- `feature/query-engine`
- `feature/ui`

## Local data layout

```text
data/
  raw/
    videos/
    keyframes/
    objects/
    clip_features/
    metadata/
  processed/

 database/
 indexes/
```

The actual dataset lives outside Git. Configure paths in `configs/` or environment variables.

## API v1

The UI-facing API is versioned under `/api/v1`.

Planned endpoints:

- `POST /api/v1/search`
- `GET /api/v1/result/{query_id}`
- `GET /api/v1/video/{video_id}`
- `GET /api/v1/video/{video_id}/frame/{frame_id}`
- `POST /api/v1/submission`

The API must remain independent of internal model choices.

## Evaluation

The evaluator will implement the official scoring logic:

- R@1
- R@5
- R@20
- R@50
- R@100
- Final Score

Third-party/reference queries may be used for internal benchmarking, but must never be presented as official AIC 2026 query/ground-truth data unless supplied by BTC.

## AI-assisted development context

Before using an AI coding assistant, read:

- `docs/PROJECT_CONTEXT.md`
- `docs/ARCHITECTURE.md`
- `docs/DATA_CONTRACT.md`
- `docs/API_CONTRACT.md`
- `docs/TEAM_WORKFLOW.md`
- `docs/AI_CONTEXT.md`

These files are the source of truth for shared assumptions. AI assistants should modify only the module owned by the current task unless the task explicitly changes a shared contract.
