# Architecture — AIC 2026

**Canonical pipeline:** `docs/AIC2026_CANONICAL_PIPELINE.md`

This file defines subsystem boundaries. The canonical document defines the single end-to-end retrieval pipeline. Do not interpret this file as a second architecture.

## 1. High-level system

```text
                         OFFLINE

Official Videos + BTC supporting data
              |
              v
+---------------------------+
| Video / Data Pipeline     |  Person 1
| - ingest + timeline       |
| - frame/keyframe mapping  |
| - BTC CLIP validation     |
| - objects + metadata      |
| - SQLite + FAISS          |
| - exact frame access      |
| - optional OCR/ASR        |
+-------------+-------------+
              |
              | stable DataStore contract
              v
+---------------------------+
| Query Engine              |  Person 2
| - query understanding     |
| - candidate retrieval     |
| - video ranking           |
| - temporal localization   |
| - semantic keyframe       |
| - multimodal reranking    |
| - KIS / Q&A / TRAKE       |
| - Top-100 ranking         |
+-------------+-------------+
              |
              | FastAPI /api/v1
              v
+---------------------------+
| Streamlit UI              |  Person 3
| - operator/debug UI       |
| - result viewer           |
| - submission export       |
+---------------------------+
```

## 2. One pipeline, three ownership domains

There is only one end-to-end pipeline. The three people own different stages of it:

```text
Video/Data offline
      -> Query Engine online
      -> API
      -> UI
```

Experimental implementations belong inside their stage. Do not create `pipeline_v2`, `new_pipeline`, or another independent orchestrator.

## 3. Ownership boundaries

### Person 1: Video/Data Pipeline

Owns:

- scan/ingest video files
- build video manifest and timeline
- validate BTC keyframe/frame mappings
- validate and index BTC CLIP features
- normalize BTC object JSON
- normalize metadata
- SQLite database
- FAISS/vector indexes and internal-id mappings
- exact frame access from source video/keyframe mapping
- optional OCR and ASR artifacts

Does not own query semantics, task-specific ranking policy, answer generation or UI.

### Person 2: Query Engine

Owns:

```text
Natural Language / structured query
 -> query understanding
 -> candidate video retrieval
 -> video-level aggregation/ranking
 -> coarse-to-fine temporal localization
 -> semantic keyframe selection
 -> multimodal reranking
 -> KIS / Q&A / TRAKE
 -> Top-100 candidate ranking
```

The Query Engine consumes `DataStore`/shared schemas and must not depend on SQLite/FAISS implementation details.

### Person 3: UI

Owns Streamlit pages/components, API client, result visualization, debug display and submission export.

UI must use FastAPI and must not import Query Engine internals or access SQLite/FAISS directly.

## 4. Runtime model

Primary target is one local machine with constrained VRAM. Expensive VLM/video models must run after candidate reduction, not over the full corpus.

```text
Streamlit :8501
FastAPI   :8000
Query Engine module/process
SQLite + FAISS on local disk
Video/data on local disk
```

The baseline must not require external network services.

## 5. Data flow

### Offline

```text
Video + BTC supporting data
  -> validation
  -> video/timeline/frame mapping
  -> normalized auxiliary records
  -> SQLite
  -> BTC CLIP FAISS index + other optional indexes
```

### Online

```text
QueryRequest
  -> query understanding
  -> visual candidate retrieval
  -> video aggregation/ranking
  -> temporal localization
  -> semantic keyframe / answer / event alignment
  -> multimodal reranking
  -> ranked candidates <= 100
  -> submission formatting
```

## 6. Retrieval principles

1. BTC CLIP ViT-B/32 is the current validated visual retrieval baseline.
2. Keep multiple hypotheses because evaluation uses R@1/R@5/R@20/R@50/R@100.
3. Use original video for fine temporal evidence when exact event frames matter.
4. Auxiliary objects/metadata/OCR/ASR are evidence signals, not automatic overrides.
5. Any new model/index must be benchmarked and remain replaceable.

## 7. TRAKE principle

TRAKE is a structured temporal event-sequence problem:

```text
candidate video
  -> event hypotheses
  -> temporal localization per event
  -> semantic keyframe candidates
  -> sequence-consistent alignment
  -> final video + event frames
```

If event order is meaningful, preserve `t1 < t2 < ... < tN`. Independent event retrieval can provide hypotheses but is not sufficient as the final alignment policy.

## 8. Shared contracts

The shared contracts in `schemas/`, `data_layer/` and `docs/` define integration boundaries.

- `video_id` identifies a video.
- `frame_id` is the dataset/source frame identifier and must not be silently renumbered.
- timestamp/frame mappings must remain cross-resolvable.
- submission formatting is isolated from inference.

Model names are not encoded into shared schemas.
