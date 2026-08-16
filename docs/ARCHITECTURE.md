# Architecture v1

## 1. High-level system

```text
                           OFFLINE

Videos + BTC auxiliary data
            |
            v
+-------------------------+
| Video Processing        |  Person 1
| - ingest                 |
| - frame mapping          |
| - feature validation     |
| - optional OCR/ASR       |
| - database               |
| - vector indexes         |
+------------+-------------+
             |
             | Stable data interfaces
             v
+-------------------------+
| Query Engine             |  Person 2
| - query understanding    |
| - candidate retrieval    |
| - temporal localization  |
| - reranking              |
| - KIS/Q&A/TRAKE          |
| - top-100 ranking        |
+------------+-------------+
             |
             | REST /api/v1
             v
+-------------------------+
| Streamlit UI             |  Person 3
| - dev/debug mode         |
| - competition mode       |
| - result viewer          |
| - submission export      |
+-------------------------+
```

## 2. Ownership boundaries

### Person 1: Video Processing

Owns `video_pipeline/`, the SQLite database and FAISS indexes.

Responsibilities:

- scan/ingest video files
- build video manifest
- validate BTC keyframe/frame mappings
- validate and index BTC CLIP features
- normalize BTC object JSON
- normalize metadata
- provide exact frame access from original video/keyframe mapping
- optionally build OCR and ASR artifacts

Does not own query semantics, task-specific ranking or answer generation.

### Person 2: Query Engine

Owns `query_engine/`.

Responsibilities:

```text
Natural Language Query
 -> normalize/parse
 -> task detection (when needed)
 -> query representation
 -> candidate video retrieval
 -> temporal localization
 -> fine-grained keyframe selection
 -> multimodal reranking
 -> task solver
 -> candidate generation and ranking (<=100)
 -> submission formatting
```

The Query Engine depends on abstractions from `schemas/` and data-access interfaces, not on the physical details of SQLite/FAISS files.

### Person 3: UI

Owns `ui/`.

Responsibilities:

- Streamlit pages/components
- API client
- video/frame result visualization
- debug information display
- candidate ranking display
- submission export trigger

UI must use FastAPI and must not import query-engine internals.

## 3. Runtime model

Primary deployment is one machine. Services may run as separate processes:

```text
Streamlit :8501
FastAPI   :8000
Query Engine process/module
SQLite + FAISS on local disk
Video data on local disk
```

The system should not require a network or external API for the baseline.

## 4. Data flow

### Offline

```text
Video
  -> video manifest
  -> frame mapping
  -> normalized auxiliary artifacts
  -> SQLite
  -> FAISS/vector indexes
```

### Online

```text
QueryRequest
  -> query understanding
  -> multi-signal candidate retrieval
  -> temporal localization
  -> reranking
  -> task-specific result
  -> ranked candidate list (max 100)
```

## 5. Retrieval architecture

Use layered retrieval:

```text
Stage 1: dense/coarse retrieval
Stage 2: multimodal candidate expansion/filtering
Stage 3: temporal localization
Stage 4: fine-grained reranking/keyframe selection
Stage 5: task-specific answer/alignment
Stage 6: candidate ranking/diversification
```

The exact model at each stage is not part of the contract.

## 6. Why layered retrieval

The available machine has an RTX 4050 Laptop GPU with approximately 6 GB dedicated VRAM. Expensive VLM/video models should therefore be used after candidate reduction, not across the entire corpus.

## 7. Candidate principle

The system must preserve multiple ranked hypotheses because BTC evaluates R@1/R@5/R@20/R@50/R@100. Do not discard alternatives too early.

## 8. TRAKE principle

TRAKE is a structured event-sequence alignment problem:

```text
video candidate
  -> event 1 localization
  -> event 2 localization
  -> ...
  -> event N localization
  -> sequence-consistent alignment
```

When the query semantics imply event order, alignment should preserve that order. The final TRAKE candidate represents one video plus all event frame predictions.

## 9. Replaceability

The following must be replaceable through interfaces/adapters:

- vector store
- embedding model
- OCR
- ASR
- temporal model
- reranker
- VLM

Do not make shared schemas encode a specific model name.
