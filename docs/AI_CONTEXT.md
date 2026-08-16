# AI Context — AIC 2026 Video Retrieval

This is the compact **source of truth for AI coding assistants** working on this repository. Before changing code, read this file and the linked contracts. Do not infer missing BTC requirements from unofficial examples.

## 1. Project identity

- Competition: AIC 2026, preliminary-round Video Retrieval / Video Understanding system.
- Repository: `ThanhVu165/BEST-AIC-TEAM-EVER-`.
- Current official data available to the team: **Batch 1**.
- Batch 2: BTC has not supplied it yet.
- Official query file for the preliminary round: BTC has not supplied one yet.
- Query files collected independently by the team are **reference/stress-test material only**, not official specification.

## 2. Official task semantics currently established

The system must support three task families:

1. **KIS / Textual Known Item Search** — retrieve the correct video and a frame inside the ground-truth event interval.
2. **Q&A** — retrieve the correct video, a frame inside the ground-truth event interval, and a semantically correct answer.
3. **TRAKE** — retrieve the correct video and one semantic keyframe for each queried event. Wrong video gives zero R-Score. Correct video is evaluated by the fraction of event frames inside their corresponding ground-truth intervals.

The ranking matters. The system must retain ranked candidates up to 100 because evaluation uses the best R-Score at ranks 1, 5, 20, 50 and 100.

## 3. Official Batch 1 audit currently known

Team audit of the supplied Batch 1 package:

- 873 videos represented by keyframes
- 177,321 keyframes
- 873 CLIP feature files
- 873 mapping CSV files
- 873 media-info JSON files
- 177,321 object JSON files

BTC data hierarchy: Video is the official competition data; Keyframes, Objects, CLIP features and Metadata are supporting/auxiliary data supplied for solution development.

## 4. Primary machine

- Windows 11 64-bit
- Intel Core i5-12450H
- 32 GB RAM
- NVIDIA GeForce RTX 4050 Laptop GPU, about 6 GB dedicated VRAM

Design for local/offline execution on one machine first. Prefer staged retrieval and candidate reduction before expensive models. Do not require a large model over the whole corpus.

## 5. Three ownership boundaries

### Person 1 — `video_pipeline/`

Owns offline ingestion and data preparation:

- video manifest
- frame/keyframe mapping
- validation of BTC CLIP features
- validation/normalization of BTC objects
- metadata normalization
- SQLite database
- FAISS indexes and internal-id mappings
- exact frame access
- optional OCR / ASR

Does **not** own query reasoning, ranking policy, task solving, or UI.

### Person 2 — `query_engine/` (Vũ)

Owns online inference:

`Natural Language Query → Query Understanding → Candidate Retrieval → Temporal Localization → Reranking → KIS/Q&A/TRAKE → Top-100 candidates`

Model choices are not frozen by the contract.

### Person 3 — `ui/`

Owns Streamlit only. UI communicates with Query Engine through FastAPI and must not import retrieval internals or access SQLite/FAISS directly.

## 6. Shared architecture

```text
Official Video + BTC supporting data
                │
                ▼
        video_pipeline (offline)
                │
                ▼
      SQLite + feature files + FAISS
                │
          DataStore contract
                ▼
          query_engine (online)
                │
             FastAPI
                │
                ▼
            Streamlit UI
```

## 7. Frozen engineering decisions

- SQLite = structured metadata/data access.
- FAISS = local vector search for the current scale.
- Raw/large features, images and videos remain files, not SQLite blobs.
- OCR and ASR are optional and must never block the baseline.
- Query input supports both structured requests and natural-language requests.
- Submission formatting is separate from retrieval logic.
- The internal system always represents ranked candidates; do not collapse to Top-1 internally.

## 8. Contract rules

Authoritative shared contracts are under `schemas/`, `data_layer/`, and `docs/`.

- `video_id` identifies a video.
- `frame_id` is the source frame identifier defined by the dataset/mapping; do not silently renumber it.
- `timestamp` and `frame_id` must remain cross-resolvable.
- Query Engine must use `DataStore`, not hard-code SQLite/FAISS details.
- UI must use FastAPI, not import Query Engine internals.
- Official submission formatting must be isolated from retrieval.
- Batch 2 must be addable through the ingestion/indexing layer without changing query semantics.

## 9. AI coding protocol

Before editing:

1. Read this file.
2. Read `docs/PROJECT_CONTEXT.md`.
3. Read `docs/ARCHITECTURE.md`.
4. Read `docs/DATA_CONTRACT.md`.
5. Read `docs/API_CONTRACT.md`.
6. Read `docs/TEAM_WORKFLOW.md`.
7. Identify the owned module and the contract(s) it touches.
8. Inspect the existing implementation before replacing it.
9. Run the relevant tests after changes.

AI assistants must not:

- invent official BTC query formats
- silently reinterpret `frame_id`
- commit videos, keyframes, feature tensors, indexes, databases, model weights, secrets, or `.env`
- rewrite another person's module to solve a local problem
- introduce a new storage/model technology without recording the decision and reason
- break a public schema/API without explicitly changing the contract

If information is missing, distinguish **official source**, **team decision**, **implementation detail**, and **hypothesis**. Never present a hypothesis as a BTC requirement.

## 10. Current runtime status

The repository contains an end-to-end **mock runtime**:

`Streamlit → FastAPI → MockQueryEngine → contract-valid result`.

This is scaffolding only. It is not a retrieval baseline and must not be used as evidence of competition performance.
