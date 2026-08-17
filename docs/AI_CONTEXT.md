# AI Context — AIC 2026 Video Retrieval

This is the compact source of truth for AI coding assistants working on this repository. Before changing code, read this file and `docs/AIC2026_CANONICAL_PIPELINE.md` plus the shared contracts. Do not infer missing BTC requirements from unofficial examples.

## 1. Project identity

- Competition: AIC 2026, preliminary-round Video Retrieval / Video Understanding system.
- Repository: `ThanhVu165/BEST-AIC-TEAM-EVER-`.
- Current official data available to the team: Batch 1.
- Batch 2: not supplied yet.
- Official preliminary-round query file: not supplied yet.
- Team-collected query files are reference/stress-test material only.

## 2. Canonical pipeline

The repository has **one** end-to-end pipeline. Its canonical definition is `docs/AIC2026_CANONICAL_PIPELINE.md`.

```text
Query
 -> Query Understanding
 -> Candidate Retrieval
 -> Video Ranking
 -> Coarse/Fine Temporal Localization
 -> Semantic Keyframe Alignment
 -> KIS / Q&A / TRAKE
 -> Final Ranking
 -> Top-100 Submission
```

Existing modules, scripts and model adapters are stages/implementations inside this pipeline. Do not create parallel end-to-end pipelines.

## 3. Official task semantics currently established

1. KIS / Textual Known Item Search — correct video + frame inside the ground-truth event interval.
2. Q&A — correct video + frame inside the ground-truth event interval + semantically correct answer.
3. TRAKE — correct video + one semantic keyframe per queried event; ordered events require sequence-consistent alignment.

The system must preserve ranked alternatives up to 100 because evaluation uses R@1/R@5/R@20/R@50/R@100.

## 4. Data and BTC evidence

Video is the official competition data. Keyframes, Objects, CLIP features and Metadata are supporting/auxiliary data.

Team audit of Batch 1:

- 873 videos
- 177,321 keyframes
- 873 CLIP feature files
- 873 mapping CSV files
- 873 media-info/metadata JSON files
- 177,321 object JSON files

The currently validated BTC CLIP index is 512-D with 177,321 vectors and 177,321 mapping entries.

OCR and ASR are currently absent from the checked SQLite database and are optional enrichment layers, not baseline prerequisites.

## 5. Ownership boundaries

### Person 1 — `video_pipeline/`

Offline ingestion, timeline/frame mapping, BTC CLIP/object/metadata validation, SQLite, FAISS, exact frame access and optional OCR/ASR.

### Person 2 — `query_engine/` (Vũ)

Query understanding, candidate retrieval, video ranking, temporal localization, semantic keyframe selection, multimodal reranking, KIS/Q&A/TRAKE and Top-100 ranking.

### Person 3 — `ui/`

Streamlit UI only. UI communicates through FastAPI and must not import retrieval internals or access SQLite/FAISS directly.

## 6. Frozen engineering decisions

- SQLite = structured metadata/data access.
- FAISS = local vector search for current corpus scale.
- Raw/large features, images and videos remain files, not SQLite blobs.
- BTC CLIP is the current visual retrieval baseline.
- OCR/ASR are optional and must not block the baseline.
- Internal inference always represents ranked candidates; do not collapse to Top-1.
- Submission formatting is separate from retrieval logic.
- Batch 2 must be addable through ingestion/indexing without changing query semantics.
- Model choices remain replaceable through adapters/interfaces.

## 7. AI coding protocol

Before editing:

1. Read this file.
2. Read `docs/AIC2026_CANONICAL_PIPELINE.md`.
3. Read `docs/PROJECT_CONTEXT.md`, `docs/ARCHITECTURE.md`, `docs/DATA_CONTRACT.md`, `docs/API_CONTRACT.md` and `docs/TEAM_WORKFLOW.md`.
4. Inspect the existing implementation before replacing it.
5. Identify the pipeline stage and owned module being changed.
6. Run relevant tests/benchmarks after changes.

AI assistants must not:

- invent official BTC query formats;
- silently reinterpret `frame_id`;
- commit videos, keyframes, feature tensors, indexes, databases, model weights, secrets or `.env`;
- rewrite another person's module to solve a local problem;
- introduce a second end-to-end pipeline;
- introduce a new model/database/index without recording its stage, purpose and benchmark evidence;
- present a research baseline or team hypothesis as an official BTC requirement;
- break a public schema/API without explicitly changing the contract.

When information is missing, distinguish **official source**, **team decision**, **implementation detail** and **hypothesis**.

## 8. Current status warning

The repository has retrieval/temporal/ranking scaffolding, but these are not automatically competition-complete. In particular, Q&A answer extraction, sub-10-frame temporal accuracy, semantic event-aware keyframe selection and sequence-aware TRAKE alignment still require validation/strengthening. See the canonical pipeline's `Known current gaps` section before claiming a stage is complete.
