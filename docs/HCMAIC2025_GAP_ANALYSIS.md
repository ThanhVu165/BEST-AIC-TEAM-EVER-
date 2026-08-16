# HCMC AIC 2025 Reference Review — Gap Analysis

This document records ideas from `lducc/hcm-aic` that are useful as engineering patterns for AIC 2026. It is a reference-analysis document, not a claim that AIC 2026 scoring or artifacts are identical to AIC 2025.

## Reference architecture worth adopting

The strongest design pattern is a **single shared frame catalog**. In `hcm-aic`, `frames.csv` is the identity boundary for every modality: visual indexes, ASR, OCR, captions, thumbnails, and results all resolve through the same frame identity. The repository explicitly validates `FAISS ID == frames.csv frame_id` and vector-count compatibility before search.

For our system this should become:

```text
source video / organizer keyframe
        ↓
canonical frame catalog
        ↓
CLIP / BEiT-3 / Objects / OCR / ASR / metadata
        ↓
DataStore
        ↓
shared retrieval evidence
```

Our current `DataStore` + FAISS mapping is directionally correct, but we need a richer canonical catalog than only `video_id + frame_id` if the supplied Batch 1 data permits it. Preserve timestamp, keyframe ordinal, source path, and deterministic provenance.

## Missing capability: modality fusion beyond CLIP

The reference system runs independent channels:

- CLIP visual retrieval
- BEiT-3 visual retrieval
- ASR FTS5
- OCR FTS5
- caption retrieval

and fuses them using weighted Reciprocal Rank Fusion (RRF). This is materially stronger than a single cosine-similarity path when a query contains speech or visible-text clues.

Our Query Engine currently has CLIP + object evidence, but no first-class ASR/OCR channel. For AIC 2026, we should add interfaces for auxiliary evidence without coupling the query layer to storage implementations. Do not enable a modality merely because it exists: require benchmark evidence.

## Missing capability: query routing / decomposition

The reference system has an optional typed query router. It produces:

- one to three English visual descriptions;
- a speech/ASR query;
- a literal OCR query;
- selected modalities;
- explicit constraints such as count, visible text, and temporal order.

The important engineering idea is **query decomposition by evidence type**, not the specific Gemini model. This is directly applicable to our natural-language query module.

Recommended AIC 2026 abstraction:

```text
raw query
  ↓
QueryPlan
  ├── visual_queries[]
  ├── object/action constraints
  ├── OCR query
  ├── ASR query
  ├── temporal events[]
  └── required vs soft constraints
```

Keep the router optional and cacheable. Never allow it to fabricate facts or become a hard filter by default.

## Missing capability: candidate recall gate

A major lesson from the reference benchmark methodology is to measure **candidate recall before reranking**. Their diagnostics record target rank after raw RRF and after subsequent stages.

We should add explicit metrics:

```text
raw_candidate_video_recall@50
raw_candidate_video_recall@100
raw_candidate_frame/moment_recall@50
raw_candidate_frame/moment_recall@100
```

Then decide whether the problem is:

```text
candidate missing → improve retrieval / query encoder
candidate present but low-ranked → reranking
correct video but wrong moment → temporal localization
```

This should become a hard research gate before adding large models.

## Missing capability: reproducible benchmark trace

The reference repo records stage-by-stage candidate state:

```text
raw_rrf
post_verify
post_neighbors
post_local_refine
post_temporal_chain
post_semantic
post_dedup
```

For each stage it records target video/moment rank and raw hit@50/@100. It also records query-routing flags, cache state, and failure category.

Our evaluator currently scores final candidates but does not yet provide an equally detailed stage trace. Add a non-mutating trace interface so experiments answer **which stage improved or damaged ranking**.

## Missing capability: local moment refinement

The reference system uses a shortlist-then-localize pattern:

1. retrieve candidate videos globally;
2. inspect only nearby frames within a bounded window;
3. rescore local frames with CLIP/BEiT-3;
4. inject nearby ASR/OCR evidence;
5. softly move the candidate timestamp.

This is a useful baseline before a learned temporal grounder. Our current temporal module only ranks already retrieved source frames. We should add a bounded local refinement module with explicit provenance and a conservative score gain.

Important constraint: it must not claim a ground-truth boundary and must not fabricate frame IDs.

## Missing capability: ordered TRAKE chain reasoning

The reference temporal-chain implementation treats each explicit event as a separate visual query, then searches for one video containing a chronological sequence of frames with positive evidence within a bounded span. It uses a soft geometric-mean chain score and retains the ordinary candidate when no chain is available.

This is closer to the TRAKE requirement than our current common-video filtering alone. Our implementation should evolve toward:

```text
event_1 retrieval
   + event_2 retrieval
   + ...
        ↓
common-video candidates
        ↓
monotonic timestamp chain
        ↓
per-event semantic keyframe
        ↓
soft chain score
```

The chain must be soft evidence, not a hard filter that can destroy candidate recall.

## Missing capability: verification as a separate evidence stage

The reference system uses GroundingDINO to verify object/count/relation/color constraints on a small top-N shortlist. It explicitly separates retrieval from verification and uses verification as a small score contribution.

For our supplied OpenImages detections, the first implementation should use those detections before introducing GroundingDINO. GroundingDINO can later be an optional verifier for cases that OpenImages cannot resolve.

Constraints such as exact count, visible text, and ordered events should be represented explicitly and scored separately. Do not let an unreliable verifier hard-filter candidates.

## Missing capability: local text evidence integration

The reference system snaps ASR hits to the nearest catalog keyframe and collects ASR/OCR evidence within a local time window around top visual candidates. This is useful for both KIS and future Q&A.

AIC 2026 should expose:

```text
frame candidate
  ↓
local temporal window
  ├── nearby source frames
  ├── nearby ASR segments
  ├── nearby OCR records
  └── object evidence
```

This evidence should be passed downstream to VQA and TRAKE rather than rebuilding retrieval independently.

## Missing capability: immutable data release / manifest

The reference project has a `doctor` validator and release metadata including hashes, row counts, model IDs, dimensions, and coverage. It explicitly warns not to mix artifacts from different releases.

Our Batch 1 pipeline should produce a manifest containing at least:

```text
source dataset/version
catalog checksum
row count
vector dimension
model/checkpoint ID
index checksum
mapping checksum
object coverage
OCR/ASR coverage when available
build revision
```

A mismatched index/catalog should fail before retrieval.

## Missing capability: benchmark discipline

The reference repository does not treat a 40-query benchmark as a blanket promotion claim. It separates verified visual/OCR rows from ASR rows that are only artifact-aligned, and requires held-out evidence before promotion.

For AIC 2026, we should maintain:

- development benchmark;
- held-out benchmark;
- failure taxonomy;
- per-query result files;
- immutable run manifest.

Do not tune weights on the same labels used for final reporting.

## What we should NOT copy directly

1. Do not copy the AIC 2025 RRF weights or thresholds. They must be re-estimated on AIC 2026 labels.
2. Do not assume BEiT-3 is automatically superior for AIC 2026. Benchmark candidate recall and ranking first.
3. Do not introduce Gemini/GroundingDINO solely because the reference repo uses them. Query routing and verification are optional stages.
4. Do not use their `frames.csv` schema literally if Batch 1 provides a different official mapping. Preserve the same identity principle but adapt the actual fields.
5. Do not treat their benchmark percentages as AIC 2026 performance numbers.

## Priority for our branch

### P0 — before model research

- canonical frame catalog + deterministic provenance;
- strict artifact validator/manifest;
- candidate recall@50/@100 diagnostics;
- stage trace;
- Batch 1 integration.

### P1 — highest expected retrieval gain

- query decomposition / typed QueryPlan;
- auxiliary OCR/ASR interfaces if Batch 1 actually provides useful coverage;
- multimodal RRF or equivalent rank fusion;
- local moment refinement;
- ordered temporal-chain scoring for TRAKE.

### P2 — conditional research

- BEiT-3 independent candidate generator;
- BLIP/ITM semantic reranker;
- GroundingDINO verifier;
- VLM answer extractor.

The decision rule is always candidate recall first, then reranking, then temporal/answer quality.
